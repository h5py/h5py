# cython: language_level=3
from numpy cimport ndarray, npy_intp, PyArray_SimpleNew, PyArray_DATA, import_array
from cpython cimport PyIndex_Check, PyNumber_Index

import numpy as np
from .defs cimport *
from .h5d cimport DatasetID
from .h5t cimport TypeID, typewrap, py_create
from .utils cimport emalloc, efree

import_array()

cdef class Reader:
    cdef hid_t dataset, space
    cdef TypeID h5_memory_datatype
    cdef int rank, np_typenum
    cdef hsize_t* dims
    cdef hsize_t* start
    cdef hsize_t* stride
    cdef hsize_t* count
    cdef bint* scalar

    def __cinit__(self, DatasetID dsid):
        self.dataset = dsid.id
        self.space = H5Dget_space(self.dataset)
        self.rank = H5Sget_simple_extent_ndims(self.space)

        # HDF5 can use e.g. custom float datatypes which don't have an exact
        # match in numpy. Translating it to a numpy dtype chooses the smallest
        # dtype which won't lose any data, then we translate that back to a
        # HDF5 datatype (h5_memory_datatype).
        h5_stored_datatype = typewrap(H5Dget_type(self.dataset))
        np_dtype = h5_stored_datatype.py_dtype()
        self.np_typenum = np_dtype.num
        self.h5_memory_datatype = py_create(np_dtype)

        self.dims = <hsize_t*>emalloc(sizeof(hsize_t) * self.rank)
        self.start = <hsize_t*>emalloc(sizeof(hsize_t) * self.rank)
        self.stride = <hsize_t*>emalloc(sizeof(hsize_t) * self.rank)
        self.count = <hsize_t*>emalloc(sizeof(hsize_t) * self.rank)
        self.scalar = <bint*>emalloc(sizeof(bint) * self.rank)

        H5Sget_simple_extent_dims(self.space, self.dims, NULL)

    def __dealloc__(self):
        efree(self.dims)
        efree(self.start)
        efree(self.stride)
        efree(self.count)
        efree(self.scalar)

    cdef bint apply_args(self, tuple args) except 0:
        cdef:
            int nargs, ellipsis_ix, array_ix = -1
            bint seen_ellipsis = False
            int dim_ix = 0
            hsize_t l
            ndarray array_arg

        # If no explicit ellipsis, implicit ellipsis is after args
        nargs = ellipsis_ix = len(args)

        for a in args:
            if a is Ellipsis:
                # [...] - Ellipsis (fill any unspecified dimensions here)
                if seen_ellipsis:
                    raise ValueError("Only one ellipsis may be used.")
                seen_ellipsis = True

                ellipsis_ix = dim_ix
                nargs -= 1  # Don't count the ... itself
                if nargs > self.rank:
                    raise ValueError(f"{nargs} indexing arguments for {self.rank} dimensions")
                dim_ix += self.rank - nargs  # Skip ahead to the remaining dimensions
                continue

            if dim_ix >= self.rank:
                raise ValueError(f"{nargs} indexing arguments for {self.rank} dimensions")

            # Length of the relevant dimension
            l = self.dims[dim_ix]

            # [0:10] - slicing
            if isinstance(a, slice):
                start, stop, step = a.indices(l)
                # Now if step > 0, then start and stop are in [0, length];
                # if step < 0, they are in [-1, length - 1] (Python 2.6b2 and later;
                # Python issue 3004).

                if step < 1:
                    raise ValueError("Step must be >= 1 (got %d)" % step)
                if stop < start:
                    # list/tuple and numpy consider stop < start to be an empty selection
                    start, count, step = 0, 0, 1
                else:
                    count = 1 + (stop - start - 1) // step

                self.start[dim_ix] = start
                self.stride[dim_ix] = step
                self.count[dim_ix] = count
                self.scalar[dim_ix] = False

            # [0] - simple integer indices
            elif PyIndex_Check(a):
                a = PyNumber_Index(a)
                if a < 0:
                    a += l

                if not 0 <= a < l:
                    if l == 0:
                        msg = f"Index ({a}) out of range for empty dimension"
                    else:
                        msg = f"Index ({a}) out of range for (0-{l-1})"
                    raise IndexError(msg)

                self.start[dim_ix] = a
                self.stride[dim_ix] = 1
                self.count[dim_ix] = 1
                self.scalar[dim_ix] = True

            # [[0, 2, 10]] - list/array of indices ('fancy indexing')
            elif isinstance(a, (list, np.ndarray)):
                a = np.asarray(a)
                if a.ndim != 1:
                    raise TypeError("Only 1D arrays allowed for fancy indexing")
                if not np.issubdtype(a.dtype, np.integer):
                    raise TypeError("Indexing arrays must have integer dtypes")
                if array_ix != -1:
                    raise TypeError("Only one indexing vector or array is currently allowed for fancy indexing")

                a[a < 0] += l
                if np.any((a < 0) | (a > l)):
                    if l == 0:
                        msg = "Fancy indexing out of range for empty dimension"
                    else:
                        msg = f"Fancy indexing our of range for (0-{l-1})"
                    raise IndexError(msg)

                if np.any(np.diff(a) <= 0):
                    raise TypeError("Indexing elements must be in increasing order")

                array_ix = dim_ix
                array_arg = a
                self.start[dim_ix] = 0
                self.stride[dim_ix] = 1
                self.count[dim_ix] = a.shape[0]
                self.scalar[dim_ix] = False

            else:
                raise TypeError("Simple selection can't process %r" % a)

            dim_ix += 1

        if nargs < self.rank:
            # Fill in ellipsis or trailing dimensions
            ellipsis_end = ellipsis_ix + (self.rank - nargs)
            for dim_ix in range(ellipsis_ix, ellipsis_end):
                self.start[dim_ix] = 0
                self.stride[dim_ix] = 1
                self.count[dim_ix] = self.dims[dim_ix]
                self.scalar[dim_ix] = False

        if nargs == 0:
            H5Sselect_all(self.space)
        elif array_ix != -1:
            self.select_fancy(array_ix, array_arg)
        else:
            H5Sselect_hyperslab(self.space, H5S_SELECT_SET, self.start, self.stride, self.count, NULL)
        return True

    cdef select_fancy(self, int array_ix, ndarray array_arg):
        """Apply a 'fancy' selection (array of indices) to the dataspace"""
        cdef hsize_t* tmp_start
        cdef hsize_t* tmp_count
        cdef uint64_t i

        H5Sselect_none(self.space)

        tmp_start = <hsize_t*>emalloc(sizeof(hsize_t) * self.rank)
        tmp_count = <hsize_t*>emalloc(sizeof(hsize_t) * self.rank)
        try:
            memcpy(tmp_start, self.start, sizeof(hsize_t) * self.rank)
            memcpy(tmp_count, self.count, sizeof(hsize_t) * self.rank)
            tmp_count[array_ix] = 1

            # Iterate over the array of indices, add each hyperslab to the selection
            for i in array_arg:
                tmp_start[array_ix] = i
                H5Sselect_hyperslab(self.space, H5S_SELECT_OR, tmp_start, self.stride, tmp_count, NULL)
        finally:
            efree(tmp_start)
            efree(tmp_count)

    cdef ndarray make_array(self):
        cdef int i, arr_rank = 0
        cdef npy_intp* arr_shape

        arr_shape = <npy_intp*>emalloc(sizeof(npy_intp) * self.rank)
        try:
            # Copy any non-scalar selection dimensions for the array shape
            for i in range(self.rank):
                if not self.scalar[i]:
                    arr_shape[arr_rank] = self.count[i]
                    arr_rank += 1

            arr = PyArray_SimpleNew(arr_rank, arr_shape, self.np_typenum)
        finally:
            efree(arr_shape)

        return arr

    def read(self, tuple args):
        cdef void* buf
        cdef ndarray arr
        cdef hid_t mspace
        cdef int i

        self.apply_args(args)

        arr = self.make_array()
        buf = PyArray_DATA(arr)

        mspace = H5Screate_simple(self.rank, self.count, NULL)

        H5Dread(self.dataset, self.h5_memory_datatype.id, mspace, self.space, H5P_DEFAULT, buf)

        if arr.ndim == 0:
            return arr[()]
        else:
            return arr
