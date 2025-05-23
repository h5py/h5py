"""Support for NpyStrings - NumPy's native variable-width strings.

This module requires NumPy >=2.0 to be imported. To allow the rest of h5py
to work with NumPy 1.x installed, this module must be imported conditionally
and can't be cimport'ed.
"""
include "config.pxi"

from .defs cimport *
from .utils cimport emalloc
from numpy cimport PyArray_Descr
import numpy as np

assert NUMPY_BUILD_VERSION >= '2.0'  # See pyproject.toml

# =========================================================================
# Scatter/gather routines for vlen strings to/from NumPy StringDType arrays
# See also: _proxy.pyx::h5py_copy, h5py_scatter_cb, h5py_gather_cb

IF NUMPY_BUILD_VERSION < '2.3':
    # Backport NpyStrings from 2.3 to 2.0
    # The C API was available since 2.0, but the Cython API was added in 2.3.
    # This is a copy-paste from numpy/__init__.pxd of NumPy 2.3
    cdef extern from "numpy/ndarraytypes.h":
        ctypedef struct npy_string_allocator:
            pass

        ctypedef struct npy_packed_static_string:
            pass

        ctypedef struct npy_static_string:
            size_t size
            const char *buf

        ctypedef struct PyArray_StringDTypeObject:
            PyArray_Descr base
            PyObject *na_object
            char coerce
            char has_nan_na
            char has_string_na
            char array_owned
            npy_static_string default_string
            npy_static_string na_name
            npy_string_allocator *allocator

    cdef extern from "numpy/arrayobject.h":
        npy_string_allocator *NpyString_acquire_allocator(const PyArray_StringDTypeObject *descr)
        void NpyString_acquire_allocators(size_t n_descriptors, PyArray_Descr *const descrs[], npy_string_allocator *allocators[])
        void NpyString_release_allocator(npy_string_allocator *allocator)
        void NpyString_release_allocators(size_t length, npy_string_allocator *allocators[])
        int NpyString_load(npy_string_allocator *allocator, const npy_packed_static_string *packed_string, npy_static_string *unpacked_string)
        int NpyString_pack_null(npy_string_allocator *allocator, npy_packed_static_string *packed_string)
        int NpyString_pack(npy_string_allocator *allocator, npy_packed_static_string *packed_string, const char *buf, size_t size)

ELSE:
    from numpy cimport (
        npy_string_allocator,
        npy_static_string,
        npy_packed_static_string,
        NpyString_pack,
        NpyString_load,
        NpyString_acquire_allocator,
        NpyString_release_allocator,
        PyArray_StringDTypeObject,
    )

# Can't call sizeof(npy_packed_static_string) because the struct is
# internal to NumPy
cdef size_t SIZEOF_NPY_PACKED_STATIC_STRING = np.dtype("T").itemsize

ctypedef struct npystrings_pack_t:
    size_t i
    npy_string_allocator *allocator
    const char **contig

ctypedef struct npystrings_unpack_t:
    size_t i
    npy_string_allocator *allocator
    npy_static_string *unpacked


cdef herr_t npystrings_pack_cb(
    void* elem, hid_t type_id, unsigned ndim,
    const hsize_t *point, void *operator_data
) except -1:
    cdef npystrings_pack_t* info = <npystrings_pack_t*>operator_data
    cdef const char* buf = info[0].contig[info[0].i]
    # Deep copy char* from h5py into the NumPy array
    res = NpyString_pack(
        info[0].allocator,
        <npy_packed_static_string*>elem,
        buf,
        strlen(buf),
    )
    info[0].i += 1
    return res


def npystrings_pack(hid_t space, size_t contig, size_t noncontig, size_t descr):
    """Convert a zero-terminated char**, which is the in-memory representation
    for a HDF5 variable-width string dataset, into a NpyString array
    (NumPy's native variable-width strings dtype).

    Parameters
    ----------
    space : hid_t
        HDF5 dataspace ID for the output non-contiguous buffer
        containing packed NpyStrings
    contig : void * (actually a char **)
        Input buffer for the contiguous zero-terminated char **
    noncontig : void *
        Output buffer for the non-contiguous packed NpyStrings
    descr : PyArray_Descr * (actually a PyArray_StringDTypeObject *)
        NumPy dtype descriptor for the NpyStrings
    npoints : size_t
        Number of points to copy

    Notes
    -----
    This function deep-copies the input zero-terminated strings.
    Memory management for the outputs is handled by NumPy.

    This function defines a pure-python API interface to _proxy.pyx.
    This is necessary to allow this module to be conditionally imported,
    allowing NumPy 1.x to continue working everywhere else.
    """
    _npystrings_pack(space, <void *>contig, <void *>noncontig, <PyArray_Descr *>descr)


cdef void _npystrings_pack(hid_t space, void *contig, void *noncontig,
                           PyArray_Descr *descr):
    """Cython API interface of npystrings_pack"""
    cdef npystrings_pack_t info
    info.i = 0
    info.contig = <const char**>contig
    info.allocator = NpyString_acquire_allocator(<PyArray_StringDTypeObject *>descr)
    if info.allocator is NULL:
        raise RuntimeError("Failed to acquire string allocator")

    # H5Diterate needs a tid of the correct size.
    # Disregard mtype from _proxy::dset_rw.
    # The memory type is actually npy_packed_static_string
    # (an opaque struct of arbitrary size)
    # and not HDF5 variable length strings (char*).
    tid = H5Tcreate(H5T_OPAQUE, SIZEOF_NPY_PACKED_STATIC_STRING)

    # Read char*[] (zero-terminated) from h5py
    # and deep-copy to npy_packed_static_string[] for NumPy
    H5Diterate(noncontig, tid, space, npystrings_pack_cb, &info)
    NpyString_release_allocator(info.allocator)
    H5Tclose(tid)


cdef herr_t npystrings_unpack_cb(
    void *elem, hid_t type_id, unsigned ndim,
    const hsize_t *point, void *operator_data
) except -1:
    cdef npystrings_unpack_t *info = <npystrings_unpack_t *>operator_data
    cdef npy_static_string *unpacked = info[0].unpacked + info[0].i
    # Obtain a reference to the string (NOT zero-terminated) and its size
    res = NpyString_load(info[0].allocator, <npy_packed_static_string*>elem, unpacked)
    info[0].i += 1
    # res == -1 if unpacking the string fails, 1 if packed_string is the null string,
    # and 0 otherwise.
    return -1 if res == -1 else 0


def npystrings_unpack(hid_t space, size_t contig, size_t noncontig, size_t descr,
                      size_t npoints):
    """Convert a NpyString array (NumPy's native variable-width strings dtype)
    to a zero-terminated char**, which is the in-memory representation for a
    HDF5 variable-width string dataset.

    Parameters
    ----------
    space : hid_t
        HDF5 dataspace ID for the input non-contiguous buffer
        containing packed NpyStrings
    contig : void * (actually a char **)
        Output buffer for the contiguous zero-terminated char **
    noncontig : void *
        Input buffer for the non-contiguous packed NpyStrings
    descr : PyArray_Descr * (actually a PyArray_StringDTypeObject *)
        NumPy dtype descriptor for the NpyStrings
    npoints : size_t
        Number of points to copy

    Returns
    -------
    char *
        A temporary buffer containing the zero-terminated strings.
        This buffer must be freed after writing to disk.

        Note that this always the same as contig[0] when this
        function returns, but H5Tconvert may change the contents
        of contig[0] in place later on.

    Notes
    -----
    This function defines a pure-python API interface to _proxy.pyx.
    This is necessary to allow this module to be conditionally imported,
    allowing NumPy 1.x to continue working everywhere else.
    """
    return <size_t>_npystrings_unpack(space, <void *>contig, <void *>noncontig,
                                      <PyArray_Descr *>descr, npoints)


cdef char * _npystrings_unpack(hid_t space, void *contig, void *noncontig,
                               PyArray_Descr *descr, size_t npoints):
    """Cython API interface of npystrings_unpack"""
    cdef npystrings_unpack_t info
    cdef size_t total_size
    cdef size_t cur_size
    cdef char *zero_terminated = NULL
    cdef char *zero_terminated_cur = NULL
    info.unpacked = NULL
    info.i = 0

    info.allocator = NpyString_acquire_allocator(<PyArray_StringDTypeObject *>descr)
    if info.allocator is NULL:
        raise RuntimeError("Failed to acquire string allocator")

    tid = H5Tcreate(H5T_OPAQUE, SIZEOF_NPY_PACKED_STATIC_STRING)
    try:
        # Multiple steps needed:
        # 1. Read npy_packed_static_string[] from NumPy and unpack
        #    to npy_static_string[]; which is
        #    {const char* buf, size_t size}[] - NOT zero-terminated
        info.unpacked = <npy_static_string*>emalloc(
            npoints * sizeof(npy_static_string)
        )
        H5Diterate(noncontig, tid, space, npystrings_unpack_cb, &info)
        assert info.i == npoints

        # 2. Calculate total size of strings with zero termination
        total_size = npoints  # zero termination characters
        for i in range(npoints):
            total_size += info.unpacked[i].size

        # 3. Copy to temporary buffer which is a concatenation of
        #    zero-terminated char* and point to it from the
        #    output char*[] for h5py
        zero_terminated_buf = <char *>emalloc(total_size)
        zero_terminated_cur = zero_terminated_buf
        for i in range(npoints):
            cur_size = info.unpacked[i].size
            memcpy(zero_terminated_cur, info.unpacked[i].buf, cur_size)
            zero_terminated_cur[cur_size] = 0
            (<char **>contig)[i] = zero_terminated_cur
            zero_terminated_cur += cur_size + 1

        return zero_terminated_buf

    finally:
        free(info.unpacked)
        NpyString_release_allocator(info.allocator)
        H5Tclose(tid)
        # after H5Dwrite, user must free(zero_terminated_buf)
