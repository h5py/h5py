#+
# 
# This file is part of h5py, a low-level Python interface to the HDF5 library.
# 
# Copyright (C) 2008 Andrew Collette
# http://h5py.alfven.org
# License: BSD  (See LICENSE.txt for full license)
# 
# $Date$
# 
#-

cdef class PropDCID(PropCreateID):

    """
        Dataset creation property list.
    """

    
    def set_layout(self, int layout_code):
        """(INT layout_code)

        Set dataset storage strategy; legal values are:

        - h5d.COMPACT
        - h5d.CONTIGUOUS
        - h5d.CHUNKED
        """
        H5Pset_layout(self.id, layout_code)
    
    
    def get_layout(self):
        """() => INT layout_code

        Determine the storage strategy of a dataset; legal values are:

        - h5d.COMPACT
        - h5d.CONTIGUOUS
        - h5d.CHUNKED
        """
        return <int>H5Pget_layout(self.id)

    
    def set_chunk(self, object chunksize):
        """(TUPLE chunksize)

        Set the dataset chunk size.  It's up to you to provide 
        values which are compatible with your dataset.
        """
        cdef int rank
        cdef hsize_t* dims
        dims = NULL

        require_tuple(chunksize, 0, -1, "chunksize")
        rank = len(chunksize)

        dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
        try:
            convert_tuple(chunksize, dims, rank)
            H5Pset_chunk(self.id, rank, dims)
        finally:
            efree(dims)

    
    def get_chunk(self):
        """() => TUPLE chunk_dimensions

        Obtain the dataset chunk size, as a tuple.
        """
        cdef int rank
        cdef hsize_t *dims

        rank = H5Pget_chunk(self.id, 0, NULL)
        assert rank >= 0
        dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)

        try:
            H5Pget_chunk(self.id, rank, dims)
            tpl = convert_dims(dims, rank)
            return tpl
        finally:
            efree(dims)

    
    def set_fill_value(self, ndarray value not None):
        """(NDARRAY value)

        Set the dataset fill value.  The object provided should be an
        0-dimensional NumPy array; otherwise, the value will be read from
        the first element.
        """
        cdef TypeID tid

        check_numpy_read(value, -1)
        tid = py_create(value.dtype)        
        H5Pset_fill_value(self.id, tid.id, value.data)

    
    def get_fill_value(self, ndarray value not None):
        """(NDARRAY value)

        Read the dataset fill value into a NumPy array.  It will be
        converted to match the array dtype.  If the array has nonzero
        rank, only the first element will contain the value.
        """
        cdef TypeID tid

        check_numpy_write(value, -1)
        tid = py_create(value.dtype)
        H5Pget_fill_value(self.id, tid.id, value.data)

    
    def fill_value_defined(self):
        """() => INT fill_status

        Determine the status of the dataset fill value.  Return values are:

        - h5d.FILL_VALUE_UNDEFINED
        - h5d.FILL_VALUE_DEFAULT
        - h5d.FILL_VALUE_USER_DEFINED
        """
        cdef H5D_fill_value_t val
        H5Pfill_value_defined(self.id, &val)
        return <int>val

    
    def set_fill_time(self, int fill_time):
        """(INT fill_time)

        Define when fill values are written to the dataset.  Legal
        values (defined in module h5d) are:

        - h5d.FILL_TIME_ALLOC
        - h5d.FILL_TIME_NEVER
        - h5d.FILL_TIME_IFSET
        """
        H5Pset_fill_time(self.id, <H5D_fill_time_t>fill_time)

    
    def get_fill_time(self):
        """ () => INT

        Determine when fill values are written to the dataset.  Legal
        values (defined in module h5d) are:

        - h5d.FILL_TIME_ALLOC
        - h5d.FILL_TIME_NEVER
        - h5d.FILL_TIME_IFSET
        """
        cdef H5D_fill_time_t fill_time
        H5Pget_fill_time(self.id, &fill_time)
        return <int>fill_time

    
    def set_alloc_time(self, int alloc_time):
        """(INT alloc_time)

        Set the storage space allocation time.  One of h5d.ALLOC_TIME*.
        """
        H5Pset_alloc_time(self.id, <H5D_alloc_time_t>alloc_time)

    
    def get_alloc_time(self):
        """() => INT alloc_time

        Get the storage space allocation time.  One of h5d.ALLOC_TIME*.
        """
        cdef H5D_alloc_time_t alloc_time
        H5Pget_alloc_time(self.id, &alloc_time)
        return <int>alloc_time


    # === Filter functions ====================================================
    
    
    def set_filter(self, int filter_code, unsigned int flags=0, object values=None):
        """(INT filter_code, UINT flags=0, TUPLE values=None)

        Set a filter in the pipeline.  Params are:

        filter_code
            One of the following:

            - h5z.FILTER_DEFLATE
            - h5z.FILTER_SHUFFLE
            - h5z.FILTER_FLETCHER32
            - h5z.FILTER_SZIP

        flags
            Bit flags (h5z.FLAG*) setting filter properties

        values
            TUPLE of UINTs giving auxiliary data for the filter
        """
        cdef size_t nelements
        cdef unsigned int *cd_values
        cdef int i
        cd_values = NULL

        require_tuple(values, 1, -1, "values")
        
        try:
            if values is None or len(values) == 0:
                nelements = 0
                cd_values = NULL
            else:
                nelements = len(values)
                cd_values = <unsigned int*>emalloc(sizeof(unsigned int)*nelements)

                for i from 0<=i<nelements:
                    cd_values[i] = int(values[i])
            
            H5Pset_filter(self.id, <H5Z_filter_t>filter_code, flags, nelements, cd_values)
        finally:
            efree(cd_values)

    
    def all_filters_avail(self):
        """() => BOOL

        Determine if all the filters in the pipelist are available to
        the library.
        """
        return <bint>(H5Pall_filters_avail(self.id))

    
    def get_nfilters(self):
        """() => INT

        Determine the number of filters in the pipeline.
        """
        return H5Pget_nfilters(self.id)

    
    def get_filter(self, int filter_idx):
        """(UINT filter_idx) => TUPLE filter_info

        Get information about a filter, identified by its index.  Tuple
        elements are:

        0. INT filter code (h5z.FILTER*)
        1. UINT flags (h5z.FLAG*)
        2. TUPLE of UINT values; filter aux data (16 values max)
        3. STRING name of filter (256 chars max)
        """
        cdef list vlist
        cdef int filter_code
        cdef unsigned int flags
        cdef size_t nelements
        cdef unsigned int cd_values[16]
        cdef char name[257]
        cdef int i
        nelements = 16 # HDF5 library actually complains if this is too big.

        if filter_idx < 0:
            raise ValueError("Filter index must be a non-negative integer")

        filter_code = <int>H5Pget_filter(self.id, filter_idx, &flags,
                                         &nelements, cd_values, 256, name)
        name[256] = c'\0'  # in case it's > 256 chars

        vlist = []
        for i from 0<=i<nelements:
            vlist.append(cd_values[i])

        return (filter_code, flags, tuple(vlist), name)

    
    def _has_filter(self, int filter_code):
        """(INT filter_code)

        Slow & stupid method to determine if a filter is used in this
        property list.  Used because the HDF5 function H5Pget_filter_by_id
        is broken.
        """
        cdef int nfilters
        nfilters = self.get_nfilters()
        for i from 0<=i<nfilters:
            if self.get_filter(i)[0] == filter_code:
                return True
        return False

    
    def get_filter_by_id(self, int filter_code):
        """(INT filter_code) => TUPLE filter_info or None

        Get information about a filter, identified by its code (one
        of h5z.FILTER*).  If the filter doesn't exist, returns None.
        Tuple elements are:

        0. UINT flags (h5z.FLAG*)
        1. TUPLE of UINT values; filter aux data (16 values max)
        2. STRING name of filter (256 chars max)
        """
        cdef list vlist
        cdef unsigned int flags
        cdef size_t nelements
        cdef unsigned int cd_values[16]
        cdef char name[257]
        cdef herr_t retval
        cdef int i
        nelements = 16 # HDF5 library actually complains if this is too big.

        if not self._has_filter(filter_code):
            # Avoid library segfault
            return None

        retval = H5Pget_filter_by_id(self.id, <H5Z_filter_t>filter_code,
                                     &flags, &nelements, cd_values, 256, name)
        assert nelements <= 16

        name[256] = c'\0'  # In case HDF5 doesn't terminate it properly

        vlist = []
        for i from 0<=i<nelements:
            vlist.append(cd_values[i])

        return (flags, tuple(vlist), name)

    
    def remove_filter(self, int filter_class):
        """(INT filter_class)

        Remove a filter from the pipeline.  The class code is one of 
        h5z.FILTER*.
        """
        H5Premove_filter(self.id, <H5Z_filter_t>filter_class)

    
    def set_deflate(self, unsigned int level=5):
        """(UINT level=5)

        Enable deflate (gzip) compression, at the given level.
        Valid levels are 0-9, default is 5.
        """
        H5Pset_deflate(self.id, level)

    
    def set_fletcher32(self):
        """()

        Enable Fletcher32 error correction on this list.
        """
        H5Pset_fletcher32(self.id)

    
    def set_shuffle(self):
        """()

        Enable to use of the shuffle filter.  Use this immediately before 
        the deflate filter to increase the compression ratio.
        """
        H5Pset_shuffle(self.id)

    
    def set_szip(self, unsigned int options, unsigned int pixels_per_block):
        """(UINT options, UINT pixels_per_block)

        Enable SZIP compression.  See the HDF5 docs for argument meanings, 
        and general restrictions on use of the SZIP format.
        """
        H5Pset_szip(self.id, options, pixels_per_block)




