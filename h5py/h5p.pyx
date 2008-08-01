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


"""
    HDF5 property list interface.
"""

# Pyrex compile-time imports

from utils cimport  require_tuple, convert_dims, convert_tuple, \
                    emalloc, efree, pybool, require_list

# Runtime imports
import h5

# === C API ===================================================================

cdef hid_t pdefault(PropID pid):

    if pid is None:
        return <hid_t>H5P_DEFAULT
    return pid.id

cdef object propwrap(hid_t id_in):

    clsid = H5Pget_class(id_in)
    try:
        if H5Pequal(clsid, H5P_FILE_CREATE):
            pcls = PropFCID
        elif H5Pequal(clsid, H5P_FILE_ACCESS):
            pcls = PropFAID
        elif H5Pequal(clsid, H5P_DATASET_CREATE):
            pcls = PropDCID
        elif H5Pequal(clsid, H5P_DATASET_XFER):
            pcls = PropDXID
        else:
            raise ValueError("No class found for ID %d" % id_in)

        return pcls(id_in)
    finally:
        H5Pclose_class(clsid)

cdef object lockcls(hid_t id_in):
    cdef PropClassID pid
    pid = PropClassID(id_in)
    pid._locked = 1
    return pid


# === Public constants and data structures ====================================

# Property list classes
# These need to be locked, as the library won't let you close them.
NO_CLASS       = lockcls(H5P_NO_CLASS)
FILE_CREATE    = lockcls(H5P_FILE_CREATE)
FILE_ACCESS    = lockcls(H5P_FILE_ACCESS)
DATASET_CREATE = lockcls(H5P_DATASET_CREATE)
DATASET_XFER   = lockcls(H5P_DATASET_XFER)
# MOUNT renamed in 1.8.X; removed for now

DEFAULT = None   # In the HDF5 header files this is actually 0, which is an
                 # invalid identifier.  The new strategy for default options
                 # is to make them all None, to better match the Python style
                 # for keyword arguments.


# === Property list functional API ============================================

def create(PropClassID cls not None):
    """ (PropClassID cls) => PropID
    
        Create a new property list as an instance of a class; classes are:
            FILE_CREATE
            FILE_ACCESS
            DATASET_CREATE
            DATASET_XFER
    """
    cdef hid_t newid
    newid = H5Pcreate(cls.id)
    return propwrap(newid)

# === Class API ===============================================================

cdef class PropID(ObjectID):

    """
        Base class for all property lists and classes
    """

    def equal(self, PropID plist not None):
        """ (PropID plist) => BOOL

            Compare this property list (or class) to another for equality.
        """
        return pybool(H5Pequal(self.id, plist.id))

cdef class PropInstanceID(PropID):

    """
        Base class for property list instance objects.  Provides methods which
        are common across all HDF5 property list classes.
    """

    def copy(self):
        """ () => PropList newid

            Create a new copy of an existing property list object.
        """
        return type(self)(H5Pcopy(self.id))

    def _close(self):
        """ ()
    
            Terminate access through this identifier.  You shouldn't have to
            do this manually, as propery lists are automatically deleted when
            their Python wrappers are freed.
        """
        H5Pclose(self.id)

    def get_class(self):
        """ () => PropClassID

            Determine the class of a property list object.
        """
        return PropClassID(H5Pget_class(self.id))


# === File creation ===========================================================

cdef class PropFCID(PropInstanceID):

    """
        File creation property list.
    """

    def get_version(self):
        """ () => TUPLE version_info

            Determine version information of various file attributes. 
            Elements are:

            0:  UINT Super block version number
            1:  UINT Freelist version number
            2:  UINT Symbol table version number
            3:  UINT Shared object header version number
        """
        cdef herr_t retval
        cdef unsigned int super_
        cdef unsigned int freelist
        cdef unsigned int stab
        cdef unsigned int shhdr

        H5Pget_version(self.id, &super_, &freelist, &stab, &shhdr)

        return (super_, freelist, stab, shhdr)


    def set_userblock(self, hsize_t size):
        """ (INT/LONG size)

            Set the file user block size, in bytes.  
            Must be a power of 2, and at least 512.
        """
        H5Pset_userblock(self.id, size)

    def get_userblock(self):
        """ () => LONG size

            Determine the user block size, in bytes.
        """
        cdef hsize_t size
        H5Pget_userblock(self.id, &size)
        return size

    def set_sizes(self, size_t addr, size_t size):
        """ (UINT addr, UINT size)

            Set the addressing offsets and lengths for objects 
            in an HDF5 file, in bytes.
        """
        H5Pset_sizes(self.id, addr, size)

    def get_sizes(self):
        """ () => TUPLE sizes

            Determine addressing offsets and lengths for objects in an 
            HDF5 file, in bytes.  Return value is a 2-tuple with values:

            0:  UINT Address offsets
            1:  UINT Lengths
        """
        cdef size_t addr
        cdef size_t size
        H5Pget_sizes(self.id, &addr, &size)
        return (addr, size)

    def set_sym_k(self, unsigned int ik, unsigned int lk):
        """ (INT ik, INT lk)

            Symbol table node settings.  See the HDF5 docs for H5Pset_sym_k.
        """
        H5Pset_sym_k(self.id, ik, lk)

    def get_sym_k(self):
        """ () => TUPLE settings

            Determine symbol table node settings.  See the HDF5 docs for
            H5Pget_sym_k.  Return is a 2-tuple (ik, lk).
        """
        cdef unsigned int ik
        cdef unsigned int lk
        H5Pget_sym_k(self.id, &ik, &lk)
        return (ik, lk)

    def set_istore_k(self, unsigned int ik):
        """ (UINT ik)

            See hdf5 docs for H5Pset_istore_k.
        """
        H5Pset_istore_k(self.id, ik)
    
    def get_istore_k(self):
        """ () => UINT ik

            See HDF5 docs for H5Pget_istore_k
        """
        cdef unsigned int ik
        H5Pget_istore_k(self.id, &ik)
        return ik

# === Dataset creation properties =============================================

cdef class PropDCID(PropInstanceID):

    """
        Dataset creation property list.
    """

    def set_layout(self, int layout_code):
        """ (INT layout_code)

            Set dataset storage strategy; legal values are:
                h5d.COMPACT
                h5d.CONTIGUOUS
                h5d.CHUNKED
        """
        H5Pset_layout(self.id, layout_code)
    
    def get_layout(self):
        """ () => INT layout_code

            Determine the storage strategy of a dataset; legal values are:
                h5d.COMPACT
                h5d.CONTIGUOUS
                h5d.CHUNKED
        """
        return <int>H5Pget_layout(self.id)


    def set_chunk(self, object chunksize):
        """ (TUPLE chunksize)

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
        """ () => TUPLE chunk_dimensions

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

    # === Filter functions ====================================================
    
    def set_deflate(self, unsigned int level=5):
        """ (UINT level=5)

            Enable DEFLATE (gzip) compression, at the given level.
            Valid levels are 0-9, default is 5.
        """
        H5Pset_deflate(self.id, level)
    
    def set_fletcher32(self):
        """ ()

            Enable Fletcher32 error correction on this list.
        """
        H5Pset_fletcher32(self.id)

    def set_shuffle(self):
        """ ()

            Enable to use of the shuffle filter.  Use this immediately before 
            the DEFLATE filter to increase the compression ratio.
        """
        H5Pset_shuffle(self.id)

    def set_szip(self, unsigned int options, unsigned int pixels_per_block):
        """ (UINT options, UINT pixels_per_block)

            Enable SZIP compression.  See the HDF5 docs for argument meanings, 
            and general restrictions on use of the SZIP format.
        """
        H5Pset_szip(self.id, options, pixels_per_block)

    def get_nfilters(self):
        """ () => INT

            Determine the number of filters in the pipeline.
        """
        return H5Pget_nfilters(self.id)

    def set_filter(self, int filter_code, unsigned int flags=0, object values=None):
        """ (INT filter_code, UINT flags=0, TUPLE values=None)

            Set a filter in the pipeline.  Params are:

            filter_code:
                h5z.FILTER_DEFLATE
                h5z.FILTER_SHUFFLE
                h5z.FILTER_FLETCHER32
                h5z.FILTER_SZIP

            flags:  Bit flags (h5z.FLAG_*) setting filter properties

            values: TUPLE of UINTS giving auxiliary data for the filter.
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
        """ () => BOOL

            Determine if all the filters in the pipelist are available to
            the library.
        """
        return pybool(H5Pall_filters_avail(self.id))

    def get_filter(self, int filter_idx):
        """ (UINT filter_idx) => TUPLE filter_info

            Get information about a filter, identified by its index.

            Tuple entries are:
            0: INT filter code (h5z.FILTER_*)
            1: UINT flags (h5z.FLAG_*)
            2: TUPLE of UINT values; filter aux data (16 values max)
            3: STRING name of filter (256 chars max)
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
        """ (INT filter_code)

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
        """ (INT filter_code) => TUPLE filter_info or None

            Get information about a filter, identified by its code (one
            of h5z.FILTER_*).  If the filter doesn't exist, returns None.

            Tuple entries are:
            0: UINT flags (h5z.FLAG_*)
            1: TUPLE of UINT values; filter aux data (16 values max)
            2: STRING name of filter (256 chars max)
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
        """ (INT filter_class)

            Remove a filter from the pipeline.  The class code is one of 
            h5z.FILTER_*.
        """
        H5Premove_filter(self.id, <H5Z_filter_t>filter_class)

    def set_fill_time(self, int fill_code):
        """ (INT fill_code)

            Set the fill time.  Legal values are:
             h5d.FILL_TIME_ALLOC
             h5d.FILL_TIME_NEVER
             h5d.FILL_TIME_IFSET
        """
        H5Pset_fill_time(self.id, fill_time)

    def fill_value_defined(self):
        """ () => INT fill_status

            Determine the status of the dataset fill value.  Return values are:
              h5d.FILL_VALUE_UNDEFINED
              h5d.FILL_VALUE_DEFAULT
              h5d.FILL_VALUE_USER_DEFINED
        """
        cdef H5D_fill_value_t val
        H5Pfill_value_defined(self.id, &val)
        return <int>val


# === File access =============================================================

cdef class PropFAID(PropInstanceID):

    """
        File access property list
    """

    def set_fclose_degree(self, int close_degree):
        """ (INT close_degree)

            Set the file-close degree, which determines library behavior when
            a file is closed when objects are still open.  Legal values:

            * h5f.CLOSE_WEAK
            * h5f.CLOSE_SEMI
            * h5f.CLOSE_STRONG
            * h5f.CLOSE_DEFAULT
        """
        H5Pset_fclose_degree(self.id, <H5F_close_degree_t>close_degree)

    def get_fclose_degree(self):
        """ () => INT close_degree

            Get the file-close degree, which determines library behavior when
            a file is closed when objects are still open.  Legal values:

            * h5f.CLOSE_WEAK
            * h5f.CLOSE_SEMI
            * h5f.CLOSE_STRONG
            * h5f.CLOSE_DEFAULT
        """
        cdef H5F_close_degree_t deg
        H5Pget_fclose_degree(self.id, &deg)
        return deg

    def set_fapl_core(self, size_t increment=1024*1024, hbool_t backing_store=0):
        """ (UINT increment=1M, BOOL backing_store=False)

            Use the CORE (memory-resident) file driver.
            increment:      Chunk size for new memory requests (default 1 meg)
            backing_store:  If True, write the memory contents to disk when
                            the file is closed.
        """
        H5Pset_fapl_core(self.id, increment, backing_store)

    def get_fapl_core(self):
        """ () => TUPLE core_settings

            Determine settings for the CORE (memory-resident) file driver.
            Tuple entries are:
            0: UINT "increment": Chunk size for new memory requests
            1: BOOL "backing_store": If True, write the memory contents to 
                                     disk when the file is closed.
        """
        cdef size_t increment
        cdef hbool_t backing_store
        H5Pget_fapl_core(self.id, &increment, &backing_store)
        return (increment, pybool(backing_store))

    def set_fapl_family(self, hsize_t memb_size, PropID memb_fapl=None):
        """ (UINT memb_size, PropFAID memb_fapl=None)

            Set up the family driver.
            memb_size:  Member file size
            memb_fapl:  File access property list for each member access
        """
        cdef hid_t plist_id
        plist_id = pdefault(memb_fapl)
        H5Pset_fapl(self.id, memb_size, plist_id)

    def get_fapl_family(self):
        """ () => TUPLE info

            Determine family driver settings. Tuple values are:
            0: UINT memb_size
            1: PropFAID memb_fapl or None
        """
        cdef hid_t mfapl_id
        cdef hsize_t msize
        cdef PropFAID plist
        plist = None

        H5Pget_fapl_family(self.id, &msize, &mfapl_id)

        if mfapl_id > 0:
            plist = PropFAID(mfapl_id)

        return (msize, plist)

    def set_fapl_log(self, char* logfile, unsigned int flags, size_t buf_size):
        """ (STRING logfile, UINT flags, UINT buf_size)

            Enable the use of the logging driver.  See the HDF5 documentation
            for details.  Flag constants are stored in module h5fd.
        """
        H5Pset_fapl_log(self.id, logfile, flags, buf_size)

    def set_fapl_sec2(self):
        """ ()

            Select the "section-2" driver (h5fd.SEC2).
        """
        H5Pset_fapl_sec2(self.id)

    def set_fapl_stdio(self):
        """ ()

            Select the "stdio" driver (h5fd.STDIO)
        """
        H5Pset_fapl_stdio(self.id)

    def get_driver(self):
        """ () => INT driver code

            Return an integer identifier for the driver used by this list.
            Although HDF5 implements these as full-fledged objects, they are
            treated as integers by Python.  Built-in drivers identifiers are
            listed in module h5fd; they are:
                CORE
                FAMILY
                LOG
                MPIO
                MULTI
                SEC2
                STDIO
        """
        return H5Pget_driver(self.id)

    def set_cache(self, int mdc, int rdcc, size_t rdcc_nbytes, double rdcc_w0):
        """ (INT mdc, INT rdcc, UINT rdcc_nbytes, DOUBLE rdcc_w0)

            Set the metadata (mdc) and raw data chunk (rdcc) cache properties.
            See the HDF5 docs for a full explanation.
        """
        H5Pset_cache(self.id, mdc, rdcc, rdcc_nbytes, rdcc_w0)

    def get_cache(self):
        """ () => TUPLE cache info

            Get the metadata and raw data chunk cache settings.  See the HDF5
            docs for element definitions.  Return is a 4-tuple with entries:

            1. INT mdc              Number of metadata objects
            2. INT rdcc             Number of raw data chunks
            3. UINT rdcc_nbytes     Size of raw data cache
            4. DOUBLE rdcc_w0       Preemption policy for data cache.
        """
        cdef int mdc, rdcc
        cdef size_t rdcc_nbytes
        cdef double w0

        H5Pget_cache(self.id, &mdc, &rdcc, &rdcc_nbytes, &w0)
        return (mdc, rdcc, rdcc_nbytes, w0)







