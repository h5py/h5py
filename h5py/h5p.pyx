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
                    emalloc, efree, pybool

# Runtime imports
import h5
import h5t
from h5 import DDict

cdef object lockid(hid_t id_in):
    cdef PropClassID pid
    pid = PropClassID(id_in)
    pid._locked = 1
    return pid

# === Public constants and data structures ====================================

# Property list classes
# These need to be locked, as the library won't let you close them.
NO_CLASS       = lockid(H5P_NO_CLASS)
FILE_CREATE    = lockid(H5P_FILE_CREATE)
FILE_ACCESS    = lockid(H5P_FILE_ACCESS)
DATASET_CREATE = lockid(H5P_DATASET_CREATE)
DATASET_XFER   = lockid(H5P_DATASET_XFER)
MOUNT          = lockid(H5P_MOUNT)

DEFAULT = lockid(H5P_DEFAULT)  # really 0 but whatever

_classmapper = { H5P_FILE_CREATE: PropFCID,
                 H5P_FILE_ACCESS: PropFAID,
                 H5P_DATASET_CREATE: PropDCID,
                 H5P_DATASET_XFER: PropDXID,
                 H5P_MOUNT: PropMID }

# === C API and extension types ===============================================

cdef hid_t pdefault(PropID pid):

    if pid is None:
        return <hid_t>H5P_DEFAULT
    
    return pid.id

cdef class PropID(ObjectID):
    
    """ Base class for all operations which are valid on both property list 
        instances and classes.
    """
    pass

# === Property list HDF5 classes ==============================================

cdef class PropClassID(PropID):
    pass

# === Property list HDF5 instances ============================================

def create(PropClassID cls not None):
    """ (PropClassID cls) => PropID
    
        Create a new property list as an instance of a class; classes are:
            FILE_CREATE
            FILE_ACCESS
            DATASET_CREATE
            DATASET_XFER
            MOUNT
    """
    try:
        type_ = _classmapper[cls.id]
    except KeyError:
        raise ValueError("Invalid class")

    return type_(H5Pcreate(cls.id))

cdef class PropInstanceID(PropID):

    """
        Base class for property list instance objects
    """

    def copy(self):
        """ () => PropList new_property_list_id

            Create a new copy of an existing property list object.
        """
        return type(self)(H5Pcopy(self.id))

    def close(self):
        H5Pclose(self.id)

    def get_class(self):
        """ () => PropClassID

            Determine the class of a property list object.
        """
        return PropClassID(H5Pget_class(self.id))

    def equal(self, PropID plist not None):
        """ (PropID plist) => BOOL

            Compare this property list to another for equality.
        """
        return pybool(H5Pequal(self.id, plist.id))

# === File creation ===========================================================

cdef class PropFCID(PropInstanceID):

    """
        Represents a file creation property list
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
        """ () => TUPLE sizes    [File creation]

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
        """ (UINT ik)    [File creation]

            See hdf5 docs for H5Pset_istore_k.
        """
        H5Pset_istore_k(self.id, ik)
    
    def get_istore_k(self):
        """ () => UINT ik    [File creation]

            See HDF5 docs for H5Pget_istore_k
        """
        cdef unsigned int ik
        H5Pget_istore_k(self.id, &ik)
        return ik

# === Dataset creation properties =============================================

cdef class PropDCID(PropInstanceID):

    """
        Represents a dataset creation property list
    """

    def set_layout(self, int layout_code):
        """ (INT layout_code)    [Dataset creation]

            Set dataset storage strategy; legal values are:
            * h5d.COMPACT
            * h5d.CONTIGUOUS
            * h5d.CHUNKED
        """
        H5Pset_layout(self.id, layout_code)
    
    def get_layout(self):
        """ () => INT layout_code   [Dataset creation]

            Determine the storage strategy of a dataset; legal values are:
            * h5d.COMPACT
            * h5d.CONTIGUOUS
            * h5d.CHUNKED
        """
        return <int>H5Pget_layout(self.id)

    def set_chunk(self, object chunksize):
        """ (INT plist_id, TUPLE chunksize)    [Dataset creation]

            Set the dataset chunk size.  It's up to you to provide 
            values which are compatible with your dataset.
        """
        cdef herr_t retval
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
        """ () => TUPLE chunk_dimensions    [Dataset creation]

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

# === Filter functions ========================================================

    def set_deflate(self, unsigned int level=5):
        """ (UINT level=5)    [Dataset creation]

            Enable DEFLATE (gzip) compression, at the given level.
            Valid levels are 0-9, default is 5.
        """
        H5Pset_deflate(self.id, level)
    
    def set_fletcher32(self):
        """ ()    [Dataset creation]

            Enable Fletcher32 error correction on an existing list.
        """
        H5Pset_fletcher32(self.id)

    def set_shuffle(self):
        """ ()    [Dataset creation]

            Enable to use of the shuffle filter.  Use this immediately before the
            DEFLATE filter to increase the compression ratio.
        """
        H5Pset_shuffle(self.id)

    def set_szip(self, unsigned int options, unsigned int pixels_per_block):
        """ (UINT options, UINT pixels_per_block)   [Dataset creation]

            Enable SZIP compression.  See the HDF5 docs for argument meanings, and
            general restrictions on use of the SZIP format.
        """
        H5Pset_szip(self.id, options, pixels_per_block)

    def remove_filter(self, int filter_class):
        """ (INT filter_class)    [Dataset creation]

            Remove a filter from the pipeline.  The class code is one of 
            h5z.FILTER_*.
        """
        H5Premove_filter(self.id, <H5Z_filter_t>filter_class)

# === File access =============================================================

cdef class PropFAID(PropInstanceID):

    """
        Represents a file access property list
    """

    def set_fclose_degree(self, int close_degree):
        """ (INT close_degree)

            Set the file-close degree, which determines the library behavior when
            a file is closed when objects are still open.  See the HDF5 docs for 
            a full explanation.  Legal values:

            * h5f.CLOSE_WEAK
            * h5f.CLOSE_SEMI
            * h5f.CLOSE_STRONG
            * h5f.CLOSE_DEFAULT
        """
        H5Pset_fclose_degree(self.id, <H5F_close_degree_t>close_degree)


    












