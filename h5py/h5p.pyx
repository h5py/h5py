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
from h5d cimport H5D_layout_t
from h5z cimport H5Z_filter_t

from utils cimport  require_tuple, convert_dims, convert_tuple, \
                    emalloc, efree, pybool

# Runtime imports
import h5
import h5t
from h5 import DDict

# === Public constants and data structures ====================================

# Property list classes
NO_CLASS       = H5P_NO_CLASS
FILE_CREATE    = H5P_FILE_CREATE
FILE_ACCESS    = H5P_FILE_ACCESS
DATASET_CREATE = H5P_DATASET_CREATE
DATASET_XFER   = H5P_DATASET_XFER
MOUNT          = H5P_MOUNT

DEFAULT = H5P_DEFAULT

# === Generic property list operations ========================================

def create(hid_t cls_id):
    """ (INT cls_id) => INT property_list_id
    
        Create a new property list as an instance of a class; classes are:
            FILE_CREATE
            FILE_ACCESS
            DATASET_CREATE
            DATASET_XFER
            MOUNT
    """
    return H5Pcreate(cls_id)

def copy(hid_t plist):
    """ (INT plist) => INT new_property_list_id

        Create a new copy of an existing property list object.
    """
    return H5Pcopy(plist)

def close(hid_t plist):
    """ (INT plist)
    """
    H5Pclose(plist)

def get_class(hid_t plist):
    """ (INT plist) => INT class_code

        Determine the class of a property list object.
    """
    return H5Pget_class(plist)

def equal(hid_t plist1, hid_t plist2):
    """ (INT plist1, INT plist2) => BOOL lists_are_equal

        Compare two existing property lists or classes for equality.
    """
    return pybool(H5Pequal(plist1, plist2))

# === File creation ===========================================================

def get_version(hid_t plist):
    """ (INT plist) => TUPLE version_info   [File creation]

        Determine version information of various file attributes. Elements are:

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

    H5Pget_version(plist, &super_, &freelist, &stab, &shhdr)

    return (super_, freelist, stab, shhdr)

def set_userblock(hid_t plist, hsize_t size):
    """ (INT plist, INT/LONG size)    [File creation]

        Set the file user block size, in bytes.  
        Must be a power of 2, and at least 512.
    """
    H5Pset_userblock(plist, size)

def get_userblock(hid_t plist):
    """ (INT plist) => LONG size    [File creation]

        Determine the user block size, in bytes.
    """
    cdef hsize_t size
    H5Pget_userblock(plist, &size)
    return size

def set_sizes(hid_t plist, size_t addr, size_t size):
    """ (INT plist, INT addr, INT size)    [File creation]

        Set the addressing offsets and lengths for objects 
        in an HDF5 file, in bytes.
    """
    H5Pset_sizes(plist, addr, size)

def get_sizes(hid_t plist):
    """ (INT plist) => TUPLE sizes    [File creation]

        Determine addressing offsets and lengths for objects in an 
        HDF5 file, in bytes.  Return value is a 2-tuple with values:

        0:  UINT Address offsets
        1:  UINT Lengths
    """
    cdef size_t addr
    cdef size_t size
    H5Pget_sizes(plist, &addr, &size)
    return (addr, size)

def set_sym_k(hid_t plist, unsigned int ik, unsigned int lk):
    """ (INT plist, INT ik, INT lk)    [File creation]

        Symbol table node settings.  See the HDF5 docs for H5Pset_sym_k.
    """
    H5Pset_sym_k(plist, ik, lk)

def get_sym_k(hid_t plist):
    """ (INT plist) => TUPLE settings    [File creation]

        Determine symbol table node settings.  See the HDF5 docs for
        H5Pget_sym_k.  Return is a 2-tuple (ik, lk).
    """
    cdef unsigned int ik
    cdef unsigned int lk
    H5Pget_sym_k(plist, &ik, &lk)
    return (ik, lk)

def set_istore_k(hid_t plist, unsigned int ik):
    """ (INT plist, UINT ik)    [File creation]

        See hdf5 docs for H5Pset_istore_k.
    """
    H5Pset_istore_k(plist, ik)
    
def get_istore_k(hid_t plist):
    """ (INT plist) => UINT ik    [File creation]

        See HDF5 docs for H5Pget_istore_k
    """
    cdef unsigned int ik
    H5Pget_istore_k(plist, &ik)
    return ik

# === Dataset creation properties =============================================

def set_layout(hid_t plist, int layout_code):
    """ (INT plist, INT layout_code)    [Dataset creation]

        Set dataset storage strategy; legal values are:
        * h5d.COMPACT
        * h5d.CONTIGUOUS
        * h5d.CHUNKED
    """
    H5Pset_layout(plist, <H5D_layout_t>layout_code)
    
def get_layout(hid_t plist):
    """ (INT plist) => INT layout_code   [Dataset creation]

        Determine the storage strategy of a dataset; legal values are:
        * h5d.COMPACT
        * h5d.CONTIGUOUS
        * h5d.CHUNKED
    """
    return <int>H5Pget_layout(plist)

def set_chunk(hid_t plist, object chunksize):
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
        H5Pset_chunk(plist, rank, dims)
    finally:
        efree(dims)
    
def get_chunk(hid_t plist):
    """ (INT plist_id) => TUPLE chunk_dimensions    [Dataset creation]

        Obtain the dataset chunk size, as a tuple.
    """
    cdef int rank
    cdef hsize_t *dims

    rank = H5Pget_chunk(plist, 0, NULL)
    assert rank >= 0
    dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)

    try:
        H5Pget_chunk(plist, rank, dims)
        tpl = convert_dims(dims, rank)
        return tpl
    finally:
        efree(dims)

# === Filter functions ========================================================

def set_deflate(hid_t plist, unsigned int level=5):
    """ (INT plist_id, UINT level=5)    [Dataset creation]

        Enable DEFLATE (gzip) compression, at the given level.
        Valid levels are 0-9, default is 5.
    """
    H5Pset_deflate(plist, level)
    
def set_fletcher32(hid_t plist):
    """ (INT plist_id)    [Dataset creation]

        Enable Fletcher32 error correction on an existing list.
    """
    H5Pset_fletcher32(plist)

def set_shuffle(hid_t plist):
    """ (INT plist_id)    [Dataset creation]

        Enable to use of the shuffle filter.  Use this immediately before the
        DEFLATE filter to increase the compression ratio.
    """
    H5Pset_shuffle(plist)

def set_szip(hid_t plist, unsigned int options, unsigned int pixels_per_block):
    """ (INT plist, UINT options, UINT pixels_per_block)   [Dataset creation]

        Enable SZIP compression.  See the HDF5 docs for argument meanings, and
        general restrictions on use of the SZIP format.
    """
    H5Pset_szip(plist, options, pixels_per_block)

def remove_filter(hid_t plist, int filter_class):
    """ (INT plist, INT filter_class)    [Dataset creation]

        Remove a filter from the pipeline.  The class code is one of 
        h5z.FILTER_*.
    """
    H5Premove_filter(plist, <H5Z_filter_t>filter_class)

# === File access =============================================================

def set_fclose_degree(hid_t fapl_id, int close_degree):
    """ (INT fapl_id, INT close_degree)

        Set the file-close degree, which determines the library behavior when
        a file is closed when objects are still open.  See the HDF5 docs for 
        a full explanation.  Legal values:

        * h5f.CLOSE_WEAK
        * h5f.CLOSE_SEMI
        * h5f.CLOSE_STRONG
        * h5f.CLOSE_DEFAULT
    """
    H5Pset_fclose_degree(fapl_id, <H5F_close_degree_t>close_degree)
    

# === Python extensions =======================================================

def py_has_filter(hid_t plist, int filter_class):
    """ (INT plist_id, INT filter_class_code) 
        => BOOL has_filter    [Dataset creation]
        
        Determine if a property list has the given filter.
    """
    cdef herr_t retval
    cdef unsigned int flags
    cdef size_t dmp
    dmp = 0
    try:
        H5Pget_filter_by_id(plist, filter_class, &flags, &dmp, NULL, 0, NULL)
    except:
        return False
    return True
    
PY_CLASS = DDict({ H5P_NO_CLASS: 'ERROR', H5P_FILE_CREATE: 'FILE CREATION',
            H5P_FILE_ACCESS: 'FILE ACCESS', H5P_DATASET_CREATE: 'DATASET CREATION',
            H5P_DATASET_XFER: 'DATASET TRANSFER', H5P_DEFAULT: 'DEFAULT'})

    

    












