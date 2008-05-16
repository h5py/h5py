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
from defs_c   cimport malloc, free, size_t
from h5  cimport herr_t, hid_t, htri_t, herr_t, hsize_t
from h5d cimport H5D_layout_t
from h5z cimport H5Z_filter_t
from utils cimport tuple_to_dims, dims_to_tuple

# Runtime imports
import h5
import h5t
from h5 import DDict
from errors import PropertyError, ConversionError

# === Public constants and data structures ====================================

# Property list classes (I'm surprised there's no enum for this)
CLASS_NO_CLASS       = H5P_NO_CLASS
CLASS_FILE_CREATE    = H5P_FILE_CREATE
CLASS_FILE_ACCESS    = H5P_FILE_ACCESS
CLASS_DATASET_CREATE = H5P_DATASET_CREATE
CLASS_DATASET_XFER   = H5P_DATASET_XFER

CLASS_MAPPER = { H5P_NO_CLASS: 'ERROR', H5P_FILE_CREATE: 'FILE CREATION',
                  H5P_FILE_ACCESS: 'FILE ACCESS', H5P_DATASET_CREATE: 'DATASET CREATION',
                  H5P_DATASET_XFER: 'DATASET TRANSFER'}
CLASS_MAPPER = DDict(CLASS_MAPPER)

DEFAULT = H5P_DEFAULT # not really a "class"

# === Generic property list operations ========================================

def create(hid_t cls_id):
    """ (INT cls_id) => INT property_list_id
    
        Create a new property list as an instance of a class, which should be
        one of CLASS_*.
    """
    cdef hid_t retval
    retval = H5Pcreate(cls_id)
    if retval < 0:
        raise PropertyError("Failed to create instance of property list class %d" % cls_id)
    return retval

def copy(hid_t plist):
    """ (INT plist) => INT new_property_list_id

        Create a new copy of an existing property list object.
    """
    cdef hid_t retval
    retval = H5Pcopy(plist)
    if retval < 0:
        raise PropertyError("Failed to copy property list %d" % plist)
    return retval

def close(hid_t plist):
    """ (INT plist)
    """
    cdef herr_t retval
    retval = H5Pclose(plist)
    if retval < 0:
        raise PropertyError("Failed to close property list %d" % plist)
    return retval

def get_class(hid_t plist):
    """ (INT plist) => INT class_code

        Determine the class of a property list object (one of CLASS_*).
    """
    cdef int retval
    retval = H5Pget_class(plist)
    if retval < 0:
        raise PropertyError("Failed to determine class of property list %d" % plist)
    return retval

def equal(hid_t plist1, hid_t plist2):
    """ (INT plist1, INT plist2) => BOOL lists_are_equal

        Compare two existing property lists for equality.
    """
    cdef htri_t retval
    retval = H5Pequal(plist1, plist2)
    if retval < 0:
        raise PropertyError("Could not compare for equality: %d vs %d" % (plist1, plist2))
    return bool(retval)

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

    retval = H5Pget_version(plist, &super_, &freelist, &stab, &shhdr)
    if retval < 0:
        raise PropertyError("Failed to determine version info for list %d" % plist)
    return (super_, freelist, stab, shhdr)

def set_userblock(hid_t plist, long size):
    """ (INT plist, LONG size)    [File creation]

        Set the file user block size, in bytes.  Must be a power of 2, and at
        least 512.
    """
    cdef herr_t retval
    retval = H5Pset_userblock(plist, <hsize_t>size)
    if retval < 0:
        raise PropertyError("Failed to set userblock size to %d on list %d" % (size, plist))

def get_userblock(hid_t plist):
    """ (INT plist) => LONG size    [File creation]

        Determine the user block size, in bytes.
    """
    cdef herr_t retval
    cdef hsize_t size

    retval = H5Pget_userblock(plist, &size)
    if retval < 0:
        raise PropertyError("Failed to determine user block size on list %d" % plist)
    return size

def set_sizes(hid_t plist, size_t addr, size_t size):
    """ (INT plist, INT addr, INT size)    [File creation]

        Set the addressing offsets and lengths for objects in an HDF5 file, in
        bytes.
    """
    cdef herr_t retval
    retval = H5Pset_sizes(plist, addr, size)
    if retval < 0:
        raise PropertyError("Failed to set sizes on list %d (addr %d, length %d)" % (plist, addr, size))

def get_sizes(hid_t plist):
    """ (INT plist) => TUPLE sizes    [File creation]

        Determine addressing offsets and lengths for objects in an HDF5 file, in
        bytes.  Return value is a 2-tuple with values:

        0:  UINT Address offsets
        1:  UINT Lengths
    """
    cdef herr_t retval
    cdef size_t addr
    cdef size_t size
    retval = H5Pget_sizes(plist, &addr, &size)
    if retval < 0:
        raise PropertyError("Failed to get sizes for plist %d" % plist)
    return (addr, size)

def set_sym_k(hid_t plist, unsigned int ik, unsigned int lk):
    """ (INT plist, INT ik, INT lk)    [File creation]

        Symbol table node settings.  See the HDF5 docs for H5Pset_sym_k.
    """
    cdef herr_t retval
    retval = H5Pset_sym_k(plist, ik, lk)
    if retval < 0:
        raise PropertyError("Failed to set symbol table params on list %d" % plist)

def get_sym_k(hid_t plist):
    """ (INT plist) => TUPLE settings    [File creation]

        Determine symbol table node settings.  See the HDF5 docs for
        H5Pget_sym_k.  Return is a 2-tuple (ik, lk).
    """
    cdef herr_t retval
    cdef unsigned int ik
    cdef unsigned int lk
    
    retval = H5Pget_sym_k(plist, &ik, &lk)
    if retval < 0:
        raise PropertyError("Failed to get symbol table params on list %d" % plist)
    return (ik, lk)

def set_istore_k(hid_t plist, unsigned int ik):
    """ (INT plist, UINT ik)    [File creation]

        See hdf5 docs for H5Pset_istore_k.
    """
    cdef herr_t retval
    retval = H5Pset_istore_k(plist, ik)
    if retval < 0:
        raise PropertyError("Failed to set istore params on list %d (value %d)" % (plist,ik))
    
def get_istore_k(hid_t plist):
    """ (INT plist) => UINT ik    [File creation]

        See HDF5 docs for H5Pget_istore_k
    """
    cdef herr_t retval
    cdef unsigned int ik

    retval = H5Pget_istore_k(plist, &ik)
    if retval < 0:
        raise PropertyError("Failed to get istore params on list %d" % plist)
    return ik

# === Dataset creation properties =============================================

def set_layout(hid_t plist, int layout_code):
    """ (INT plist, INT layout_code)    [Dataset creation]

        Set dataset storage strategy; legal values are:
        * h5d.LAYOUT_COMPACT
        * h5d.LAYOUT_CONTIGUOUS
        * h5d.LAYOUT_CHUNKED
    """
    cdef herr_t retval
    retval = H5Pset_layout(plist, <H5D_layout_t>layout_code)
    if retval < 0:
        raise PropertyError("Failed to set layout of list %d to %d" % (plist, layout_code))
    
def get_layout(hid_t plist):
    """ (INT plist) => INT layout_code   [Dataset creation]

        Determine the storage strategy of a dataset; legal values are:
        * h5d.LAYOUT_COMPACT
        * h5d.LAYOUT_CONTIGUOUS
        * h5d.LAYOUT_CHUNKED
    """
    cdef int retval
    retval = <int>H5Pget_layout(plist)
    if retval < 0:
        raise PropertyError("Failed to get layout of list %d" % plist)

def set_chunk(hid_t plist, object chunksize):
    """ (INT plist_id, TUPLE chunksize)    [Dataset creation]

        Set the dataset chunk size.  It's up to you to provide values which
        are compatible with your dataset.
    """
    cdef herr_t retval
    cdef int rank
    cdef hsize_t* dims
    dims = NULL

    rank = len(chunksize)
    dims = tuple_to_dims(chunksize)
    if dims == NULL:
        raise ValueError("Bad input dimensions tuple: %s" % repr(chunksize))

    retval = H5Pset_chunk(plist, rank, dims)
    if retval < 0:
        free(dims)
        raise PropertyError("Failed to set chunk size to %s on list %d" % (str(chunksize), plist))
    
    free(dims)
    
def get_chunk(hid_t plist):
    """ (INT plist_id) => TUPLE chunk_dimensions    [Dataset creation]

        Obtain the dataset chunk size, as a tuple.
    """
    cdef int rank
    cdef hsize_t *dims

    rank = H5Pget_chunk(plist, 0, NULL)
    if rank < 0:
        raise PropertyError("Failed to get chunk size on list %d" % plist)

    dims = <hsize_t*>malloc(sizeof(hsize_t)*rank)
    rank = H5Pget_chunk(plist, rank, dims)
    if rank < 0:
        free(dims)
        raise PropertyError("Failed to get chunk size on list %d" % plist)

    tpl = dims_to_tuple(dims, rank)
    if tpl is None:
        free(dims)
        raise ConversionError("Bad dims/tuple conversion (plist %d rank %d)" % (plist, rank))

    free(dims)
    return tpl

# === Filter functions ========================================================

def set_deflate(hid_t plist, unsigned int level=5):
    """ (INT plist_id, UINT level=5)    [Dataset creation]

        Enable DEFLATE (gzip) compression, at the given level (0-9, default 5).
    """
    cdef herr_t retval
    retval = H5Pset_deflate(plist, level)
    if retval < 0:
        raise PropertyError("Error enabling DEFLATE (level %d) on list %d" % (level, plist))
    
def set_fletcher32(hid_t plist):
    """ (INT plist_id)    [Dataset creation]

        Enable Fletcher32 error correction on an existing list.
    """
    cdef herr_t retval
    retval = H5Pset_fletcher32(plist)
    if retval < 0:
        raise PropertyError("Error enabling Fletcher32 checksum filter on list %d" % plist)

def set_shuffle(hid_t plist):
    """ (INT plist_id)    [Dataset creation]

        Enable to use of the shuffle filter.  Use this immediately before the
        DEFLATE filter to increase the compression ratio.
    """
    cdef herr_t retval
    retval = H5Pset_shuffle(plist)
    if retval < 0:
        raise PropertyError("Error enabling shuffle filter on list %d" % plist)

def set_szip(hid_t plist, unsigned int options, unsigned int pixels_per_block):
    """ (INT plist, UINT options, UINT pixels_per_block)   [Dataset creation]

        Enable SZIP compression.  See the HDF5 docs for argument meanings, and
        general restrictions on use of the SZIP format.
    """
    cdef herr_t retval
    retval = H5Pset_szip(plist, options, pixels_per_block)
    if retval < 0:
        raise PropertyError("Error enabling szip filter on list %d" % plist)

def remove_filter(hid_t plist, int filter_class):
    """ (INT plist, INT filter_class)    [Dataset creation]

        Remove a filter from the pipeline.  The class code is one of 
        h5z.FILTER_*.
    """
    cdef herr_t retval
    retval = H5Premove_filter(plist, <H5Z_filter_t>filter_class)
    if retval < 0:
        raise PropertyError("Error removing filter %d from list %d" % (filter_class, plist))

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
    cdef herr_t retval
    retval = H5Pset_fclose_degree(fapl_id, <H5F_close_degree_t>close_degree)
    if retval < 0:
        raise PropertyError("Failed to set file close degree on list %d to %d" % (fapl_id, close_degree))
    

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
    retval = H5Pget_filter_by_id(plist, filter_class, &flags, &dmp, NULL, 0, NULL)
    if retval <= 0:
        return False
    return True
    
    

    












