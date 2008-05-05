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
    Provides access to the low-level HDF5 "H5A" attribute interface.

    Functions in this module will raise errors.H5AttributeError.
"""

# Pyrex compile-time imports
from defs_c   cimport malloc, free
from h5  cimport herr_t, hid_t
from h5p cimport H5P_DEFAULT
from h5t cimport H5Tclose
from numpy cimport ndarray, import_array

# Runtime imports
import h5
import h5t
import h5s
from errors import H5AttributeError

import_array()

# === General attribute operations ============================================

def create(hid_t loc_id, char* name, hid_t type_id, hid_t space_id):
    """ (INT loc_id, STRING name, INT type_id, INT space_id) => INT attr_id

        Create a new attribute attached to a parent object, specifiying an 
        HDF5 datatype and dataspace.  For a friendlier version of this function
        try py_create().
    """
    cdef hid_t retval
    retval = H5Acreate(loc_id, name, type_id, space_id, H5P_DEFAULT)
    if retval < 0:
        raise H5AttributeError("Failed to create attribute '%s' on object %d" % (name, loc_id))
    return retval

def open_idx(hid_t loc_id, unsigned int idx):
    """ (INT loc_id, UINT index) => INT attr_id

        Open an exisiting attribute on an object, by zero-based index.
    """
    cdef hid_t retval
    retval = H5Aopen_idx(loc_id, idx)
    if retval < 0:
        raise H5AttributeError("Failed to open attribute at index %d on object %d" % (idx, loc_id))
    return retval

def open_name(hid_t loc_id, char* name):
    """ (INT loc_id, STRING name) => INT attr_id

        Open an existing attribute on an object, by name.
    """
    cdef hid_t retval
    retval = H5Aopen_name(loc_id, name)
    if retval < 0:
        raise H5AttributeError("Failed to open attribute '%s' on object %d" % (name, loc_id))
    return retval

def close(hid_t attr_id):
    """ (INT attr_id)
    """
    cdef hid_t retval
    retval = H5Aclose(attr_id)
    if retval < 0:
        raise H5AttributeError("Failed to close attribute %d" % attr_id)


def delete(hid_t loc_id, char* name):
    """ (INT loc_id, STRING name)

        Remove an attribute from an object.
    """
    cdef herr_t retval
    retval = H5Adelete(loc_id, name)
    if retval < 0:
        raise H5AttributeError("Failed delete attribute '%s' on object %d" % (name, loc_id))


# === Attribute I/O ===========================================================

def read(hid_t attr_id, ndarray arr_obj):
    """ (INT attr_id, NDARRAY arr_obj)
        
        Read the attribute data into the given Numpy array.  Note that the 
        Numpy array must have the same shape as the HDF5 attribute, and a 
        conversion-compatible datatype.  It must also be writable and
        C-contiguous.  This is not currently checked.
    """
    cdef hid_t mtype_id
    cdef herr_t retval
    mtype_id = 0

    try:
        mtype_id = h5t.py_dtype_to_h5t(arr_obj.dtype)
        retval = H5Aread(attr_id, mtype_id, <void*>arr_obj.data)
        if retval < 0:
            raise H5AttributeError("Error reading from attribute %d" % attr_id)
    finally:
        if mtype_id:
            H5Tclose(mtype_id)

def write(hid_t attr_id, ndarray arr_obj):
    """ (INT attr_id, NDARRAY arr_obj)

        Write the contents of a Numpy array too the attribute.  Note that the 
        Numpy array must have the same shape as the HDF5 attribute, and a 
        conversion-compatible datatype.  The Numpy array must also be
        C-contiguous; this is not currently checked.
    """
    
    cdef hid_t mtype_id
    cdef herr_t retval
    mtype_id = 0
    try:
        mtype_id = h5t.py_dtype_to_h5t(arr_obj.dtype)
        retval = H5Awrite(attr_id, mtype_id, <void*>arr_obj.data)
        if retval < 0:
            raise H5AttributeError("Error writing to attribute %d" % attr_id)
    finally:
        if mtype_id:
            H5Tclose(mtype_id)

# === Attribute inspection ====================================================

def get_num_attrs(hid_t loc_id):
    """ (INT loc_id) => INT number_of_attributes

        Determine the number of attributes attached to an HDF5 object.
    """
    cdef int retval
    retval = H5Aget_num_attrs(loc_id)
    if retval < 0:
        raise H5AttributeError("Failed to enumerate attributes of object %d" % loc_id)
    return retval

def get_name(hid_t attr_id):
    """ (INT attr_id) => STRING name

        Determine the name of an attribute, given its identifier.
    """
    cdef int blen
    cdef char* buf
    cdef object strout
    buf = NULL

    blen = H5Aget_name(attr_id, 0, NULL)
    if blen < 0:
        raise H5AttributeError("Failed to get name of attribute %d" % attr_id)
    
    buf = <char*>malloc(sizeof(char)*blen+1)
    blen = H5Aget_name(attr_id, blen+1, buf)
    strout = buf
    free(buf)

    return strout

def get_space(hid_t attr_id):
    """ (INT attr_id) => INT space_id

        Create and return a copy of the attribute's dataspace.
    """
    cdef hid_t retval
    retval = H5Aget_space(attr_id)
    if retval < 0:
        raise H5AttributeError("Failed to retrieve dataspace of attribute %d" % attr_id)
    return retval

def get_type(hid_t attr_id):
    """ (INT attr_id) => INT type_id

        Create and return a copy of the attribute's datatype.
    """
    cdef hid_t retval
    retval = H5Aget_type(attr_id)
    if retval < 0:
        raise H5AttributeError("Failed to retrieve datatype of attribute %d" % attr_id)
    return retval


cdef herr_t iter_cb(hid_t loc_id, char *attr_name, object int_tpl):

    func = int_tpl[0]
    data = int_tpl[1]
    exc_list = int_tpl[2]

    try:
        func(loc_id, attr_name, data)
    except StopIteration:
        return 1
    except Exception, e:
        exc_list.append(e)
        return -1

    return 0


def iterate(hid_t loc_id, object func, object data=None, unsigned int startidx=0):
    """ (INT loc_id, FUNCTION func, OBJECT data=None, UINT startidx=0)
        => INT last_attribute_index

        Iterate an arbitrary Python function over the attributes attached
        to an object.  You can also start at an arbitrary attribute by
        specifying its (zero-based) index.  The return value is the index of 
        the last attribute processed.

        Your function:
        1.  Should accept three arguments: the (INT) id of the parent object, 
            the (STRING) name of the attribute, and an arbitary Python object
            you provide as data.  Any return value is ignored.
        2.  Raise StopIteration to bail out before all attributes are processed.
        3.  Raising anything else immediately aborts iteration, and the
            exception is propagated.
    """
    cdef unsigned int i
    cdef herr_t retval
    i = startidx

    int_tpl = (func, data,[])

    retval = H5Aiterate(loc_id, &i, <H5A_operator_t>iter_cb, int_tpl)

    if retval < 0:
        if len(int_tpl[2]) != 0:
            raise int_tpl[2][0]
        raise H5AttributeError("Error occured during iteration")
    return i-2

# === Python extensions =======================================================

# Pyrex doesn't allow lambdas
def _name_cb(hid_t loc_id, char* name, data):
    data.append(name)

def py_listattrs(hid_t loc_id):
    """ (INT loc_id) => LIST attribute_list

        Create a Python list of attribute names attached to this object.
    """
    nlist = []
    iterate(loc_id, _name_cb, nlist)
    return nlist
    
def py_create(hid_t loc_id, char* name, object dtype_in, object shape):
    """ (INT loc_id, STRING name, DTYPE dtype_in, TUPLE shape)

        Create an attribute from a Numpy dtype and a shape tuple.  To
        create a scalar attribute, provide an empty tuple. If you're creating
        an attribute from an existing array or scalar, consider using py_set().
    """
    cdef hid_t sid
    cdef hid_t type_id
    cdef hid_t aid
    sid = 0
    type_id = 0

    try:
        sid = h5s.create_simple(shape)
        type_id = h5t.py_dtype_to_h5t(dtype_in)

        aid = create(loc_id, name, type_id, sid)
    finally:
        if sid:
            h5s.close(sid)
        if type_id:
            H5Tclose(type_id)

    return aid

def py_shape(hid_t attr_id):
    """ (INT attr_id) => TUPLE shape

        Retrieve the dataspace of this attribute, as a Numpy-style shape tuple.
    """
    cdef hid_t sid
    sid = 0
    
    try:
        sid = get_space(attr_id)
        tpl = h5s.get_simple_extent_dims(sid)
    finally:
        if sid:
            h5s.close(sid)
    return tpl

def py_dtype(hid_t attr_id):
    """ (INT attr_id) => DTYPE numpy_dtype

        Obtain the data-type of this attribute as a Numpy dtype.  Note that the
        resulting dtype is not guaranteed to be byte-for-byte compatible with
        the underlying HDF5 datatype, but is appropriate for use in e.g. the 
        read() and write() functions defined in this module.
    """
    cdef hid_t type_id
    type_id = 0
    
    try:
        type_id = get_type(attr_id)
        dtype_out = h5t.py_h5t_to_dtype(type_id)
    finally:
        if type_id:
            H5Tclose(type_id)
    return dtype_out

def py_get(hid_t parent_id, char* name):
    """ (INT parent_id, STRING name)

        Read an attribute and return the contents as a Numpy ndarray.
        A 0-dimensional array is returned in the case of a scalar attribute.
    """
    cdef hid_t attr_id
    attr_id = open_name(parent_id, name)
    try:
        space = py_shape(attr_id)
        dtype = py_dtype(attr_id)

        arr = ndarray(space, dtype=dtype)
        read(attr_id, arr)
    finally:
        H5Aclose(attr_id)
    return arr

def py_set(hid_t parent_id, char* name, ndarray arr):
    """ (INT parent_id, STRING name, NDARRAY arr)

        Create an attribute and initialize its type, space, and contents to
        a Numpy ndarray.  Note that this function does not return an
        identifier; the attribute is created and then closed.  Fails if an 
        attribute of the same name already exists.
    """
    cdef hid_t attr_id
    attr_id = 0
    attr_id = py_create(parent_id, name, arr.dtype, arr.shape)
    try:
        write(attr_id, arr)
    except:
        H5Aclose(attr_id)
        H5Adelete(parent_id, name)
        raise

    H5Aclose(attr_id)

def py_exists(hid_t parent_id, char* name):
    """ (INT parent_id, STRING name) => BOOL exists

        Determine if the specified attribute exists.  Useful before calling
        py_set().
    """
    cdef hid_t attr_id
    response = None
    attr_id = H5Aopen_name(parent_id, name)
    if attr_id < 0:
        response = False
    else:
        response = True
        H5Aclose(attr_id)

    return response










