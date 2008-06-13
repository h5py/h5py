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
"""

# Pyrex compile-time imports
from h5p cimport H5P_DEFAULT
from h5t cimport PY_H5Tclose
from h5s cimport H5Sclose

from numpy cimport import_array, ndarray, PyArray_DATA
from utils cimport  check_numpy_read, check_numpy_write, \
                    emalloc, efree

# Runtime imports
import h5
import h5t
import h5s

import_array()

# === General attribute operations ============================================

def create(hid_t loc_id, char* name, hid_t type_id, hid_t space_id):
    """ (INT loc_id, STRING name, INT type_id, INT space_id) => INT attr_id

        Create a new attribute attached to a parent object, specifiying an 
        HDF5 datatype and dataspace.  For a friendlier version of this function
        try py_create().
    """
    return H5Acreate(loc_id, name, type_id, space_id, H5P_DEFAULT)

def open_idx(hid_t loc_id, int idx):
    """ (INT loc_id, UINT idx) => INT attr_id

        Open an exisiting attribute on an object, by zero-based index.
    """
    # If the argument is declared UINT and someone passes in -1, the Pyrex
    # layer happily converts it to something like 2**32 -1, which crashes the
    # HDF5 library.
    if idx < 0:
        raise ValueError("Index must be a non-negative integer.")
    return H5Aopen_idx(loc_id, idx)

def open_name(hid_t loc_id, char* name):
    """ (INT loc_id, STRING name) => INT attr_id

        Open an existing attribute on an object, by name.
    """
    return H5Aopen_name(loc_id, name)

def close(hid_t attr_id):
    """ (INT attr_id)

        Close this attribute and release resources.
    """
    H5Aclose(attr_id)

def delete(hid_t loc_id, char* name):
    """ (INT loc_id, STRING name)

        Remove an attribute from an object.
    """
    H5Adelete(loc_id, name)

# === Attribute I/O ===========================================================

def read(hid_t attr_id, ndarray arr_obj):
    """ (INT attr_id, NDARRAY arr_obj)
        
        Read the attribute data into the given Numpy array.  Note that the 
        Numpy array must have the same shape as the HDF5 attribute, and a 
        conversion-compatible datatype.

        The Numpy array must be writable, C-contiguous and own its data.  If
        this is not the case, an ValueError is raised and the read fails.
    """
    cdef hid_t mtype_id
    cdef hid_t space_id
    mtype_id = 0
    space_id = 0

    try:
        mtype_id = h5t.py_translate_dtype(arr_obj.dtype)
        space_id = H5Aget_space(attr_id)
        check_numpy_write(arr_obj, space_id)

        H5Aread(attr_id, mtype_id, PyArray_DATA(arr_obj))

    finally:
        if mtype_id:
            PY_H5Tclose(mtype_id)
        if space_id:
            H5Sclose(space_id)

def write(hid_t attr_id, ndarray arr_obj):
    """ (INT attr_id, NDARRAY arr_obj)

        Write the contents of a Numpy array too the attribute.  Note that the 
        Numpy array must have the same shape as the HDF5 attribute, and a 
        conversion-compatible datatype.  

        The Numpy array must be C-contiguous and own its data.  If this is not
        the case, ValueError will be raised and the write will fail.
    """
    
    cdef hid_t mtype_id
    cdef hid_t space_id
    mtype_id = 0
    space_id = 0

    try:
        mtype_id = h5t.py_translate_dtype(arr_obj.dtype)
        space_id = H5Aget_space(attr_id)
        check_numpy_read(arr_obj, space_id)

        H5Awrite(attr_id, mtype_id, PyArray_DATA(arr_obj))

    finally:
        if mtype_id:
            PY_H5Tclose(mtype_id)
        if space_id:
            H5Sclose(space_id)

# === Attribute inspection ====================================================

def get_num_attrs(hid_t loc_id):
    """ (INT loc_id) => INT number_of_attributes

        Determine the number of attributes attached to an HDF5 object.
    """
    return H5Aget_num_attrs(loc_id)

def get_name(hid_t attr_id):
    """ (INT attr_id) => STRING name

        Determine the name of an attribute, given its identifier.
    """
    cdef int blen
    cdef char* buf
    buf = NULL

    try:
        blen = H5Aget_name(attr_id, 0, NULL)
        assert blen >= 0
        buf = <char*>emalloc(sizeof(char)*blen+1)
        blen = H5Aget_name(attr_id, blen+1, buf)
        strout = buf
    finally:
        efree(buf)

    return strout

def get_space(hid_t attr_id):
    """ (INT attr_id) => INT space_id

        Create and return a copy of the attribute's dataspace.
    """
    return H5Aget_space(attr_id)

def get_type(hid_t attr_id):
    """ (INT attr_id) => INT type_id

        Create and return a copy of the attribute's datatype.
    """
    return H5Aget_type(attr_id)


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


def iterate(hid_t loc_id, object func, object data=None, int startidx=0):
    """ (INT loc_id, FUNCTION func, OBJECT data=None, UINT startidx=0)
        => INT last_attribute_index

        Iterate an arbitrary Python function over the attributes attached
        to an object.  You can also start at an arbitrary attribute by
        specifying its (zero-based) index.

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
    if startidx < 0:
        raise ValueError("Starting index must be a non-negative integer.")
    i = startidx

    int_tpl = (func, data,[])

    retval = H5Aiterate(loc_id, &i, <H5A_operator_t>iter_cb, int_tpl)

    if retval < 0:
        if len(int_tpl[2]) != 0:
            raise int_tpl[2][0]

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
        type_id = h5t.py_translate_dtype(dtype_in)

        return create(loc_id, name, type_id, sid)

    finally:
        if sid:
            H5Sclose(sid)
        if type_id:
            PY_H5Tclose(type_id)


def py_shape(hid_t attr_id):
    """ (INT attr_id) => TUPLE shape

        Retrieve the dataspace of this attribute, as a Numpy-style shape tuple.
    """
    cdef hid_t sid
    sid = 0
    
    try:
        sid = H5Aget_space(attr_id)
        return h5s.get_simple_extent_dims(sid)

    finally:
        if sid:
            H5Sclose(sid)

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
        type_id = H5Aget_type(attr_id)
        return h5t.py_translate_h5t(type_id)
    finally:
        if type_id:
            PY_H5Tclose(type_id)

def py_get(hid_t parent_id, char* name):
    """ (INT parent_id, STRING name)

        Read an attribute and return the contents as a Numpy ndarray.
        A 0-dimensional array is returned in the case of a scalar attribute.
    """
    cdef hid_t attr_id
    attr_id = H5Aopen_name(parent_id, name)
    try:
        space = py_shape(attr_id)
        dtype = py_dtype(attr_id)

        arr = ndarray(space, dtype=dtype)
        read(attr_id, arr)
        return arr

    finally:
        H5Aclose(attr_id)

def py_set(hid_t parent_id, char* name, ndarray arr):
    """ (INT parent_id, STRING name, NDARRAY arr)

        Create an attribute and initialize its type, space, and contents to
        a Numpy ndarray.  Note that this function does not return an
        identifier; the attribute is created and then closed.  Fails if an 
        attribute of the same name already exists.
    """
    cdef hid_t attr_id
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
    
    try:
        attr_id = H5Aopen_name(parent_id, name)
    except:
        return False

    H5Aclose(attr_id)
    return True










