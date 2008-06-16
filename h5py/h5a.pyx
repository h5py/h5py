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
from h5t cimport TypeID, PY_H5Tclose
from h5s cimport SpaceID, H5Sclose

from numpy cimport import_array, ndarray, PyArray_DATA
from utils cimport  check_numpy_read, check_numpy_write, \
                    emalloc, efree

# Runtime imports
import h5
import h5t
import h5s

import_array()

# === General attribute operations ============================================

def create(ObjectID loc_id not None, char* name, TypeID type_id not None, 
            SpaceID space_id not None):
    """ (ObjectID loc_id, STRING name, TypeID type_id, SpaceID space_id) 
        => INT attr_id

        Create a new attribute attached to a parent object, specifiying an 
        HDF5 datatype and dataspace.  For a friendlier version of this function
        try py_create().
    """
    return AttrID(H5Acreate(loc_id.id, name, type_id.id, space_id.id, H5P_DEFAULT))

def open_idx(ObjectID loc_id not None, int idx):
    """ (ObjectID loc_id, UINT idx) => INT attr_id

        Open an exisiting attribute on an object, by zero-based index.
    """
    # If the argument is declared UINT and someone passes in -1, the Pyrex
    # layer happily converts it to something like 2**32 -1, which crashes the
    # HDF5 library.
    if idx < 0:
        raise ValueError("Index must be a non-negative integer.")
    return AttrID(H5Aopen_idx(loc_id.id, idx))

def open_name(ObjectID loc_id not None, char* name):
    """ (ObjectID loc_id, STRING name) => INT attr_id

        Open an existing attribute on an object, by name.
    """
    return AttrID(H5Aopen_name(loc_id.id, name))

def close(AttrID attr_id not None):
    """ (AttrID attr_id)

        Close this attribute and release resources.
    """
    H5Aclose(attr_id.id)

def delete(ObjectID loc_id not None, char* name):
    """ (ObjectID loc_id, STRING name)

        Remove an attribute from an object.
    """
    H5Adelete(loc_id.id, name)

def get_num_attrs(ObjectID loc_id not None):
    """ (ObjectID loc_id) => INT number_of_attributes

        Determine the number of attributes attached to an HDF5 object.
    """
    return H5Aget_num_attrs(loc_id.id)

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

def iterate(ObjectID loc_id not None, object func, object data=None, int startidx=0):
    """ (ObjectID loc_id, FUNCTION func, OBJECT data=None, UINT startidx=0)
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

    retval = H5Aiterate(loc_id.id, &i, <H5A_operator_t>iter_cb, int_tpl)

    if retval < 0:
        if len(int_tpl[2]) != 0:
            raise int_tpl[2][0]

# === Attribute class & methods ===============================================

cdef class AttrID(ObjectID):

    """
        Logical representation of an HDF5 attribute identifier.

        Objects of this class can be used in any HDF5 function call
        which expects an attribute identifier.  Additionally, all H5A*
        functions which always take an attribute instance as the first
        argument are presented as methods of this class.  

        Properties:

        name:   The attribute's name
        dtype:  A Numpy dtype representing this attribute's type
        shape:  A Numpy-style shape tuple representing the dataspace
    """
    property name:
        def __get__(self):
            return self.get_name()

    property shape:
        def __get__(self):
            """ Retrieve the dataspace of this attribute, as a Numpy-style 
                shape tuple.
            """
            cdef hid_t sid
            sid = 0
            try:
                sid = H5Aget_space(self.id)
                return h5s.get_simple_extent_dims(sid)
            finally:
                if sid:
                    H5Sclose(sid)

    property dtype:
        def __get__(self):
            """ Obtain the data-type of this attribute as a Numpy dtype.  Note that the
                resulting dtype is not guaranteed to be byte-for-byte compatible with
                the underlying HDF5 datatype, but is appropriate for use in e.g. the 
                read() and write() functions defined in this module.
            """
            cdef hid_t type_id
            type_id = 0
            
            try:
                type_id = H5Aget_type(self.id)
                return h5t.py_translate_h5t(type_id)
            finally:
                if type_id:
                    PY_H5Tclose(type_id)

    def read(self, ndarray arr_obj not None):
        """ (NDARRAY arr_obj)
            
            Read the attribute data into the given Numpy array.  Note that the 
            Numpy array must have the same shape as the HDF5 attribute, and a 
            conversion-compatible datatype.

            The Numpy array must be writable, C-contiguous and own its data.  If
            this is not the case, an ValueError is raised and the read fails.
        """
        cdef hid_t attr_id
        cdef hid_t mtype_id
        cdef hid_t space_id
        attr_id = self.id
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

    def write(self, ndarray arr_obj not None):
        """ (NDARRAY arr_obj)

            Write the contents of a Numpy array too the attribute.  Note that the 
            Numpy array must have the same shape as the HDF5 attribute, and a 
            conversion-compatible datatype.  

            The Numpy array must be C-contiguous and own its data.  If this is not
            the case, ValueError will be raised and the write will fail.
        """
        cdef hid_t attr_id
        cdef hid_t mtype_id
        cdef hid_t space_id
        attr_id = self.id
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

    def get_name(self):
        """ () => STRING name

            Determine the name of an attribute, given its identifier.
        """
        cdef hid_t attr_id
        cdef int blen
        cdef char* buf
        attr_id = self.id
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

    def get_space(self):
        """ () => INT space_id

            Create and return a copy of the attribute's dataspace.
        """
        return SpaceID(H5Aget_space(self.id))

    def get_type(self):
        """ () => INT type_id

            Create and return a copy of the attribute's datatype.
        """
        return TypeID(H5Aget_type(self.id))











