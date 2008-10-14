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

include "config.pxi"
include "sync.pxi"

# Compile-time imports
from h5 cimport init_hdf5, SmartStruct
from h5t cimport TypeID, typewrap, py_create
from h5s cimport SpaceID
from h5p cimport PropID, pdefault
from numpy cimport import_array, ndarray, PyArray_DATA
from utils cimport check_numpy_read, check_numpy_write, emalloc, efree

# Initialization
init_hdf5()
import_array()


# === General attribute operations ============================================

# --- create, create_by_name ---

IF H5PY_18API:
    @sync
    def create(ObjectID loc not None, char* name, TypeID tid not None,
        SpaceID space not None, *, char* obj_name='.', PropID lapl=None):
        """(ObjectID loc, STRING name, TypeID tid, SpaceID space, **kwds) => AttrID
            
        Create a new attribute, attached to an existing object.

        Keywords:
        * STRING obj_name (".")     Attach attribute to this group member instead
        * PropID lapl (None)        Determines how obj_name is interpreted
        """

        return AttrID(H5Acreate_by_name(loc.id, obj_name, name, tid.id,
                space.id, H5P_DEFAULT, H5P_DEFAULT, pdefault(lapl)))

ELSE:
    @sync
    def create(ObjectID loc not None, char* name, TypeID tid not None, 
        SpaceID space not None):
        """(ObjectID loc, STRING name, TypeID tid, SpaceID space) => AttrID

        Create a new attribute attached to a parent object, specifiying an 
        HDF5 datatype and dataspace.
        """
        return AttrID(H5Acreate(loc.id, name, tid.id, space.id, H5P_DEFAULT))


# --- open, open_by_name, open_by_idx ---

IF H5PY_18API:
    @sync
    def open(ObjectID loc not None, char* name=NULL, int index=-1, *,
        char* obj_name='.', int index_type=H5_INDEX_NAME, int order=H5_ITER_NATIVE,
        PropID lapl=None):
        """(ObjectID loc, STRING name=, INT index=, **kwds) => AttrID
       
        Open an attribute attached to an existing object.  You must specify
        exactly one of either name or idx.

        Keyword-only arguments:
        * STRING obj_name ("."):         Attribute is attached to this group member
        * PropID lapl (None):            Controls how "obj_name" is interpreted
        * INT index_type (h5.INDEX_NAME) Controls how idx is interpreted
        * INT order (h5.ITER_NATIVE)     Controls how idx is interpreted
        """
        if (name == NULL and index < 0) or (name != NULL and index >= 0):
            raise TypeError("Exactly one of name or idx must be specified")

        if name != NULL:
            return AttrID(H5Aopen_by_name(loc.id, obj_name, name,
                            H5P_DEFAULT, pdefault(lapl)))
        else:
            return AttrID(H5Aopen_by_idx(loc.id, obj_name,
                <H5_index_t>index_type, <H5_iter_order_t>order, index,
                H5P_DEFAULT, pdefault(lapl)))

ELSE:
    @sync
    def open(ObjectID loc not None, char* name=NULL, int index=-1):
        """(ObjectID loc, STRING name=, INT index=) => AttrID

        Open an attribute attached to an existing object.  You must specify
        exactly one of either name or idx.
        """
        if (name == NULL and index < 0) or (name != NULL and index >= 0):
            raise TypeError("Exactly one of name or idx must be specified")

        if name != NULL:
            return AttrID(H5Aopen_name(loc.id, name))
        else:
            return AttrID(H5Aopen_idx(loc.id, index))


# --- exists, exists_by_name ---

IF H5PY_18API:
    @sync
    def exists(ObjectID loc not None, char* name, *,
                char* obj_name=NULL, PropID lapl=None):
        """(ObjectID loc, STRING name, **kwds) => BOOL

        Determine if an attribute is attached to this object.

        Keywords:
        * STRING obj_name:  Look for attributes attached to this group member
        * PropID lapl:      Determines how "obj_name" is interpreted
        """
        if obj_name is NULL:
            return <bint>H5Aexists(loc.id, name)
        else:
            return <bint>H5Aexists_by_name(loc.id, obj_name, name, pdefault(lapl))
ELSE:
    cdef herr_t cb_exist(hid_t loc_id, char* attr_name, void* ref_name) except 2:

        if strcmp(attr_name, <char*>ref_name) == 0:
            return 1
        return 0

    @sync
    def exists(ObjectID loc not None, char* name):
        """(ObjectID loc, STRING name) => BOOL

        Determine if an attribute named "ref_name" is attached to this object.
        """
        cdef unsigned int i=0

        return <bint>H5Aiterate(loc.id, &i, <H5A_operator_t>cb_exist, <void*>name)


# --- rename, rename_by_name ---

IF H5PY_18API:
    @sync
    def rename(ObjectID loc not None, char* name, char* new_name, *,
        char* obj_name='.', PropID lapl=None):
        """(ObjectID loc, STRING name, STRING new_name, **kwds)

        Rename an attribute.

        Keywords:
        * STRING obj_name (".")     Attribute is attached to this group member
        * PropID lapl (None)        Determines how obj_name is interpreted
        """
        H5Arename_by_name(loc.id, obj_name, name, new_name, pdefault(lapl))

@sync
def delete(ObjectID loc not None, char* name):
    """(ObjectID loc, STRING name)

    Remove an attribute from an object.
    """
    H5Adelete(loc.id, name)

@sync
def get_num_attrs(ObjectID loc not None):
    """(ObjectID loc) => INT

    Determine the number of attributes attached to an HDF5 object.
    """
    return H5Aget_num_attrs(loc.id)

cdef class _AttrVisitor:

    cdef object func
    cdef object retval

    def __init__(self, func):
        self.func = func
        self.retval = None

cdef herr_t cb_attr_iter(hid_t loc_id, char* attr_name, void* vis_in) except 2:

    cdef _AttrVisitor vis = <_AttrVisitor>vis_in

    vis.retval = vis.func(attr_name)

    if vis.retval is not None:
        return 1
    return 0

@sync
def iterate(ObjectID loc not None, object func, int index=0):
    """(ObjectID loc, CALLABLE func, INT index=0) => <Return value from func>

    Iterate a callable (function, method or callable object) over the
    attributes attached to this object.  You callable should have the
    signature:

        func(STRING name) => Result

    Returning None continues iteration; returning anything else aborts
    iteration and returns that value.

    Tip: To make your code forward-compatible with later versions of this
    function (which supply more information to the callback), add an
    additional *args parameter.
    """
    if index < 0:
        raise ValueError("Starting index must be a non-negative integer.")

    cdef unsigned int i = index
    cdef _AttrVisitor vis = _AttrVisitor(func)

    H5Aiterate(loc.id, &i, <H5A_operator_t>cb_attr_iter, <void*>vis)

    return vis.retval


@sync
def py_listattrs(ObjectID loc not None):
    """(ObjectID loc) => LIST

    Get a list of the names of the attributes attached to an object.
    """
    retlist = []
    iterate(loc, retlist.append)
    return retlist




IF H5PY_18API:
    cdef class AttrInfo(SmartStruct):

        cdef H5A_info_t info

        property corder_valid:
            """Indicates if the creation order is valid"""
            def __get__(self):
                return <bint>self.info.corder_valid
        property corder:
            """Creation order"""
            def __get__(self):
                return <int>self.info.corder
        property cset:
            """Character set of attribute name (integer typecode from h5t)"""
            def __get__(self):
                return <int>self.info.cset
        property data_size:
            """Size of raw data"""
            def __get__(self):
                return self.info.data_size

        def _hash(self):
            return hash((self.corder_valid, self.corder, self.cset, self.data_size))

    @sync
    def get_info(ObjectID loc not None, char* name=NULL, int index=-1, *,
                char* obj_name='.', PropID lapl=None,
                int index_type=H5_INDEX_NAME, int order=H5_ITER_NATIVE):
        """(ObjectID loc, STRING name=, INT index=, **kwds) => AttrInfo

        Get information about an attribute, in one of two ways:

        1. If you have the attribute identifier, just pass it in
        2. If you have the parent object, supply it and exactly one of
           either name or index.

        Keywords:
        * STRING obj_name (".")             Use this group member instead
        * PropID lapl (None)                How "obj_name" is resolved
        * INT index_type (h5.INDEX_NAME)    Which index to use
        * INT order (h5.ITER_NATIVE)        What order the index is in
        """
        cdef AttrInfo info = AttrInfo()

        if name == NULL and index < 0:
            H5Aget_info(loc.id, &info.info)
        elif name != NULL and index >= 0:
            raise TypeError("At most one of name and index may be specified")
        elif name != NULL:
            H5Aget_info_by_name(loc.id, obj_name, name, &info.info, pdefault(lapl))
        elif index >= 0:
            H5Aget_info_by_idx(loc.id, obj_name, <H5_index_t>index_type,
                <H5_iter_order_t>order, index, &info.info, pdefault(lapl))

        return info

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

        Hashable: No
        Equality: Identifier comparison
    """
    property name:
        """The attribute's name"""
        def __get__(self):
            return self.get_name()

    property shape:
        """A Numpy-style shape tuple representing the attribute's dataspace"""
        def __get__(self):

            cdef SpaceID space
            space = self.get_space()
            return space.get_simple_extent_dims()

    property dtype:
        """A Numpy-stype dtype object representing the attribute's datatype"""
        def __get__(self):

            cdef TypeID tid
            tid = self.get_type()
            return tid.py_dtype()

    @sync
    def _close(self):
        """()

        Close this attribute and release resources.  You don't need to
        call this manually; attributes are automatically destroyed when
        their Python wrappers are freed.
        """
        H5Aclose(self.id)

    @sync
    def read(self, ndarray arr_obj not None):
        """(NDARRAY arr_obj)

        Read the attribute data into the given Numpy array.  Note that the 
        Numpy array must have the same shape as the HDF5 attribute, and a 
        conversion-compatible datatype.

        The Numpy array must be writable and C-contiguous.  If this is not
        the case, the read will fail with an exception.
        """
        cdef TypeID mtype
        cdef hid_t space_id
        space_id = 0

        try:
            space_id = H5Aget_space(self.id)
            check_numpy_write(arr_obj, space_id)

            mtype = py_create(arr_obj.dtype)

            H5Aread(self.id, mtype.id, PyArray_DATA(arr_obj))

        finally:
            if space_id:
                H5Sclose(space_id)

    @sync
    def write(self, ndarray arr_obj not None):
        """(NDARRAY arr_obj)

        Write the contents of a Numpy array too the attribute.  Note that
        the Numpy array must have the same shape as the HDF5 attribute, and
        a conversion-compatible datatype.  

        The Numpy array must be C-contiguous.  If this is not the case, 
        the write will fail with an exception.
        """
        cdef TypeID mtype
        cdef hid_t space_id
        space_id = 0

        try:
            space_id = H5Aget_space(self.id)
            check_numpy_read(arr_obj, space_id)
            mtype = py_create(arr_obj.dtype)

            H5Awrite(self.id, mtype.id, PyArray_DATA(arr_obj))

        finally:
            if space_id:
                H5Sclose(space_id)

    @sync
    def get_name(self):
        """() => STRING name

        Determine the name of an attribute, given its identifier.
        """
        cdef int blen
        cdef char* buf
        buf = NULL

        try:
            blen = H5Aget_name(self.id, 0, NULL)
            assert blen >= 0
            buf = <char*>emalloc(sizeof(char)*blen+1)
            blen = H5Aget_name(self.id, blen+1, buf)
            strout = buf
        finally:
            efree(buf)

        return strout

    @sync
    def get_space(self):
        """() => SpaceID

        Create and return a copy of the attribute's dataspace.
        """
        return SpaceID(H5Aget_space(self.id))

    @sync
    def get_type(self):
        """() => TypeID

        Create and return a copy of the attribute's datatype.
        """
        return typewrap(H5Aget_type(self.id))











