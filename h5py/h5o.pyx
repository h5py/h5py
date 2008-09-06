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

# Module for the new "H5O" functions introduced in HDF5 1.8.0.  Not even
# built with API compatibility level below 1.8.

# Pyrex compile-time imports
from h5f cimport wrap_identifier

# Runtime imports
import h5

cdef class ObjInfo:

    property fileno:
        def __get__(self):
            return self.infostruct.fileno
    property addr:
        def __get__(self):
            return self.infostruct.addr
    property type:
        def __get__(self):
            return <int>self.infostruct.type
    property rc:
        def __get__(self):
            return self.infostruct.rc

    cdef H5O_info_t infostruct

    def __copy__(self):
        cdef ObjInfo newcopy
        newcopy = ObjInfo()
        newcopy.infostruct = self.infostruct
        return newcopy

def get_info(ObjectID obj not None):
    """ (ObjectID obj) => ObjInfo
    """

    cdef ObjInfo info
    info = ObjInfo()

    H5Oget_info(obj.id, &info.infostruct)
    return info

cdef class _Triplet:

    cdef object func
    cdef object exc
    cdef ObjInfo objinfo

    def __init__(self, func):
        self.func = func
        self.exc = None
        self.objinfo = ObjInfo()

cdef herr_t iter_cb(hid_t obj, char* name, H5O_info_t *info, void* data):

    cdef _Triplet triplet
    triplet = <_Triplet>data

    triplet.objinfo.infostruct = info[0]

    try:
        retval = triplet.func(name, triplet.objinfo)
    except BaseException, e:   # The exception MUST be propagated
        triplet.exc = e
        return 1

    if retval is not None:
        return 1

    return 0
 

def visit(ObjectID obj not None, object func, int idx_type=H5_INDEX_NAME,
          int order=H5_ITER_NATIVE):
    """ (ObjectID obj, CALLABLE func, INT idx_type=, INT order=)

        Recursively iterate a function or callable object over this group's
        contents.  Your callable should match the signature:

            func(name, info)

        where "name" is (a) name relative to the starting group, and "info" is
        an ObjInfo instance describing each object.  Please note the same
        ObjInfo instance is provided call to call, with its values mutated.
        Don't store references to it; use the copy module instead:

            mylist.append(info)             # WRONG
            mylist.append(copy.copy(info))  # RIGHT
    """
    cdef _Triplet triplet
    triplet = _Triplet(func)

    H5Ovisit(obj.id, <H5_index_t>idx_type, <H5_iter_order_t>order, iter_cb, <void*>triplet)

    if triplet.exc is not None:
        raise triplet.exc

    









