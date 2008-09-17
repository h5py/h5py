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

    cdef H5O_info_t infostruct
    cdef object __weakref__

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

cdef class _Visit_Data:

    cdef object func
    cdef object exc
    cdef object retval
    cdef ObjInfo objinfo

    def __init__(self, func):
        self.func = func
        self.exc = None
        self.retval = None
        self.objinfo = ObjInfo()

cdef herr_t visit_cb(hid_t obj, char* name, H5O_info_t *info, void* data):

    cdef _Visit_Data wrapper
    wrapper = <_Visit_Data>data

    wrapper.objinfo.infostruct = info[0]

    try:
        retval = wrapper.func(name, wrapper.objinfo)
    except StopIteration:
        return 1
    except BaseException, e:   # The exception MUST be trapped, including SystemExit
        wrapper.exc = e
        return 1

    if retval is not None:
        wrapper.retval = retval
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
        Don't store references to it; use the copy module instead.

        Your callable should also conform to the following behavior:

        1. Return None for normal iteration; raise StopIteration to cancel
           and return None from h5o.visit.

        2. Returning a value other than None cancels iteration and immediately
           returns that value from h5o.visit.

        3. Raising any other exception aborts iteration; the exception will
           be correctly propagated.
    """
    cdef _Visit_Data wrapper
    wrapper = _Visit_Data(func)

    H5Ovisit(obj.id, <H5_index_t>idx_type, <H5_iter_order_t>order, visit_cb, <void*>wrapper)

    if wrapper.exc is not None:
        raise wrapper.exc

    return wrapper.retval  # None or custom value
    









