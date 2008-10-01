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

include "config.pxi"
include "sync.pxi"

# Module for the new "H5O" functions introduced in HDF5 1.8.0.  Not even
# built with API compatibility level below 1.8.

# Pyrex compile-time imports
from h5 cimport init_hdf5, ObjectID
from h5i cimport wrap_identifier
from h5p cimport PropID, pdefault

# Initialization
init_hdf5()

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

@sync
def get_info(ObjectID obj not None):
    """ (ObjectID obj) => ObjInfo
    """

    cdef ObjInfo info
    info = ObjInfo()

    H5Oget_info(obj.id, &info.infostruct)
    return info


# === Visit routines ==========================================================

cdef class _ObjectVisitor:

    cdef object func
    cdef object retval
    cdef ObjInfo objinfo

    def __init__(self, func):
        self.func = func
        self.retval = None
        self.objinfo = ObjInfo()

cdef herr_t cb_obj_iterate(hid_t obj, char* name, H5O_info_t *info, void* data) except 2:

    cdef _ObjectVisitor visit
    visit = <_ObjectVisitor>data

    visit.objinfo.infostruct = info[0]

    visit.retval = visit.func(name, visit.objinfo)

    if (visit.retval is None) or (not visit.retval):
        return 0
    return 1
 
@sync
def visit(ObjectID loc not None, object func, *,
          int idx_type=H5_INDEX_NAME, int order=H5_ITER_NATIVE,
          char* name=".", PropID lapl=None):
    """ (ObjectID loc, CALLABLE func, **kwds) => <Return value from func>

        Iterate a function or callable object over all objects below the
        specified one.  Your callable should conform to the signature:

            func(STRING name, ObjInfo info) => Result

        Returning None or a logical False continues iteration; returning
        anything else aborts iteration and returns that value.

        Keyword-only arguments:
        * STRING name ("."):            Visit a subgroup of "loc" instead
        * PropLAID lapl (None):          Control how "name" is interpreted
        * INT idx_type (h5.INDEX_NAME):  What indexing strategy to use
        * INT order (h5.ITER_NATIVE):    Order in which iteration occurs
    """
    cdef _ObjectVisitor visit = _ObjectVisitor()

    H5Ovisit_by_name(loc.id, name, <H5_index_t>idx_type,
        <H5_iter_order_t>order, cb_obj_iterate, <void*>visit, pdefault(lapl))

    return visit.retval








