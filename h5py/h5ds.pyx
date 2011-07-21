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
    Low-level HDF5 "H5G" group interface.
"""

include "config.pxi"

# Compile-time imports
from h5d cimport DatasetID


# === Public constants and data structures ====================================


## cdef class GroupIter:

##     """
##         Iterator over the names of group members.  After this iterator is
##         exhausted, it releases its reference to the group ID.
##     """

##     cdef unsigned long idx
##     cdef unsigned long nobjs
##     cdef GroupID grp

##     def __init__(self, GroupID grp not None):
##         self.idx = 0
##         self.grp = grp
##         self.nobjs = grp.get_num_objs()

##     def __iter__(self):
##         return self

##     def __next__(self):
##         if self.idx == self.nobjs:
##             self.grp = None
##             raise StopIteration

##         retval = self.grp.get_objname_by_idx(self.idx)
##         self.idx = self.idx + 1
##         return retval

# === Basic group management ==================================================


def set_scale(DatasetID dset not None, char* dimname=''):
    """(DatasetID dset, STRING dimname)

    Convert dataset dset to a dimension scale, with optional name dimname.
    """
    H5DSset_scale(dset.id, dimname)

## def create(ObjectID loc not None, object name, PropID lcpl=None,
##            PropID gcpl=None):
##     """(ObjectID loc, STRING name or None, PropLCID lcpl=None,
##         PropGCID gcpl=None)
##     => GroupID

##     Create a new group, under a given parent group.  If name is None,
##     an anonymous group will be created in the file.
##     """
##     cdef hid_t gid
##     cdef char* cname = NULL
##     if name is not None:
##         cname = name

##     if cname != NULL:
##         gid = H5Gcreate2(loc.id, cname, pdefault(lcpl), pdefault(gcpl), H5P_DEFAULT)
##     else:
##         gid = H5Gcreate_anon(loc.id, pdefault(gcpl), H5P_DEFAULT)

##     return GroupID(gid)


## cdef class _GroupVisitor:

##     cdef object func
##     cdef object retval

##     def __init__(self, func):
##         self.func = func
##         self.retval = None

## cdef herr_t cb_group_iter(hid_t gid, char *name, void* vis_in) except 2:

##     cdef _GroupVisitor vis = <_GroupVisitor>vis_in

##     vis.retval = vis.func(name)

##     if vis.retval is not None:
##         return 1
##     return 0


## def iterate(GroupID loc not None, object func, int startidx=0, *,
##             char* obj_name='.'):
##     """ (GroupID loc, CALLABLE func, UINT startidx=0, **kwds)
##     => Return value from func

##     Iterate a callable (function, method or callable object) over the
##     members of a group.  Your callable should have the signature::

##         func(STRING name) => Result

##     Returning None continues iteration; returning anything else aborts
##     iteration and returns that value. Keywords:

##     STRING obj_name (".")
##         Iterate over this subgroup instead
##     """
##     if startidx < 0:
##         raise ValueError("Starting index must be non-negative")

##     cdef int i = startidx
##     cdef _GroupVisitor vis = _GroupVisitor(func)

##     H5Giterate(loc.id, obj_name, &i, <H5G_iterate_t>cb_group_iter, <void*>vis)

##     return vis.retval
