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

# Pyrex compile-time imports
from defs_c   cimport malloc, free, time_t
from h5  cimport herr_t, hid_t, size_t, hsize_t

# Runtime imports
import h5
from h5 import DDict
from errors import GroupError

# === Public constants and data structures ====================================

# Enumerated object types for groups "H5G_obj_t"
OBJ_UNKNOWN  = H5G_UNKNOWN
OBJ_LINK     = H5G_LINK
OBJ_GROUP    = H5G_GROUP
OBJ_DATASET  = H5G_DATASET
OBJ_DATATYPE = H5G_TYPE
OBJ_MAPPER = { H5G_UNKNOWN: "UNKNOWN", H5G_LINK: "LINK", H5G_GROUP: "GROUP",
                 H5G_DATASET: "DATASET", H5G_TYPE: "DATATYPE" }
OBJ_MAPPER = DDict(OBJ_MAPPER)

# Enumerated link types "H5G_link_t"
LINK_ERROR = H5G_LINK_ERROR
LINK_HARD  = H5G_LINK_HARD
LINK_SOFT  = H5G_LINK_SOFT
LINK_MAPPER = { H5G_LINK_ERROR: "ERROR", H5G_LINK_HARD: "HARDLINK", 
                H5G_LINK_SOFT: "SOFTLINK" }
LINK_MAPPER = DDict(LINK_MAPPER)

cdef class GroupStat:
    """ Represents the H5G_stat_t structure containing group member info.

        Fields:
        fileno -> 2-tuple uniquely* identifying the current file
        objno  -> 2-tuple uniquely* identifying this object
        nlink  -> Number of hard links to this object
        mtime  -> Modification time of this object (flaky)

        *"Uniquely identifying" means unique among currently open files, 
        not universally unique.
    """
    cdef public object fileno  # will be a 2-tuple
    cdef public object objno   # will be a 2-tuple
    cdef public int nlink
    cdef public int type
    cdef public time_t mtime
    cdef public size_t linklen


# === Basic group management ==================================================

def open(hid_t loc_id, char* name):
    """ (INT loc_id, STRING name)

        Open an existing HDF5 group, attached to some other group.
    """
    cdef herr_t retval
    
    retval = H5Gopen(loc_id, name)
    if retval < 0:
        raise GroupError("Failed to open group %s at %d" % (name, loc_id))
    return retval

def close(hid_t group_id):
    """ (INT group_id)
    """
    cdef herr_t retval

    retval = H5Gclose(group_id)
    if retval < 0:
        raise GroupError("Can't close group %d" % group_id)

def create(hid_t loc_id, char* name, int size_hint=-1):
    """ (INT loc_id, STRING name, INT size_hint=-1)

        Create a new group named "name", under a parent group identified by
        "loc_id".  See the HDF5 documentation for the meaning of size_hint.
    """
    cdef herr_t retval
    
    retval = H5Gcreate(loc_id, name, size_hint)
    if retval < 0:
        raise GroupError("Can't create group %s under %d" % (name, loc_id))
    return retval

# === Group member management =================================================

def link(hid_t loc_id, char* current_name, char* new_name, int link_type=H5G_LINK_HARD, hid_t remote_id=-1):
    """ ( INT loc_id, STRING current_name, STRING new_name, 
          INT link_type=LINK_HARD, INT remote_id=-1) 

        Create a new hard or soft link.  The link target (object the link will
        point to) is identified by its parent group "loc_id", and the string
        current_name.  The name of the new link is new_name.  If you want to
        create the link in another group, pass its identifier through
        remote_id.

        Hard links are created by default (link_type=LINK_HARD).  To create a
        symbolic link, pass in link_type=LINK_SOFT.
    """
    cdef herr_t retval
    
    if remote_id < 0:
        remote_id = loc_id

    retval = H5Glink2(loc_id, current_name, <H5G_link_t>link_type, remote_id, new_name)
    if retval < 0:
        raise GroupError('Failed to link %d=>"%s" to %d=>"%s"' % (loc_id, current_name, remote_id, new_name))

def unlink(hid_t loc_id, char* name):
    """ (INT loc_id, STRING name)

        Remove a link to an object from the given group.
    """
    cdef herr_t retval

    retval = H5Gunlink(loc_id, name)
    if retval < 0:
        raise GroupError("Failed to unlink member '%s' from group %d" % (name, loc_id))


def move(hid_t loc_id, char* current_name, char* new_name, hid_t remote_id=-1):
    """ (INT loc_id, STRING current_name, STRING new_name, INT new_group_id=-1)

        Relink an object, identified by its parent group loc_id and string
        current_name.  The new name of the link is new_name.  You can create
        the link in a different group by passing its identifier to remote_id.
    """
    cdef int retval
    if remote_id < 0:
        remote_id = loc_id

    retval = H5Gmove2(loc_id, current_name, remote_id, new_name)
    if retval < 0:
        raise GroupError('Failed to move %d=>"%s" to %d=>"%s"' % (loc_id, current_name, remote_id, new_name))

# === Member inspection and iteration =========================================

def get_num_objs(hid_t loc_id):
    """ (INT loc_id) => INT number_of_objects

        Get the number of objects attached to a given group.
    """
    cdef hsize_t size
    cdef herr_t retval
    
    retval = H5Gget_num_objs(loc_id, &size)
    if retval < 0:
        raise GroupError("Group enumeration failed: %d" % loc_id)

    return size

def get_objname_by_idx(hid_t loc_id, hsize_t idx):
    """ (INT loc_id, INT idx) => STRING object_name

        Get the name of a group member given its zero-based index.
    """
    cdef int retval
    cdef char* buf
    cdef object pystring

    retval = H5Gget_objname_by_idx(loc_id, idx, NULL, 0)
    if retval < 0:
        raise GroupError("Error accessing element %d of group %d" % (idx, loc_id))
    elif retval == 0:
        return None
    else:
        buf = <char*>malloc(retval+1)
        retval = H5Gget_objname_by_idx(loc_id, idx, buf, retval+1)
        pystring = buf
        free(buf)
        return pystring

def get_objtype_by_idx(hid_t loc_id, hsize_t idx):
    """ (INT loc_id, INT idx) => INT object_type_code

        Get the type of an object attached to a group, given its zero-based
        index.  Return value is one of the OBJ_* constants.
    """
    cdef int retval

    retval = H5Gget_objtype_by_idx(loc_id, idx)
    if retval < 0:
        raise GroupError("Error accessing element %d of group %d" % (idx, loc_id))

    return retval

def get_objinfo(hid_t loc_id, char* name, int follow_link=1):
    """ (INT loc_id, STRING name, BOOL follow_link=True)
        => GroupStat object

        Obtain information about an arbitrary object attached to a group. The
        return value is a GroupStat object; see that class's docstring
        for a description of its attributes.  If follow_link is True (default)
        and the object is a symbolic link, the information returned describes 
        its target.  Otherwise the information describes the link itself.
    """
    cdef int retval
    cdef H5G_stat_t stat
    cdef object statobj

    retval = H5Gget_objinfo(loc_id, name, follow_link, &stat)
    if retval < 0:
        raise GroupError("Can't stat member \"%s\" of group %d" % (name, loc_id))

    statobj = GroupStat()
    statobj.fileno = (stat.fileno[0], stat.fileno[1])
    statobj.objno = (stat.objno[0], stat.objno[1])
    statobj.nlink = <int>stat.nlink
    statobj.type = <int>stat.type
    statobj.mtime = stat.mtime
    statobj.linklen = stat.linklen

    return statobj

cdef herr_t iter_cb_helper(hid_t gid, char *name, object int_tpl):

    cdef object func
    cdef object data
    cdef object outval

    func = int_tpl[0]
    data = int_tpl[1]
    exc_list = int_tpl[2]

    try:
        func(gid, name, data)
    except StopIteration:
        return 1
    except Exception, e:
        exc_list.append(e)
        return -1

    return 0

def iterate(hid_t loc_id, char* name, object func, object data=None, int startidx=0):
    """ (INT loc_id, STRING name, FUNCTION func, OBJECT data=None, 
            UINT startidx=0) => INT last_index_processed

        Iterate an arbitrary Python function over a group.  Note that the
        group is specified by a parent and a name; if you have a group
        identifier and want to iterate over it; pass in "." for the name.

        You can also start at an arbitrary member by specifying its 
        (zero-based) index.  The return value is the index of the last 
        group member processed.

        Your function:
        1.  Should accept three arguments: the (INT) id of the group, the 
            (STRING) name of the member, and an arbitary Python object you 
            provide as data.  Any return value is ignored.
        2.  Raise StopIteration to bail out before all members are processed.
        3.  Raising anything else immediately aborts iteration, and the
            exception is propagated.
    """
    cdef int i
    cdef herr_t retval

    i = startidx

    int_tpl = (func, data, [])

    retval = H5Giterate(loc_id, name, &i, <H5G_iterate_t>iter_cb_helper, int_tpl)

    if retval < 0:
        if len(int_tpl[2]) != 0:
            raise int_tpl[2][0]
        raise GroupError("Error occured during iteration")
    return i-2

# === Custom extensions =======================================================

def py_listnames(hid_t group_id):
    """ (INT group_id) => LIST names_list

        Create a Python list of the object names directly attached to a group.
    """
    cdef int nitems
    cdef object thelist
    cdef int i

    thelist = []
    nitems = get_num_objs(group_id)

    for i from 0 <= i < nitems:
        thelist.append(get_objname_by_idx(group_id, i))

    return thelist

cdef class _GroupIterator:

    """ Iterator object which yields names of group members.
        These objects are created by py_iternames; don't create them yourself.
    """

    cdef hid_t gid
    cdef int idx
    cdef int nitems

    def __init__(self, int gid):
        self.gid = gid
        self.idx = 0
        self.nitems = get_num_objs(gid)

    def __next__(self):
        cdef hsize_t nobjs
        nobjs = -1
        H5Gget_num_objs(self.gid, &nobjs)
        if nobjs != self.nitems:
            raise GroupError("Group length changed during iteration")
        if self.idx >= self.nitems:
            raise StopIteration()
        name = get_objname_by_idx(self.gid, self.idx)
        self.idx  = self.idx + 1
        return name

    def __iter__(self):
        return self

def py_iternames(hid_t group_id):
    """ (INT group_id) => ITERATOR names_iterator

        Create an iterator object which yields names attached to the current
        group.  Mutating group members is OK, but do *NOT* change the group 
        membership while iterating over it.
    """
    return _GroupIterator(group_id)

def py_exists(hid_t group_id, char* name, int follow_link=1):
    """ (INT group_id, STRING name, BOOL follow_link=True) => BOOL exists

        Determine if a named member exists in the given group.  If follow_link
        is True (default), symbolic links will be dereferenced.
    """
    cdef int retval
    retval = H5Gget_objinfo(group_id, name, follow_link, NULL)
    if retval < 0:
        return False
    return True































    


