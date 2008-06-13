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
from utils cimport emalloc, efree

# Runtime imports
import h5
from h5 import DDict
from h5e import H5Error

# === Public constants and data structures ====================================

# Enumerated object types for groups "H5G_obj_t"
UNKNOWN  = H5G_UNKNOWN
LINK     = H5G_LINK
GROUP    = H5G_GROUP
DATASET  = H5G_DATASET
DATATYPE = H5G_TYPE

# Enumerated link types "H5G_link_t"
LINK_ERROR = H5G_LINK_ERROR
LINK_HARD  = H5G_LINK_HARD
LINK_SOFT  = H5G_LINK_SOFT

cdef class GroupStat:
    """ Represents the H5G_stat_t structure containing group member info.

        Fields (read-only):
        fileno  ->  2-tuple uniquely* identifying the current file
        objno   ->  2-tuple uniquely* identifying this object
        nlink   ->  Number of hard links to this object
        mtime   ->  Modification time of this object
        linklen ->  Length of the symbolic link name, or 0 if not a link.

        *"Uniquely identifying" means unique among currently open files, 
        not universally unique.
    """
    cdef readonly object fileno  # will be a 2-tuple
    cdef readonly object objno   # will be a 2-tuple
    cdef readonly unsigned int nlink
    cdef readonly int type
    cdef readonly time_t mtime
    cdef readonly size_t linklen


# === Basic group management ==================================================

def open(hid_t loc_id, char* name):
    """ (INT loc_id, STRING name)

        Open an existing HDF5 group, attached to some other group.
    """
    return H5Gopen(loc_id, name)

def close(hid_t group_id):
    """ (INT group_id)
    """
    H5Gclose(group_id)

def create(hid_t loc_id, char* name, int size_hint=-1):
    """ (INT loc_id, STRING name, INT size_hint=-1)

        Create a new group, under a given parent group.  If given, size_hint
        is an estimate of the space to reserve (in bytes) for group member
        names.
    """
    return H5Gcreate(loc_id, name, size_hint)

# === Group member management =================================================

def link(hid_t loc_id, char* current_name, char* new_name, int link_type=H5G_LINK_HARD, hid_t remote_id=-1):
    """ ( INT loc_id, STRING current_name, STRING new_name, 
          INT link_type=LINK_HARD, INT remote_id=-1) 

        Create a new hard or soft link.  loc_id and current_name identify
        the link target (object the link will point to).  The new link is
        identified by new_name and (optionally) another group id "remote_id".

        Link types are:
            LINK_HARD:  Hard link to existing object (default)
            LINK_SOFT:  Symbolic link; link target need not exist.
    """
    if remote_id < 0:
        remote_id = loc_id

    H5Glink2(loc_id, current_name, <H5G_link_t>link_type, remote_id, new_name)

def unlink(hid_t loc_id, char* name):
    """ (INT loc_id, STRING name)

        Remove a link to an object from the given group.
    """
    H5Gunlink(loc_id, name)

def move(hid_t loc_id, char* current_name, char* new_name, hid_t remote_id=-1):
    """ (INT loc_id, STRING current_name, STRING new_name, INT remote_id=-1)

        Relink an object.  loc_id and current_name identify the object.
        new_name and (optionally) another group id "remote_id" determine
        where it should be moved.
    """
    if remote_id < 0:
        remote_id = loc_id
    H5Gmove2(loc_id, current_name, remote_id, new_name)


# === Member inspection =======================================================

def get_num_objs(hid_t loc_id):
    """ (INT loc_id) => INT number_of_objects

        Get the number of objects attached to a given group.
    """
    cdef hsize_t size
    H5Gget_num_objs(loc_id, &size)
    return size

def get_objname_by_idx(hid_t loc_id, hsize_t idx):
    """ (INT loc_id, INT idx) => STRING object_name

        Get the name of a group member given its zero-based index.

        Due to a limitation of the HDF5 library, the generic exception
        H5Error (errno 1) is raised if the idx parameter is out-of-range.
    """
    cdef int size
    cdef char* buf
    buf = NULL

    # This function does not properly raise an exception
    size = H5Gget_objname_by_idx(loc_id, idx, NULL, 0)
    if size < 0:
        raise H5Error((1,"Invalid argument"))

    buf = <char*>emalloc(sizeof(char)*(size+1))
    try:
        H5Gget_objname_by_idx(loc_id, idx, buf, size+1)
        pystring = buf
        return pystring
    finally:
        efree(buf)

def get_objtype_by_idx(hid_t loc_id, hsize_t idx):
    """ (INT loc_id, INT idx) => INT object_type_code

        Get the type of an object attached to a group, given its zero-based
        index.  Possible return values are:
            - LINK
            - GROUP
            - DATASET
            - DATATYPE

        Due to a limitation of the HDF5 library, the generic exception
        H5Error (errno 1) is raised if the idx parameter is out-of-range.
    """
    # This function does not properly raise an exception
    cdef herr_t retval
    retval = H5Gget_objtype_by_idx(loc_id, idx)
    if retval < 0:
        raise H5Error((0,"Invalid argument."))
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
    cdef H5G_stat_t stat
    cdef GroupStat statobj

    H5Gget_objinfo(loc_id, name, follow_link, &stat)

    statobj = GroupStat()
    statobj.fileno = (stat.fileno[0], stat.fileno[1])
    statobj.objno = (stat.objno[0], stat.objno[1])
    statobj.nlink = stat.nlink
    statobj.type = stat.type
    statobj.mtime = stat.mtime
    statobj.linklen = stat.linklen

    return statobj

cdef herr_t iter_cb_helper(hid_t gid, char *name, object int_tpl) except -1:
    # Callback function for H5Giterate

    func = int_tpl[0]
    data = int_tpl[1]

    # An unhandled exception (anything except StopIteration) will 
    # cause Pyrex to immediately return -1, which stops H5Giterate.
    try:
        func(gid, name, data)
    except StopIteration:
        return 1

    return 0

def iterate(hid_t loc_id, char* name, object func, object data=None, int startidx=0):
    """ (INT loc_id, STRING name, FUNCTION func, OBJECT data=None, 
            UINT startidx=0) => INT last_index_processed

        Iterate an arbitrary Python function over a group.  Note that the
        group is specified by a parent and a name; if you have a group
        identifier and want to iterate over it; pass in "." for the name.
        You can also start at an arbitrary member by specifying its 
        (zero-based) index.

        Your function:
        1.  Should accept three arguments: the (INT) id of the group, the 
            (STRING) name of the member, and an arbitary Python object you 
            provide as data.  Any return value is ignored.
        2.  Raise StopIteration to bail out before all members are processed.
        3.  Raising anything else immediately aborts iteration, and the
            exception is propagated.
    """
    cdef int i

    if startidx < 0:
        raise ValueError("Starting index must be non-negative.")

    i = startidx
    int_tpl = (func, data)

    H5Giterate(loc_id, name, &i, <H5G_iterate_t>iter_cb_helper, int_tpl)

def get_linkval(hid_t loc_id, char* name):
    """ (INT loc_id, STRING name) => STRING link_value

        Retrieve the value of the given symbolic link.
    """
    cdef char* value
    cdef H5G_stat_t statbuf
    value = NULL

    H5Gget_objinfo(loc_id, name, 0, &statbuf)

    if statbuf.type != H5G_LINK:
        raise ValueError('"%s" is not a symbolic link.' % name)

    value = <char*>emalloc(sizeof(char)*(statbuf.linklen+1))
    try:
        H5Gget_linkval(loc_id, name, statbuf.linklen+1, value)
        pyvalue = value
        return pyvalue

    finally:
        efree(value)


def set_comment(hid_t loc_id, char* name, char* comment):
    """ (INT loc_id, STRING name, STRING comment)

        Set the comment on a group member.
    """
    H5Gset_comment(loc_id, name, comment)


def get_comment(hid_t loc_id, char* name):
    """ (INT loc_id, STRING name) => STRING comment

        Retrieve the comment for a group member.
    """
    cdef int cmnt_len
    cdef char* cmnt
    cmnt = NULL

    cmnt_len = H5Gget_comment(loc_id, name, 0, NULL)
    assert cmnt_len >= 0

    cmnt = <char*>emalloc(sizeof(char)*(cmnt_len+1))
    try:
        H5Gget_comment(loc_id, name, cmnt_len+1, cmnt)
        py_cmnt = cmnt
        return py_cmnt
    finally:
        efree(cmnt)

# === Custom extensions =======================================================

def py_listnames(hid_t group_id):
    """ (INT group_id) => LIST names_list

        Create a Python list of the object names directly attached to a group.
    """
    cdef int nitems
    cdef int i

    namelist = []
    nitems = get_num_objs(group_id)

    for i from 0 <= i < nitems:
        namelist.append(get_objname_by_idx(group_id, i))

    return namelist

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
            raise RuntimeError("Group length changed during iteration")
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
        is True (default), symbolic links will be dereferenced. Note this
        function will not raise an exception if group_id is invalid.
    """
    try:
        H5Gget_objinfo(group_id, name, follow_link, NULL)
    except H5Error:
        return False
    return True

PY_TYPE = DDict({H5G_UNKNOWN: "UNKNOWN OBJ TYPE", 
            H5G_LINK: "LINK", H5G_GROUP: "GROUP",
            H5G_DATASET: "DATASET", H5G_TYPE: "DATATYPE" })
PY_LINK = DDict({H5G_LINK_ERROR: "ERROR", H5G_LINK_HARD: "HARDLINK", 
                H5G_LINK_SOFT: "SOFTLINK" })





























    


