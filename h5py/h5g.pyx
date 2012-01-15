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

# Compile-time imports
from _objects cimport pdefault
from utils cimport emalloc, efree
from h5p cimport PropID
cimport _hdf5 # to implement container testing for 1.6

import _objects

# === Public constants and data structures ====================================

# Enumerated object types for groups "H5G_obj_t"
UNKNOWN  = H5G_UNKNOWN
LINK     = H5G_LINK
GROUP    = H5G_GROUP
DATASET  = H5G_DATASET
TYPE = H5G_TYPE

# Enumerated link types "H5G_link_t"
LINK_ERROR = H5G_LINK_ERROR
LINK_HARD  = H5G_LINK_HARD
LINK_SOFT  = H5G_LINK_SOFT

cdef class GroupStat:
    """Represents the H5G_stat_t structure containing group member info.

    Fields (read-only):

    * fileno:   2-tuple uniquely identifying the current file
    * objno:    2-tuple uniquely identifying this object
    * nlink:    Number of hard links to this object
    * mtime:    Modification time of this object
    * linklen:  Length of the symbolic link name, or 0 if not a link.

    "Uniquely identifying" means unique among currently open files,
    not universally unique.

    * Hashable: Yes
    * Equality: Yes
    """
    cdef H5G_stat_t infostruct

    property fileno:
        def __get__(self):
            return (self.infostruct.fileno[0], self.infostruct.fileno[1])
    property objno:
        def __get__(self):
            return (self.infostruct.objno[0], self.infostruct.objno[1])
    property nlink:
        def __get__(self):
            return self.infostruct.nlink
    property type:
        def __get__(self):
            return self.infostruct.type
    property mtime:
        def __get__(self):
            return self.infostruct.mtime
    property linklen:
        def __get__(self):
            return self.infostruct.linklen

    def _hash(self):
        return hash((self.fileno, self.objno, self.nlink, self.type, self.mtime, self.linklen))


cdef class GroupIter:

    """
        Iterator over the names of group members.  After this iterator is
        exhausted, it releases its reference to the group ID.
    """

    cdef unsigned long idx
    cdef unsigned long nobjs
    cdef GroupID grp

    def __init__(self, GroupID grp not None):
        self.idx = 0
        self.grp = grp
        self.nobjs = grp.get_num_objs()

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == self.nobjs:
            self.grp = None
            raise StopIteration

        retval = self.grp.get_objname_by_idx(self.idx)
        self.idx = self.idx + 1
        return retval

# === Basic group management ==================================================


def open(ObjectID loc not None, char* name):
    """(ObjectID loc, STRING name) => GroupID

    Open an existing HDF5 group, attached to some other group.
    """
    return GroupID.open(H5Gopen(loc.id, name))

def create(ObjectID loc not None, object name, PropID lcpl=None,
           PropID gcpl=None):
    """(ObjectID loc, STRING name or None, PropLCID lcpl=None,
        PropGCID gcpl=None)
    => GroupID

    Create a new group, under a given parent group.  If name is None,
    an anonymous group will be created in the file.
    """
    cdef hid_t gid
    cdef char* cname = NULL
    if name is not None:
        cname = name

    if cname != NULL:
        gid = H5Gcreate2(loc.id, cname, pdefault(lcpl), pdefault(gcpl), H5P_DEFAULT)
    else:
        gid = H5Gcreate_anon(loc.id, pdefault(gcpl), H5P_DEFAULT)

    return GroupID.open(gid)


cdef class _GroupVisitor:

    cdef object func
    cdef object retval

    def __init__(self, func):
        self.func = func
        self.retval = None

cdef herr_t cb_group_iter(hid_t gid, char *name, void* vis_in) except 2:

    cdef _GroupVisitor vis = <_GroupVisitor>vis_in

    vis.retval = vis.func(name)

    if vis.retval is not None:
        return 1
    return 0


def iterate(GroupID loc not None, object func, int startidx=0, *,
            char* obj_name='.'):
    """ (GroupID loc, CALLABLE func, UINT startidx=0, **kwds)
    => Return value from func

    Iterate a callable (function, method or callable object) over the
    members of a group.  Your callable should have the signature::

        func(STRING name) => Result

    Returning None continues iteration; returning anything else aborts
    iteration and returns that value. Keywords:

    STRING obj_name (".")
        Iterate over this subgroup instead
    """
    if startidx < 0:
        raise ValueError("Starting index must be non-negative")

    cdef int i = startidx
    cdef _GroupVisitor vis = _GroupVisitor(func)

    H5Giterate(loc.id, obj_name, &i, <H5G_iterate_t>cb_group_iter, <void*>vis)

    return vis.retval


def get_objinfo(ObjectID obj not None, object name='.', int follow_link=1):
    """(ObjectID obj, STRING name='.', BOOL follow_link=True) => GroupStat object

    Obtain information about a named object.  If "name" is provided,
    "obj" is taken to be a GroupID object containing the target.
    The return value is a GroupStat object; see that class's docstring
    for a description of its attributes.

    If follow_link is True (default) and the object is a symbolic link,
    the information returned describes its target.  Otherwise the
    information describes the link itself.
    """
    cdef GroupStat statobj
    statobj = GroupStat()
    cdef char* _name
    _name = name

    H5Gget_objinfo(obj.id, _name, follow_link, &statobj.infostruct)

    return statobj

# === Group member management =================================================

cdef class GroupID(ObjectID):

    """
        Represents an HDF5 group identifier

        Python extensions:

        __contains__
            Test for group member ("if name in grpid")

        __iter__
            Get an iterator over member names

        __len__
            Number of members in this group; len(grpid) = N

        If HDF5 1.8.X is used, the attribute "links" contains a proxy object
        providing access to the H5L family of routines.  See the docs
        for h5py.h5l.LinkProxy for more information.

        * Hashable: Yes, unless anonymous
        * Equality: True HDF5 identity unless anonymous
    """

    def __init__(self, hid_t id_):
        import h5l
        self.links = h5l.LinkProxy(id_)


    def _close(self):
        """()

        Terminate access through this identifier.  You shouldn't have to
        call this manually; group identifiers are automatically released
        when their Python wrappers are freed.
        """
        with _objects.registry.lock:
            H5Gclose(self.id)
            if not self.valid:
                del _objects.registry[self.id]


    def link(self, char* current_name, char* new_name,
             int link_type=H5G_LINK_HARD, GroupID remote=None):
        """(STRING current_name, STRING new_name, INT link_type=LINK_HARD,
        GroupID remote=None)

        Create a new hard or soft link.  current_name identifies
        the link target (object the link will point to).  The new link is
        identified by new_name and (optionally) another group "remote".

        Link types are:

        LINK_HARD
            Hard link to existing object (default)

        LINK_SOFT
            Symbolic link; link target need not exist.
        """
        cdef hid_t remote_id
        if remote is None:
            remote_id = self.id
        else:
            remote_id = remote.id

        H5Glink2(self.id, current_name, <H5G_link_t>link_type, remote_id, new_name)


    def unlink(self, char* name):
        """(STRING name)

        Remove a link to an object from this group.
        """
        H5Gunlink(self.id, name)


    def move(self, char* current_name, char* new_name, GroupID remote=None):
        """(STRING current_name, STRING new_name, GroupID remote=None)

        Relink an object.  current_name identifies the object.
        new_name and (optionally) another group "remote" determine
        where it should be moved.
        """
        cdef hid_t remote_id
        if remote is None:
            remote_id = self.id
        else:
            remote_id = remote.id

        H5Gmove2(self.id, current_name, remote_id, new_name)


    def get_num_objs(self):
        """() => INT number_of_objects

        Get the number of objects directly attached to a given group.
        """
        cdef hsize_t size
        H5Gget_num_objs(self.id, &size)
        return size


    def get_objname_by_idx(self, hsize_t idx):
        """(INT idx) => STRING

        Get the name of a group member given its zero-based index.
        """
        cdef int size
        cdef char* buf
        buf = NULL

        size = H5Gget_objname_by_idx(self.id, idx, NULL, 0)

        buf = <char*>emalloc(sizeof(char)*(size+1))
        try:
            H5Gget_objname_by_idx(self.id, idx, buf, size+1)
            pystring = buf
            return pystring
        finally:
            efree(buf)


    def get_objtype_by_idx(self, hsize_t idx):
        """(INT idx) => INT object_type_code

        Get the type of an object attached to a group, given its zero-based
        index.  Possible return values are:

        - LINK
        - GROUP
        - DATASET
        - TYPE
        """
        return <int>H5Gget_objtype_by_idx(self.id, idx)


    def get_linkval(self, char* name):
        """(STRING name) => STRING link_value

        Retrieve the value (target name) of a symbolic link.
        Limited to 2048 characters on Windows.
        """
        cdef char* value
        cdef H5G_stat_t statbuf
        value = NULL

        H5Gget_objinfo(self.id, name, 0, &statbuf)

        if statbuf.type != H5G_LINK:
            raise ValueError('"%s" is not a symbolic link.' % name)

        IF UNAME_SYSNAME == "Windows":
            linklen = 2049  # Windows statbuf.linklen seems broken
        ELSE:
            linklen = statbuf.linklen+1
        value = <char*>emalloc(sizeof(char)*linklen)
        try:
            H5Gget_linkval(self.id, name, linklen, value)
            value[linklen-1] = c'\0'  # in case HDF5 doesn't null terminate on Windows
            pyvalue = value
            return pyvalue
        finally:
            efree(value)


    def set_comment(self, char* name, char* comment):
        """(STRING name, STRING comment)

        Set the comment on a group member.
        """
        H5Gset_comment(self.id, name, comment)


    def get_comment(self, char* name):
        """(STRING name) => STRING comment

        Retrieve the comment for a group member.
        """
        cdef int cmnt_len
        cdef char* cmnt
        cmnt = NULL

        cmnt_len = H5Gget_comment(self.id, name, 0, NULL)
        assert cmnt_len >= 0

        cmnt = <char*>emalloc(sizeof(char)*(cmnt_len+1))
        try:
            H5Gget_comment(self.id, name, cmnt_len+1, cmnt)
            py_cmnt = cmnt
            return py_cmnt
        finally:
            efree(cmnt)

    # === Special methods =====================================================


    def __contains__(self, char* name):
        """(STRING name)

        Determine if a group member of the given name is present
        """
        cdef herr_t retval
        retval = _hdf5.H5Gget_objinfo(self.id, name, 0, NULL)

        return bool(retval >= 0)


    def __iter__(self):
        """ Return an iterator over the names of group members. """
        return GroupIter(self)


    def __len__(self):
        """ Number of group members """
        cdef hsize_t size
        H5Gget_num_objs(self.id, &size)
        return size
