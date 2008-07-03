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
from h5 import DDict, H5Error

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
    """ (ObjectID loc, STRING name) => GroupID

        Open an existing HDF5 group, attached to some other group.
    """
    return GroupID(H5Gopen(loc.id, name))

def create(ObjectID loc not None, char* name, int size_hint=-1):
    """ (ObjectID loc, STRING name, INT size_hint=-1) => GroupID

        Create a new group, under a given parent group.  If given, size_hint
        is an estimate of the space to reserve (in bytes) for group member
        names.
    """
    return GroupID(H5Gcreate(loc.id, name, size_hint))

cdef herr_t iter_cb_helper(hid_t gid, char *name, object int_tpl) except -1:
    # Callback function for H5Giterate

    loc, func, data = int_tpl

    # An unhandled exception (anything except StopIteration) will 
    # cause Pyrex to immediately return -1, which stops H5Giterate.
    try:
        func(loc, name, data)
    except StopIteration:
        return 1

    return 0

def iterate(GroupID loc not None, char* name, object func, object data=None, 
            int startidx=0):
    """ (GroupID loc, STRING name, FUNCTION func, OBJECT data=None, 
            UINT startidx=0) => INT last_index_processed

        Iterate an arbitrary Python function over a group.  Note that the
        group is specified by a parent and a name; if you have a group
        identifier and want to iterate over it; pass in "." for the name.
        You can also start at an arbitrary member by specifying its 
        (zero-based) index.

        Your function:
        1.  Should accept three arguments: the GroupID of the group, the 
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

    int_tpl = (loc, func, data)

    H5Giterate(loc.id, name, &i, <H5G_iterate_t>iter_cb_helper, int_tpl)

# === Group member management =================================================

cdef class GroupID(ObjectID):

    """
        Represents an HDF5 group identifier
    """

    def close(self):
        """ ()

            Terminate access through this identifier.  You shouldn't have to
            call this manually; group identifiers are automatically released
            when their Python wrappers are freed.
        """
        H5Gclose(self.id)

    def link(self, char* current_name, char* new_name, 
             int link_type=H5G_LINK_HARD, GroupID remote=None):
        """ (STRING current_name, STRING new_name, INT link_type=LINK_HARD, 
             GroupID remote=None)

            Create a new hard or soft link.  current_name identifies
            the link target (object the link will point to).  The new link is
            identified by new_name and (optionally) another group "remote".

            Link types are:
                LINK_HARD:  Hard link to existing object (default)
                LINK_SOFT:  Symbolic link; link target need not exist.
        """
        cdef hid_t remote_id
        if remote is None:
            remote_id = self.id
        else:
            remote_id = remote.id

        H5Glink2(self.id, current_name, <H5G_link_t>link_type, remote_id, new_name)

    def unlink(self, char* name):
        """ (STRING name)

            Remove a link to an object from this group.
        """
        H5Gunlink(self.id, name)

    def move(self, char* current_name, char* new_name, GroupID remote=None):
        """ (STRING current_name, STRING new_name, GroupID remote=None)

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
        """ () => INT number_of_objects

            Get the number of objects directly attached to a given group.
        """
        cdef hsize_t size
        H5Gget_num_objs(self.id, &size)
        return size

    def get_objname_by_idx(self, hsize_t idx):
        """ (INT idx) => STRING object_name

            Get the name of a group member given its zero-based index.

            Due to a limitation of the HDF5 library, the generic exception
            H5Error (errno 1) is raised if the idx parameter is out-of-range.
        """
        cdef int size
        cdef char* buf
        buf = NULL

        # This function does not properly raise an exception
        size = H5Gget_objname_by_idx(self.id, idx, NULL, 0)
        if size < 0:
            raise H5Error((1,"Invalid argument"))

        buf = <char*>emalloc(sizeof(char)*(size+1))
        try:
            H5Gget_objname_by_idx(self.id, idx, buf, size+1)
            pystring = buf
            return pystring
        finally:
            efree(buf)

    def get_objtype_by_idx(self, hsize_t idx):
        """ (INT idx) => INT object_type_code

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
        retval = H5Gget_objtype_by_idx(self.id, idx)
        if retval < 0:
            raise H5Error((1,"Invalid argument."))
        return retval

    def get_objinfo(self, char* name, int follow_link=1):
        """ (STRING name, BOOL follow_link=True) => GroupStat object

            Obtain information about an arbitrary object attached to a group. 
            The return value is a GroupStat object; see that class's docstring
            for a description of its attributes.  

            If follow_link is True (default) and the object is a symbolic link, 
            the information returned describes its target.  Otherwise the 
            information describes the link itself.
        """
        cdef H5G_stat_t stat
        cdef GroupStat statobj

        H5Gget_objinfo(self.id, name, follow_link, &stat)

        statobj = GroupStat()
        statobj.fileno = (stat.fileno[0], stat.fileno[1])
        statobj.objno = (stat.objno[0], stat.objno[1])
        statobj.nlink = stat.nlink
        statobj.type = stat.type
        statobj.mtime = stat.mtime
        statobj.linklen = stat.linklen

        return statobj


    def get_linkval(self, char* name):
        """ (STRING name) => STRING link_value

            Retrieve the value (target name) of a symbolic link.
        """
        cdef char* value
        cdef H5G_stat_t statbuf
        value = NULL

        H5Gget_objinfo(self.id, name, 0, &statbuf)

        if statbuf.type != H5G_LINK:
            raise ValueError('"%s" is not a symbolic link.' % name)

        value = <char*>emalloc(sizeof(char)*(statbuf.linklen+1))
        try:
            H5Gget_linkval(self.id, name, statbuf.linklen+1, value)
            pyvalue = value
            return pyvalue

        finally:
            efree(value)

    def set_comment(self, char* name, char* comment):
        """ (STRING name, STRING comment)

            Set the comment on a group member.
        """
        H5Gset_comment(self.id, name, comment)

    def get_comment(self, char* name):
        """ (STRING name) => STRING comment

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

    def py_exists(self, char* name):

        try:
            self.get_objinfo(name)
        except H5Error:
            return False    
        return True

    def py_iter(self):
        """ () => ITERATOR

            Return an iterator over the names of group members.
        """
        return GroupIter(self)


