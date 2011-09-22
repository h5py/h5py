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
    API for the "H5L" family of link-related operations.  Defines the class
    LinkProxy, which comes attached to GroupID objects as <obj>.links.
"""

from _objects cimport pdefault
from h5p cimport PropID
from h5g cimport GroupID
from utils cimport emalloc, efree

# === Public constants ========================================================

TYPE_HARD = H5L_TYPE_HARD         
TYPE_SOFT = H5L_TYPE_SOFT      
TYPE_EXTERNAL = H5L_TYPE_EXTERNAL     

cdef class LinkInfo:

    cdef H5L_info_t infostruct

    property type:
        """ Integer type code for link (h5l.TYPE_*) """
        def __get__(self):
            return <int>self.infostruct.type
    property corder_valid:
        """ Indicates if the creation order is valid """
        def __get__(self):
            return <bint>self.infostruct.corder_valid
    property corder:
        """ Creation order """
        def __get__(self):
            return self.infostruct.corder
    property cset:
        """ Integer type code for character set (h5t.CSET_*) """
        def __get__(self):
            return self.infostruct.cset
    property u:
        """ Either the address of a hard link or the size of a soft/UD link """
        def __get__(self):
            if self.infostruct.type == H5L_TYPE_HARD:
                return self.infostruct.u.address
            else:
                return self.infostruct.u.val_size

cdef class _LinkVisitor:

    """ Helper class for iteration callback """

    cdef object func
    cdef object retval
    cdef LinkInfo info

    def __init__(self, func):
        self.func = func
        self.retval = None
        self.info = LinkInfo()

cdef herr_t cb_link_iterate(hid_t grp, char* name, H5L_info_t *istruct, void* data) except 2:
    # Standard iteration callback for iterate/visit routines

    cdef _LinkVisitor it = <_LinkVisitor?>data
    it.info.infostruct = istruct[0]
    it.retval = it.func(name, it.info)
    if (it.retval is None) or (not it.retval):
        return 0
    return 1

cdef herr_t cb_link_simple(hid_t grp, char* name, H5L_info_t *istruct, void* data) except 2:
    # Simplified iteration callback which only provides the name

    cdef _LinkVisitor it = <_LinkVisitor?>data
    it.retval = it.func(name)
    if (it.retval is None) or (not it.retval):
        return 0
    return 1


cdef class LinkProxy:

    """
        Proxy class which provides access to the HDF5 "H5L" API.

        These come attached to GroupID objects as "obj.links".  Since every
        H5L function operates on at least one group, the methods provided
        operate on their parent group identifier.  For example::

            >>> g = h5g.open(fid, '/')
            >>> g.links.exists("MyGroup")
            True
            >>> g.links.exists("FooBar")
            False

        * Hashable: No
        * Equality: Undefined
    """

    def __init__(self, hid_t id_):

        # The identifier in question is the hid_t for the parent GroupID.
        # We "borrow" this reference.
        self.id = id_

    def __richcmp__(self, object other, int how):
        return NotImplemented

    def __hash__(self):
        raise TypeError("Link proxies are unhashable; use the parent group instead.")

    
    def create_hard(self, char* new_name, GroupID cur_loc not None,
        char* cur_name, PropID lcpl=None, PropID lapl=None):
        """ (STRING new_name, GroupID cur_loc, STRING cur_name,
        PropID lcpl=None, PropID lapl=None)

        Create a new hard link in this group pointing to an existing link
        in another group.
        """
        H5Lcreate_hard(cur_loc.id, cur_name, self.id, new_name,
            pdefault(lcpl), pdefault(lapl))

    
    def create_soft(self, char* new_name, char* target,
        PropID lcpl=None, PropID lapl=None):
        """(STRING new_name, STRING target, PropID lcpl=None, PropID lapl=None)

        Create a new soft link in this group, with the given string value.
        The link target does not need to exist.
        """
        H5Lcreate_soft(target, self.id, new_name,
            pdefault(lcpl), pdefault(lapl))

    
    def create_external(self, char* link_name, char* file_name, char* obj_name,
        PropID lcpl=None, PropID lapl=None):
        """(STRING link_name, STRING file_name, STRING obj_name,
        PropLCID lcpl=None, PropLAID lapl=None)

        Create a new external link, pointing to an object in another file.
        """
        H5Lcreate_external(file_name, obj_name, self.id, link_name,
            pdefault(lcpl), pdefault(lapl))

    
    def get_val(self, char* name, PropID lapl=None):
        """(STRING name, PropLAID lapl=None) => STRING or TUPLE(file, obj)

        Get the string value of a soft link, or a 2-tuple representing
        the contents of an external link.
        """
        cdef hid_t plist = pdefault(lapl)
        cdef H5L_info_t info
        cdef size_t buf_size
        cdef char* buf = NULL
        cdef char* ext_file_name = NULL
        cdef char* ext_obj_name = NULL
        cdef unsigned int wtf = 0

        H5Lget_info(self.id, name, &info, plist)
        if info.type != H5L_TYPE_SOFT and info.type != H5L_TYPE_EXTERNAL:
            raise TypeError("Link must be either a soft or external link")

        buf_size = info.u.val_size
        buf = <char*>emalloc(buf_size)
        try:
            H5Lget_val(self.id, name, buf, buf_size, plist)
            if info.type == H5L_TYPE_SOFT:
                py_retval = buf
            else:
                H5Lunpack_elink_val(buf, buf_size, &wtf, &ext_file_name, &ext_obj_name)
                py_retval = (bytes(ext_file_name), bytes(ext_obj_name))
        finally:
            efree(buf)
        
        return py_retval

    
    def exists(self, char* name):
        """ (STRING name) => BOOL

            Check if a link of the specified name exists in this group.
        """
        return <bint>(H5Lexists(self.id, name, H5P_DEFAULT))

    
    def get_info(self, char* name, int index=-1, *, PropID lapl=None):
        """(STRING name=, INT index=, **kwds) => LinkInfo instance

        Get information about a link, either by name or its index.

        Keywords:
        """
        cdef LinkInfo info = LinkInfo()
        H5Lget_info(self.id, name, &info.infostruct, pdefault(lapl))
        return info

    
    def visit(self, object func, *,
              int idx_type=H5_INDEX_NAME, int order=H5_ITER_NATIVE,
              char* obj_name='.', PropID lapl=None, bint info=0):
        """(CALLABLE func, **kwds) => <Return value from func>

        Iterate a function or callable object over all groups below this
        one.  Your callable should conform to the signature::

            func(STRING name) => Result

        or if the keyword argument "info" is True::

            func(STRING name, LinkInfo info) => Result

        Returning None or a logical False continues iteration; returning
        anything else aborts iteration and returns that value.

        BOOL info (False)
            Provide a LinkInfo instance to callback

        STRING obj_name (".")
            Visit this subgroup instead

        PropLAID lapl (None)
            Link access property list for "obj_name"

        INT idx_type (h5.INDEX_NAME)

        INT order (h5.ITER_NATIVE)
        """
        cdef _LinkVisitor it = _LinkVisitor(func)
        cdef H5L_iterate_t cfunc

        if info:
            cfunc = cb_link_iterate
        else:
            cfunc = cb_link_simple

        H5Lvisit_by_name(self.id, obj_name, <H5_index_t>idx_type,
            <H5_iter_order_t>order, cfunc, <void*>it, pdefault(lapl))

        return it.retval





