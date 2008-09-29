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
    API for the "H5L" family of link-related operations
"""

include "config.pxi"
include "sync.pxi"

from h5 cimport init_hdf5
init_hdf5()

cdef class LinkProxy(ObjectID):

    """
        Proxy class which provides access to the HDF5 "H5L" API.

        These come attached to GroupID objects as "obj.links".  Since every
        H5L function operates on at least one group, the methods provided
        operate on their parent group identifier.  For example:

        >>> g = h5g.open(fid, '/')
        >>> g.links.exists("MyGroup")
        True
        >>> g.links.exists("FooBar")
        False

        Hashable: No
        Equality: Undefined
    """

    def __cinit__(self, hid_t id_):
        # At this point the ObjectID constructor has already been called.

        # The identifier in question is the hid_t for the parent GroupID.
        # We need to manually incref the identifier because it's now
        # shared by both this object and the parent.
        H5Iinc_ref(self.id)

    def __richcmp__(self, object other, int how):
        return NotImplemented

    def __hash__(self):
        raise TypeError("Link proxies are unhashable; use the parent group instead.")

    @sync
    def exists(self, char* name):
        """ (STRING name) => BOOL

            Check if a link of the specified name exists in this group.
        """
        return <bint>(H5Lexists(self.id, name, H5P_DEFAULT))




