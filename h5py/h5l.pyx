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
from utils cimport pybool

"""
    API for the "H5L" family of link-related operations
"""


cdef extern from "hdf5.h":

  int   H5Iinc_ref(hid_t obj_id) except *    
  int   H5P_DEFAULT

cdef class LinkProxy(ObjectID):

    """
        Proxy class which provides access to the HDF5 "H5L" API.

        These come attached to GroupID objects as "obj.links".  Since every
        H5L function operates on at least one group, the methods provided
        operate on their parent group operator.  For example:

        >>> g = h5g.open(fid, '/')
        >>> g.links.exists("MyGroup")
        True
        >>> g.links.exists("FooBar")
        False

    """

    def __cinit__(self, hid_t id_):
        # At this point the ObjectID constructor has already been called.

        # We need to manually incref the identifier because it's now
        # shared by both this object and its parent GroupID object.
        H5Iinc_ref(self.id)

    def exists(self, char* name):
        """ (STRING name) => BOOL

            Check if a link of the specified name exists in this group.
        """
        return pybool(H5Lexists(self.id, name, H5P_DEFAULT))




