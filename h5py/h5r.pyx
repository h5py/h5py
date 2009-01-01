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
__doc__ = \
"""
    H5R API for object and region references.
"""

include "config.pxi"

# Pyrex compile-time imports
from h5 cimport init_hdf5, ObjectID
from h5i cimport wrap_identifier
from h5s cimport SpaceID

# Initialization
init_hdf5()

# Runtime imports
from _sync import sync, nosync

# === Public constants and data structures ====================================

OBJECT = H5R_OBJECT
DATASET_REGION = H5R_DATASET_REGION


# === Reference API ===========================================================

@sync
def create(ObjectID loc not None, char* name, int ref_type, SpaceID space=None):
    """(ObjectID loc, STRING name, INT ref_type, SpaceID space=None)
    => ReferenceObject ref

    Create a new reference. The value of ref_type detemines the kind
    of reference created:

    OBJECT
        Reference to an object in an HDF5 file.  Parameters "loc"
        and "name" identify the object; "space" is unused.

    DATASET_REGION    
        Reference to a dataset region.  Parameters "loc" and
        "name" identify the dataset; the selection on "space"
        identifies the region.
    """
    cdef hid_t space_id
    cdef Reference ref
    ref = Reference()
    if space is None:
        space_id = -1
    else:
        space_id = space.id

    H5Rcreate(&ref.ref, loc.id, name, <H5R_type_t>ref_type, space_id)
    ref.typecode = ref_type

    return ref

cdef class Reference:

    """ 
        Represents an HDF5 reference.

        Objects of this class are created exclusively by the library and 
        cannot be modified.  The read-only attribute "typecode" determines 
        whether the reference is to an object in an HDF5 file (OBJECT) 
        or a dataset region (DATASET_REGION).
    """

    @sync
    def dereference(self, ObjectID id not None):
        """(ObjectID id) => ObjectID

        Open the object pointed to by this reference and return its
        identifier.  The file identifier (or the identifier for any object
        in the file) must also be provided.

        The reference type may be either OBJECT or DATASET_REGION.
        """
        return wrap_identifier(H5Rdereference(id.id, <H5R_type_t>self.typecode, &self.ref))

    @sync
    def get_region(self, ObjectID id not None):
        """(ObjectID id) => SpaceID

        Retrieve the dataspace selection pointed to by this reference.
        Returns a copy of the dataset's dataspace, with the appropriate
        elements selected.  The file identifier or the identifier of any
        object in the file (including the dataset itself) must also be
        provided.

        The reference object must be of type DATASET_REGION.
        """
        return SpaceID(H5Rget_region(id.id, <H5R_type_t>self.typecode, &self.ref))

    @sync
    def get_obj_type(self, ObjectID id not None):
        """(ObjectID id) => INT obj_code

        Determine what type of object this eference points to.  The
        reference may be either type OBJECT or DATASET_REGION.  The file
        identifier or the identifier of any object in the file must also
        be provided.

        The return value is one of:

        - h5g.LINK
        - h5g.GROUP
        - h5g.DATASET
        - h5g.TYPE
        """
        return <int>H5Rget_obj_type(id.id, <H5R_type_t>self.typecode, &self.ref)

    @nosync
    def __str__(self):
        if self.typecode == H5R_OBJECT:
            return "HDF5 object reference"
        elif self.typecode == H5R_DATASET_REGION:
            return "HDF5 dataset region reference"

        return "Unknown HDF5 reference"

    def __repr__(self):
        return self.__str__()














