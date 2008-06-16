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
    H5R API for object and region references.
"""

# Pyrex compile-time imports
from h5 cimport ObjectID
from h5g cimport H5G_obj_t

# Runtime imports
import h5
from h5 import DDict

# === Public constants and data structures ====================================

OBJECT = H5R_OBJECT
DATASET_REGION = H5R_DATASET_REGION

PY_TYPE = {H5R_OBJECT: 'OBJECT',  H5R_DATASET_REGION: 'DATASET REGION' }
PY_TYPE = DDict(PY_TYPE)

cdef union ref_u:
    hobj_ref_t         obj_ref
    hdset_reg_ref_t    reg_ref

cdef class Reference:

    """ 
        Represents an HDF5 reference.

        Objects of this class are created exclusively by the library and 
        cannot be modified.  The read-only attribute "typecode" determines 
        whether the reference is to an object in an HDF5 file (OBJECT) 
        or a dataset region (DATASET_REGION).
    """

    cdef ref_u ref
    cdef readonly int typecode

    def __str__(self):
        return "HDF5 reference (type %s)" % PY_TYPE[self.typecode]

    def __repr__(self):
        return self.__str__()

# === Reference API ===========================================================

def create(ObjectID loc_id not None, char* name, int ref_type, SpaceID space=None):
    """ (ObjectID loc_id, STRING name, INT ref_type, SpaceID space=None)
        => ReferenceObject ref

        Create a new reference. The value of ref_type detemines the kind
        of reference created:

        - OBJECT    Reference to an object in an HDF5 file.  Parameters loc_id
                    and name identify the object; space_id is unused.

        - DATASET_REGION    
                    Reference to a dataset region.  Parameters loc_id and
                    name identify the dataset; the selection on space_id
                    identifies the region.
    """
    cdef hid_t space_id
    cdef Reference ref
    ref = Reference()
    if space is None:
        space_id = -1
    else:
        space_id = space.id

    H5Rcreate(&ref.ref, loc_id.id, name, <H5R_type_t>ref_type, space_id)
    ref.typecode = ref_type

    return ref

def dereference(ObjectID file_id not None, Reference ref):
    """ (ObjectID file_id, ReferenceObject ref) => INT obj_id

        Open the object pointed to by "ref" and return its identifier.
        The containing file must be provided via file_id, which can be
        a file identifier or an identifier for any object which lives
        in the file.
    """
    return H5Rdereference(file_id.id, <H5R_type_t>ref.typecode, &ref.ref)

def get_region(ObjectID dataset_id not None, Reference ref):
    """ (ObjectID dataset_id, Reference ref) => INT dataspace_id

        Retrieve the dataspace selection pointed to by a reference.
        Returns a copy of the dataset's dataspace, with the appropriate
        elements selected.

        The given reference object must be of type DATASET_REGION.
    """
    return H5Rget_region(dataset_id.id, <H5R_type_t>ref.typecode, &ref.ref)

def get_obj_type(Object ID ds_id not None, Reference ref):
    """ (ObjectID ds_id, Reference ref) => INT obj_code

        Determine what type of object an object reference points to.  The
        reference may be either type OBJECT or DATASET_REGION.  For 
        DATASET_REGION, the parameter ds_id must be either the dataset 
        identifier, or the identifier for the object within which the
        dataset is contained.
        
        The return value is one of:
        h5g.LINK        Symbolic link
        h5g.GROUP       Group
        h5g.DATASET     Dataset
        h5g.TYPE        Named datatype
    """
    return <int>H5Rget_obj_type(ds_id.id, <H5R_type_t>ref.typecode, &ref.ref)











