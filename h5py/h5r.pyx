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

# Pyrex compile-time imports
from h5g cimport H5G_obj_t

# Runtime imports
import h5
from h5 import DDict

# === Public constants and data structures ====================================

OBJECT = H5R_OBJECT
REGION = H5R_DATASET_REGION

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
        whether the reference is to an object in an HDF5 file (TYPE_OBJECT) 
        or a dataspace region (TYPE_REGION).
    """

    cdef ref_u ref
    cdef readonly int typecode

    def __str__(self):
        return "HDF5 reference (type %s)" % TYPE_MAPPER[self.typecode]

    def __repr__(self):
        return self.__str__()

# === Reference API ===========================================================

def create(hid_t loc_id, char* name, int ref_type, hid_t space_id=-1):
    """ (INT loc_id, STRING name, INT ref_type, INT space_id=0)
        => ReferenceObject ref

        Create a new reference, either to an object or a dataset region.
    """
    cdef Reference ref
    ref = Reference()

    H5Rcreate(&ref.ref, loc_id, name, <H5R_type_t>ref_type, space_id)
    ref.typecode = ref_type

    return ref

def dereference(hid_t obj_id, Reference ref):
    """ (INT obj_id, ReferenceObject ref) => INT obj_id

        Open the object pointed to by "ref" and return its identifier.  The
        parameter "obj_id" may be the file ID or the ID of any object which
        lives in the file.
    """
    return H5Rdereference(obj_id, <H5R_type_t>ref.typecode, &ref.ref)

def get_region(hid_t container_id, Reference ref):
    """ (INT container_id, Reference ref) => INT dataspace_id

        Retrieve the dataspace selection pointed to by a reference.  The given
        reference must be in the dataset indicated by container_id.  Returns
        an identifier for a copy of the dataspace for the dataset pointed to
        by "ref", with the appropriate elements selected.

        The given reference object must be of type TYPE_REGION.
    """
    return H5Rget_region(container_id, <H5R_type_t>ref.typecode, &ref.ref)

def get_obj_type(hid_t container_id, Reference ref):
    """ (INT container_id, Reference ref) => INT obj_code

        Determine what type of object an object reference points to.  The
        reference may be either TYPE_OBJECT or TYPE_REGION.
        
        The return value is one of:
        h5g.OBJ_LINK        Symbolic link
        h5g.OBJ_GROUP       Group
        h5g.OBJ_DATASET     Dataset
        h5g.OBJ_TYPE        Named datatype
    """
    return <int>H5Rget_obj_type(container_id, <H5R_type_t>ref.typecode, &ref.ref)











