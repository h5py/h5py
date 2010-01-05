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

include "config.pxi"

# Pyrex compile-time imports
from h5 cimport init_hdf5, ObjectID
from h5i cimport wrap_identifier
from h5s cimport SpaceID

from python cimport PyString_FromStringAndSize

# Initialization
init_hdf5()

# === Public constants and data structures ====================================

OBJECT = H5R_OBJECT
DATASET_REGION = H5R_DATASET_REGION

# === Reference API ===========================================================


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
    if ref_type == H5R_OBJECT:
        ref = Reference()
    elif ref_type == H5R_DATASET_REGION:
        ref = RegionReference()
    else:
        raise ValueError("Unknown reference typecode")
    if space is None:
        space_id = -1
    else:
        space_id = space.id

    H5Rcreate(&ref.ref, loc.id, name, <H5R_type_t>ref_type, space_id)
    ref.typecode = ref_type

    return ref


def dereference(Reference ref not None, ObjectID id not None):
    """(Reference ref, ObjectID id) => ObjectID or None

    Open the object pointed to by the reference and return its
    identifier.  The file identifier (or the identifier for any object
    in the file) must also be provided.  Returns None if the reference
    is zero-filled.

    The reference may be either Reference or RegionReference.
    """
    if not ref:
        return None
    return wrap_identifier(H5Rdereference(id.id, <H5R_type_t>ref.typecode, &ref.ref))


def get_region(RegionReference ref not None, ObjectID id not None):
    """(Reference ref, ObjectID id) => SpaceID or None

    Retrieve the dataspace selection pointed to by the reference.
    Returns a copy of the dataset's dataspace, with the appropriate
    elements selected.  The file identifier or the identifier of any
    object in the file (including the dataset itself) must also be
    provided.

    The reference object must be a RegionReference.  If it is zero-filled,
    returns None.
    """
    if ref.typecode != H5R_DATASET_REGION or not ref:
        return None
    return SpaceID(H5Rget_region(id.id, <H5R_type_t>ref.typecode, &ref.ref))


def get_obj_type(Reference ref not None, ObjectID id not None):
    """(Reference ref, ObjectID id) => INT obj_code or None

    Determine what type of object the reference points to.  The
    reference may be a Reference or RegionReference.  The file
    identifier or the identifier of any object in the file must also
    be provided.

    The return value is one of:

    - h5g.LINK
    - h5g.GROUP
    - h5g.DATASET
    - h5g.TYPE

    If the reference is zero-filled, returns None.
    """
    if not ref:
        return None
    return <int>H5Rget_obj_type(id.id, <H5R_type_t>ref.typecode, &ref.ref)

IF H5PY_18API:
    def get_name(Reference ref not None, ObjectID loc not None):
        """(Reference ref, ObjectID loc) => STRING name

        Determine the name of the object pointed to by this reference.
        Requires the HDF5 1.8 API.
        """
        cdef ssize_t namesize = 0
        cdef char* namebuf = NULL

        namesize = H5Rget_name(loc.id, <H5R_type_t>ref.typecode, &ref.ref, NULL, 0)
        if namesize > 0:
            namebuf = <char*>malloc(namesize+1)
            try:
                namesize = H5Rget_name(loc.id, <H5R_type_t>ref.typecode, &ref.ref, namebuf, namesize+1)
                return namebuf
            finally:
                free(namebuf)

cdef class Reference:

    """ 
        Opaque representation of an HDF5 reference.

        Objects of this class are created exclusively by the library and 
        cannot be modified.  The read-only attribute "typecode" determines 
        whether the reference is to an object in an HDF5 file (OBJECT) 
        or a dataset region (DATASET_REGION).

        The object's truth value indicates whether it contains a nonzero
        reference.  This does not guarantee that is valid, but is useful
        for rejecting "background" elements in a dataset.
    """

    def __cinit__(self, *args, **kwds):
        self.typecode = -1

    def __nonzero__(self):
        # Whether or not the reference is zero-filled.  Note a True result
        # does *not* mean the reference is valid.
        cdef size_t obsize
        cdef int i
        cdef unsigned char *buf

        if self.typecode == H5R_OBJECT:
            obsize = sizeof(hobj_ref_t)
        elif self.typecode == H5R_DATASET_REGION:
            obsize = sizeof(hdset_reg_ref_t)
        else:
            raise TypeError("Unknown reference type")
        
        buf = <unsigned char*>&self.ref

        for i from 0<=i<obsize:
            if buf[i] != 0:
                return True
        return False

    def __repr__(self):
        if self.typecode == H5R_OBJECT:
            desc_str = PyString_FromStringAndSize(<char*>&self.ref, sizeof(hobj_ref_t))
            return "<HDF5 object reference (%r)>" % desc_str
        elif self.typecode == H5R_DATASET_REGION:
            desc_str = PyString_FromStringAndSize(<char*>&self.ref, sizeof(hdset_reg_ref_t))
            return "<HDF5 dataset region reference (%r)>" % desc_str
        return "<Invalid HDF5 reference>"

cdef class RegionReference(Reference):

    """
        Opaque representation of an HDF5 region reference.

        This is a subclass of Reference which exists mainly for programming
        convenience.
    """

    pass











