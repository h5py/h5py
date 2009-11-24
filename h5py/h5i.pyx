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
    Identifier interface for object inspection.
"""

include "config.pxi"

# Pyrex compile-time imports
from h5 cimport init_hdf5, ObjectID
from h5f cimport FileID
from h5d cimport DatasetID
from h5g cimport GroupID
from h5a cimport AttrID
from h5t cimport typewrap
from h5p cimport propwrap

from utils cimport emalloc, efree

init_hdf5()

# Runtime imports
from h5 import H5Error

# === Public constants and data structures ====================================

BADID       = H5I_BADID
FILE        = H5I_FILE
GROUP       = H5I_GROUP
DATASPACE   = H5I_DATASPACE
DATASET     = H5I_DATASET
ATTR        = H5I_ATTR
REFERENCE   = H5I_REFERENCE
GENPROP_CLS = H5I_GENPROP_CLS
GENPROP_LST = H5I_GENPROP_LST
DATATYPE    = H5I_DATATYPE

cdef ObjectID wrap_identifier(hid_t ident):

    cdef H5I_type_t typecode
    cdef ObjectID obj

    typecode = H5Iget_type(ident)
    if typecode == H5I_FILE:
        obj = FileID(ident)
    elif typecode == H5I_DATASET:
        obj = DatasetID(ident)
    elif typecode == H5I_GROUP:
        obj = GroupID(ident)
    elif typecode == H5I_ATTR:
        obj = AttrID(ident)
    elif typecode == H5I_DATATYPE:
        obj = typewrap(ident)
    elif typecode == H5I_GENPROP_LST:
        obj = propwrap(ident)
    else:
        raise ValueError("Unrecognized type code %d" % typecode)

    return obj

# === Identifier API ==========================================================


def get_type(ObjectID obj not None):
    """ (ObjectID obj) => INT type_code

        Determine the HDF5 typecode of an arbitrary HDF5 object.  The return 
        value is always one of the type constants defined in this module; if 
        the ID is invalid, BADID is returned.
    """
    return <int>H5Iget_type(obj.id)


def get_name(ObjectID obj not None):
    """ (ObjectID obj) => STRING name, or None

        Determine (a) name of an HDF5 object.  Because an object has as many
        names as there are hard links to it, this may not be unique.

        If the identifier is invalid or is not associated with a name
        (in the case of transient datatypes, dataspaces, etc), returns None.

        For some reason, this does not work on dereferenced objects.
    """
    cdef int namelen
    cdef char* name

    try:
        namelen = <int>H5Iget_name(obj.id, NULL, 0)
    except H5Error:
        return None

    if namelen == 0:    # 1.6.5 doesn't raise an exception
        return None

    assert namelen > 0
    name = <char*>emalloc(sizeof(char)*(namelen+1))
    try:
        H5Iget_name(obj.id, name, namelen+1)
        pystring = name
        return pystring
    finally:
        efree(name)


def get_file_id(ObjectID obj not None):
    """ (ObjectID obj) => FileID

        Obtain an identifier for the file in which this object resides.
    """
    return FileID(H5Iget_file_id(obj.id))


def inc_ref(ObjectID obj not None):
    """ (ObjectID obj)

        Increment the reference count for the given object.

        This function is provided for debugging only.  Reference counting
        is automatically synchronized with Python, and you can easily break
        ObjectID instances by abusing this function.
    """
    H5Iinc_ref(obj.id)


def get_ref(ObjectID obj not None):
    """ (ObjectID obj) => INT

        Retrieve the reference count for the given object.
    """
    return H5Iget_ref(obj.id)


def dec_ref(ObjectID obj not None):
    """ (ObjectID obj)

        Decrement the reference count for the given object.

        This function is provided for debugging only.  Reference counting
        is automatically synchronized with Python, and you can easily break
        ObjectID instances by abusing this function.
    """
    H5Idec_ref(obj.id)




    











