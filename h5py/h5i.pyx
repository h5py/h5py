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

# Pyrex compile-time imports
from defs_c   cimport size_t, malloc, free
from h5  cimport hid_t

import h5
from h5 import DDict
from errors import H5TypeError

# === Public constants and data structures ====================================

TYPE_BADID = H5I_BADID
TYPE_FILE = H5I_FILE
TYPE_GROUP = H5I_GROUP
TYPE_DATASPACE = H5I_GROUP
TYPE_DATASET = H5I_DATASET
TYPE_ATTR = H5I_ATTR
TYPE_REFERENCE = H5I_REFERENCE
TYPE_MAPPER = { H5I_BADID: 'BAD ID', H5I_FILE: 'FILE', H5I_GROUP: 'GROUP',
                 H5I_DATASET: 'DATASET', H5I_ATTR: 'ATTRIBUTE', 
                 H5I_REFERENCE: 'REFERENCE' }
TYPE_MAPPER = DDict(TYPE_MAPPER)

# === Introspection API =======================================================

def get_type(hid_t obj_id):
    """ (INT obj_id) => INT type_code

        Determine the type of an arbitrary HDF5 object.  The return value is
        always one of TYPE_*; if the ID is invalid, TYPE_BADID is returned.
    """
    cdef int retval
    retval = <int>H5Iget_type(obj_id)
    return retval

def get_name(hid_t obj_id):
    """ (INT obj_id) => STRING name or None

        Determine (a) name of an HDF5 object.  Because an object has as many
        names as there are hard links to it, this may not be unique.  If the
        object does not have a name (transient datatypes, etc.), the 
        return value is None.
    """
    cdef size_t namelen
    cdef char* name

    namelen = H5Iget_name(obj_id, NULL, 0)
    if namelen < 0:
        raise H5TypeError("Failed to determine name of object %d" % obj_id)
    if namelen == 0:
        return None

    name = <char*>malloc(namelen+1)
    namelen = H5Iget_name(obj_id, name, namelen+1)
    retstring = name
    free(name)

    return retstring

    











