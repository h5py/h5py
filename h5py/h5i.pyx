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

cpdef ObjectID wrap_identifier(hid_t ident):

    cdef H5I_type_t typecode
    cdef ObjectID obj

    typecode = H5Iget_type(ident)
    if typecode == H5I_FILE:
        import h5f
        obj = h5f.FileID.open(ident)
    elif typecode == H5I_DATASET:
        import h5d
        obj = h5d.DatasetID.open(ident)
    elif typecode == H5I_GROUP:
        import h5g
        obj = h5g.GroupID.open(ident)
    elif typecode == H5I_ATTR:
        import h5a
        obj = h5a.AttrID.open(ident)
    elif typecode == H5I_DATATYPE:
        import h5t
        obj = h5t.typewrap(ident)
    elif typecode == H5I_GENPROP_LST:
        import h5p
        obj = h5p.propwrap(ident)
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
    except Exception:
        return None

    if namelen == 0:    # 1.6.5 doesn't raise an exception
        return None

    assert namelen > 0
    name = <char*>malloc(sizeof(char)*(namelen+1))
    try:
        H5Iget_name(obj.id, name, namelen+1)
        pystring = name
        return pystring
    finally:
        free(name)


def get_file_id(ObjectID obj not None):
    """ (ObjectID obj) => FileID

        Obtain an identifier for the file in which this object resides.
    """
    import h5f
    cdef hid_t fid
    fid = H5Iget_file_id(obj.id)
    if H5Iget_ref(fid) > 1:
        H5Idec_ref(fid)
    return h5f.FileID.open(fid)


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
