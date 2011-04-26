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

from defs cimport *

from _objects cimport class ObjectID

cdef class TypeID(ObjectID):

    cdef object py_dtype(self)

# --- Top-level classes ---

cdef class TypeArrayID(TypeID):
    pass

cdef class TypeOpaqueID(TypeID):
    pass

cdef class TypeStringID(TypeID):
    # Both vlen and fixed-len strings
    pass

cdef class TypeVlenID(TypeID):
    # Non-string vlens
    pass

cdef class TypeTimeID(TypeID):
    pass

cdef class TypeBitfieldID(TypeID):
    pass

cdef class TypeReferenceID(TypeID):
    pass

# --- Numeric atomic types ---

cdef class TypeAtomicID(TypeID):
    pass

cdef class TypeIntegerID(TypeAtomicID):
    pass

cdef class TypeFloatID(TypeAtomicID):
    pass

# --- Enums & compound types ---

cdef class TypeCompositeID(TypeID):
    pass

cdef class TypeEnumID(TypeCompositeID):

    cdef int enum_convert(self, long long *buf, int reverse) except -1

cdef class TypeCompoundID(TypeCompositeID):
    pass

# === C API for other modules =================================================

cpdef TypeID typewrap(hid_t id_)
cpdef TypeID py_create(object dtype, bint logical=*)










