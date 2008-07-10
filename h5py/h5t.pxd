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

# This file is based on code from the PyTables project.  The complete PyTables
# license is available at licenses/pytables.txt, in the distribution root
# directory.

include "std_defs.pxi"
from h5 cimport class ObjectID

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


cdef object typewrap(hid_t id_)
 
# === HDF5 imports ============================================================

cdef extern from "hdf5.h":

  # --- Enumerated constants --------------------------------------------------

  # Byte orders
  ctypedef enum H5T_order_t:
    H5T_ORDER_ERROR      = -1,  # error
    H5T_ORDER_LE         = 0,   # little endian
    H5T_ORDER_BE         = 1,   # bit endian
    H5T_ORDER_VAX        = 2,   # VAX mixed endian
    H5T_ORDER_NONE       = 3    # no particular order (strings, bits,..)

  # HDF5 signed enums
  ctypedef enum H5T_sign_t:
    H5T_SGN_ERROR        = -1,  # error
    H5T_SGN_NONE         = 0,   # this is an unsigned type
    H5T_SGN_2            = 1,   # two's complement
    H5T_NSGN             = 2    # this must be last!

  ctypedef enum H5T_norm_t:
    H5T_NORM_ERROR       = -1,
    H5T_NORM_IMPLIED     = 0,
    H5T_NORM_MSBSET      = 1,
    H5T_NORM_NONE        = 2

  ctypedef enum H5T_cset_t:
    H5T_CSET_ERROR       = -1,
    H5T_CSET_ASCII       = 0

  ctypedef enum H5T_str_t:
    H5T_STR_ERROR        = -1,
    H5T_STR_NULLTERM     = 0,
    H5T_STR_NULLPAD      = 1,
    H5T_STR_SPACEPAD     = 2

  # Atomic datatype padding
  ctypedef enum H5T_pad_t:
    H5T_PAD_ZERO        = 0,
    H5T_PAD_ONE         = 1,
    H5T_PAD_BACKGROUND  = 2

  # HDF5 type classes
  cdef enum H5T_class_t:
    H5T_NO_CLASS         = -1,  # error
    H5T_INTEGER          = 0,   # integer types
    H5T_FLOAT            = 1,   # floating-point types
    H5T_TIME             = 2,   # date and time types
    H5T_STRING           = 3,   # character string types
    H5T_BITFIELD         = 4,   # bit field types
    H5T_OPAQUE           = 5,   # opaque types
    H5T_COMPOUND         = 6,   # compound types
    H5T_REFERENCE        = 7,   # reference types
    H5T_ENUM             = 8,   # enumeration types
    H5T_VLEN             = 9,   # variable-length types
    H5T_ARRAY            = 10,  # array types
    H5T_NCLASSES                # this must be last

  # Native search direction
  cdef enum H5T_direction_t:
    H5T_DIR_DEFAULT,
    H5T_DIR_ASCEND,
    H5T_DIR_DESCEND

  # --- Predefined datatypes --------------------------------------------------

  cdef enum:
    H5T_NATIVE_B8
    H5T_NATIVE_CHAR
    H5T_NATIVE_SCHAR
    H5T_NATIVE_UCHAR
    H5T_NATIVE_SHORT
    H5T_NATIVE_USHORT
    H5T_NATIVE_INT
    H5T_NATIVE_UINT
    H5T_NATIVE_LONG
    H5T_NATIVE_ULONG
    H5T_NATIVE_LLONG
    H5T_NATIVE_ULLONG
    H5T_NATIVE_FLOAT
    H5T_NATIVE_DOUBLE
    H5T_NATIVE_LDOUBLE

  # "Standard" types
  cdef enum:
    H5T_STD_I8LE
    H5T_STD_I16LE
    H5T_STD_I32LE
    H5T_STD_I64LE
    H5T_STD_U8LE
    H5T_STD_U16LE
    H5T_STD_U32LE
    H5T_STD_U64LE
    H5T_STD_B8LE
    H5T_STD_B16LE
    H5T_STD_B32LE
    H5T_STD_B64LE
    H5T_IEEE_F32LE
    H5T_IEEE_F64LE
    H5T_STD_I8BE
    H5T_STD_I16BE
    H5T_STD_I32BE
    H5T_STD_I64BE
    H5T_STD_U8BE
    H5T_STD_U16BE
    H5T_STD_U32BE
    H5T_STD_U64BE
    H5T_STD_B8BE
    H5T_STD_B16BE
    H5T_STD_B32BE
    H5T_STD_B64BE
    H5T_IEEE_F32BE
    H5T_IEEE_F64BE

  cdef enum:
    H5T_NATIVE_INT8
    H5T_NATIVE_UINT8
    H5T_NATIVE_INT16
    H5T_NATIVE_UINT16
    H5T_NATIVE_INT32
    H5T_NATIVE_UINT32
    H5T_NATIVE_INT64
    H5T_NATIVE_UINT64

  # Unix time types
  cdef enum:
    H5T_UNIX_D32LE
    H5T_UNIX_D64LE
    H5T_UNIX_D32BE
    H5T_UNIX_D64BE

  # String types
  cdef enum:
    H5T_FORTRAN_S1
    H5T_C_S1

  # References
  cdef enum:
    H5T_STD_REF_OBJ
    H5T_STD_REF_DSETREG

  # --- Datatype operations ---------------------------------------------------

  # General operations
  hid_t         H5Tcreate(H5T_class_t type, size_t size) except *
  hid_t         H5Topen(hid_t loc, char* name) except *
  herr_t        H5Tcommit(hid_t loc_id, char* name, hid_t type) except *
  htri_t        H5Tcommitted(hid_t type) except *
  hid_t         H5Tcopy(hid_t type_id) except *
  htri_t        H5Tequal(hid_t type_id1, hid_t type_id2  ) except *
  herr_t        H5Tlock(hid_t type_id) except *
  H5T_class_t   H5Tget_class(hid_t type_id) except *
  size_t        H5Tget_size(hid_t type_id) except? 0
  hid_t         H5Tget_super(hid_t type) except *
  htri_t        H5Tdetect_class(hid_t type_id, H5T_class_t dtype_class) except *
  herr_t        H5Tclose(hid_t type_id) except *

  hid_t         H5Tget_native_type(hid_t type_id, H5T_direction_t direction) except *

  # Not for public API
  herr_t        H5Tconvert(hid_t src_id, hid_t dst_id, size_t nelmts, void *buf, void *background, hid_t plist_id) except *

  # Atomic datatypes
  herr_t        H5Tset_size(hid_t type_id, size_t size) except *

  H5T_order_t   H5Tget_order(hid_t type_id) except *
  herr_t        H5Tset_order(hid_t type_id, H5T_order_t order) except *

  hsize_t       H5Tget_precision(hid_t type_id) except? 0
  herr_t        H5Tset_precision(hid_t type_id, size_t prec) except *

  int           H5Tget_offset(hid_t type_id) except *
  herr_t        H5Tset_offset(hid_t type_id, size_t offset) except *

  herr_t        H5Tget_pad(hid_t type_id, H5T_pad_t * lsb, H5T_pad_t * msb  ) except *
  herr_t        H5Tset_pad(hid_t type_id, H5T_pad_t lsb, H5T_pad_t msb  ) except *

  H5T_sign_t    H5Tget_sign(hid_t type_id) except *
  herr_t        H5Tset_sign(hid_t type_id, H5T_sign_t sign) except *

  herr_t        H5Tget_fields(hid_t type_id, size_t *spos, size_t *epos, 
                                size_t *esize, size_t *mpos, size_t *msize  ) except *
  herr_t        H5Tset_fields(hid_t type_id, size_t spos, size_t epos, 
                                size_t esize, size_t mpos, size_t msize  ) except *

  size_t        H5Tget_ebias(hid_t type_id) except? 0
  herr_t        H5Tset_ebias(hid_t type_id, size_t ebias) except *
  H5T_norm_t    H5Tget_norm(hid_t type_id) except *
  herr_t        H5Tset_norm(hid_t type_id, H5T_norm_t norm) except *
  H5T_pad_t     H5Tget_inpad(hid_t type_id) except *
  herr_t        H5Tset_inpad(hid_t type_id, H5T_pad_t inpad) except *
  H5T_cset_t    H5Tget_cset(hid_t type_id) except *
  herr_t        H5Tset_cset(hid_t type_id, H5T_cset_t cset) except *
  H5T_str_t     H5Tget_strpad(hid_t type_id) except *
  herr_t        H5Tset_strpad(hid_t type_id, H5T_str_t strpad) except *

  # VLENs
  hid_t     H5Tvlen_create(hid_t base_type_id) except *
  htri_t    H5Tis_variable_str(hid_t dtype_id) except *

  # Compound data types
  int           H5Tget_nmembers(hid_t type_id) except *
  H5T_class_t   H5Tget_member_class(hid_t type_id, int member_no) except *
  char*         H5Tget_member_name(hid_t type_id, unsigned membno) except? NULL
  hid_t         H5Tget_member_type(hid_t type_id, unsigned membno) except *
  int           H5Tget_member_offset(hid_t type_id, int membno) except *
  int           H5Tget_member_index(hid_t type_id, char* name) except *
  herr_t        H5Tinsert(hid_t parent_id, char *name, size_t offset,
                   hid_t member_id) except *
  herr_t        H5Tpack(hid_t type_id) except *

  # Enumerated types
  hid_t     H5Tenum_create(hid_t base_id) except *
  herr_t    H5Tenum_insert(hid_t type, char *name, void *value) except *
  herr_t    H5Tenum_nameof( hid_t type, void *value, char *name, size_t size  ) except *
  herr_t    H5Tenum_valueof( hid_t type, char *name, void *value  ) except *
  herr_t    H5Tget_member_value(hid_t type,  unsigned int memb_no, void *value  ) except *

  # Array data types
  hid_t H5Tarray_create(hid_t base_id, int ndims, hsize_t dims[], int perm[]) except *
  int   H5Tget_array_ndims(hid_t type_id) except *
  int   H5Tget_array_dims(hid_t type_id, hsize_t dims[], int perm[]) except *

  # Opaque data types
  herr_t    H5Tset_tag(hid_t type_id, char* tag) except *
  char*     H5Tget_tag(hid_t type_id) except? NULL









