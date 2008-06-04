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
    HDF5 "H5T" data-type API

    Provides access to the HDF5 data-type object interface.  Functions
    are provided to convert HDF5 datatype object back and forth from Numpy
    dtype objects.  Constants are also defined in this module for a variety
    of HDF5 native types and classes. Points of interest:

    (1) Enumerations
    There is no native Numpy or Python type for enumerations.  Since an
    enumerated type is simply a mapping between string names and integer
    values, I have implemented enum support through dictionaries.  An HDF5
    H5T_ENUM type is converted to the appropriate Numpy integer type (e.g.
    <u4, etc.), and a dictionary mapping names to values is also generated.
    Since dtype objects cannot be subclassed (why?) and have no provision
    for directly attached metadata, the dtype is given a single named field 
    ("enum") and the dictionary stored in the metadata for that field. An
    example dtype declaration for this is:

        enum_dict = {'RED': 0L, 'GREEN': 1L}

        dtype( ('<i4', [ ( (enum_dict, 'enum'),   '<i4' )] ) )
                  ^             ^         ^         ^
             (main type)  (metadata) (field name) (field type)

    The functions py_attach_enum and py_recover_enum simplify the attachment
    and recovery of enumeration dictionaries from integer dtype objects.

    (2) Complex numbers
    Since HDF5 has no native complex types defined, and the native Numpy
    representation is a struct with two floating-point members, complex
    numbers are saved as HDF5 compound objects with IEEE 32/64 floating point
    and field names (by default) "r" and "i".  Complex numbers can be auto-
    recovered from HDF5 objects provided they match this format and have
    compatible field names.  Since other people may have named their fields
    e.g. "img" and "real", these names can be changed.  The API functions
    py_dtype_to_h5t and py_h5t_to_dtype take arguments which specify these
    names.

"""


# Pyrex compile-time imports
from defs_c   cimport malloc, free
from h5  cimport herr_t, hid_t, size_t, hsize_t, htri_t
from h5e cimport H5Eset_auto
from python cimport PyTuple_New, PyTuple_SetItem, Py_INCREF
from h5p cimport H5P_DEFAULT
cimport numpy
from numpy cimport dtype
from utils cimport create_ieee_complex64, create_ieee_complex128,\
                        tuple_to_dims, dims_to_tuple

# Runtime imports
import h5
from h5 import DDict
from errors import DatatypeError, ConversionError
import sys

H5Eset_auto(NULL,NULL)

# === Public constants and data structures ====================================

# Enumeration H5T_class_t
NO_CLASS  = H5T_NO_CLASS
INTEGER   = H5T_INTEGER
FLOAT     = H5T_FLOAT
TIME      = H5T_TIME
STRING    = H5T_STRING
BITFIELD  = H5T_BITFIELD
OPAQUE    = H5T_OPAQUE
COMPOUND  = H5T_COMPOUND
REFERENCE = H5T_REFERENCE
ENUM      = H5T_ENUM
VLEN      = H5T_VLEN
ARRAY     = H5T_ARRAY

# Enumeration H5T_sign_t
SGN_NONE   = H5T_SGN_NONE
SGN_2      = H5T_SGN_2

# Enumeration H5T_order_t
ORDER_LE    = H5T_ORDER_LE
ORDER_BE    = H5T_ORDER_BE
ORDER_VAX   = H5T_ORDER_VAX
ORDER_NONE  = H5T_ORDER_NONE

if sys.byteorder == "little":    # Custom python addition
    ORDER_NATIVE = H5T_ORDER_LE
else:
    ORDER_NATIVE = H5T_ORDER_BE

# --- Built-in HDF5 datatypes -------------------------------------------------

# IEEE floating-point
IEEE_F32LE = H5T_IEEE_F32LE
IEEE_F32BE = H5T_IEEE_F32BE
IEEE_F64LE = H5T_IEEE_F64LE 
IEEE_F64BE = H5T_IEEE_F64BE

# Signed 2's complement integer types
STD_I8LE  = H5T_STD_I8LE
STD_I16LE = H5T_STD_I16LE
STD_I32LE = H5T_STD_I32LE
STD_I64LE = H5T_STD_I64LE

STD_I8BE  = H5T_STD_I8BE
STD_I16BE = H5T_STD_I16BE
STD_I32BE = H5T_STD_I32BE
STD_I64BE = H5T_STD_I64BE

# Unsigned integers
STD_U8LE  = H5T_STD_U8LE
STD_U16LE = H5T_STD_U16LE
STD_U32LE = H5T_STD_U32LE
STD_U64LE = H5T_STD_U64LE

STD_U8BE  = H5T_STD_U8BE
STD_U16BE = H5T_STD_U16BE
STD_U32BE = H5T_STD_U32BE
STD_U64BE = H5T_STD_U64BE

# Native integer types by bytesize
NATIVE_INT8 = H5T_NATIVE_INT8
NATIVE_UINT8 = H5T_NATIVE_UINT8
NATIVE_INT16 = H5T_NATIVE_INT16
NATIVE_UINT16 = H5T_NATIVE_UINT16
NATIVE_INT32 = H5T_NATIVE_INT32
NATIVE_UINT32 = H5T_NATIVE_UINT32
NATIVE_INT64 = H5T_NATIVE_INT64
NATIVE_UINT64 = H5T_NATIVE_UINT64

# Null terminated (C) string type
CSTRING = H5T_C_S1

# === General datatype operations =============================================

def create(int classtype, size_t size):
    """ (INT class, INT size) => INT type_id
        
        Create a new HDF5 type object.  Legal values are COMPOUND, 
        OPAQUE, ENUM
    """
    cdef hid_t retval
    retval = H5Tcreate(<H5T_class_t>classtype, size)
    if retval < 0:
        raise DatatypeError("Failed to create datatype of class %s, size %d" % (str(classtype), size))
    return retval

def open(hid_t group_id, char* name):
    """ (INT group_id, STRING name) => INT type_id

        Open a named datatype from a file.
    """
    cdef hid_t retval
    retval = H5Topen(group_id, name)
    if retval < 0:
        raise DatatypeError('Failed to open datatype "%s" on group %d' % (name, group_id))
    return retval

def commit(hid_t loc_id, char* name, hid_t type_id):
    """ (INT group_id, STRING name, INT type_id)

        Commit a transient datatype to a named datatype in a file.
    """
    cdef herr_t retval
    retval = H5Tcommit(loc_id, name, type_id)
    if retval < 0:
        raise DatatypeError("Failed to commit datatype %d under group %d with name '%s'" % (type_id, loc_id, name))

def committed(hid_t type_id):
    """ (INT type_id) => BOOL is_comitted

        Determine if a given type object is named (T) or transient (F).
    """
    cdef htri_t retval
    retval = H5Tcommitted(type_id)
    if retval < 0:
        raise DatatypeError("Failed to determine status of datatype %d" % type_id)
    return bool(retval)

def copy(hid_t type_id):
    """ (INT type_id) => INT new_type_id

        Copy an existing HDF type object.
    """
    
    cdef hid_t retval
    retval = H5Tcopy(type_id)
    if retval < 0:
        raise DatatypeError("Failed to copy datatype %d" % type_id)
    return retval

def equal(hid_t typeid_1, hid_t typeid_2):
    """ (INT typeid_1, INT typeid_2) => BOOL types_are_equal

        Test whether two identifiers point to the same datatype object.  Note
        this does NOT perform any kind of logical comparison.
    """
    cdef htri_t retval
    retval = H5Tequal(typeid_1, typeid_2)
    if retval < 0:
        raise DatatypeError("Failed to test datatypes for equality: %d and %d" % (typeid_1, typeid_2))
    return bool(retval)

def lock(hid_t type_id):
    """ (INT type_id)

        Lock a datatype, which makes it immutable and indestructible.  Once
        locked, it can't be unlocked.
    """
    cdef herr_t retval
    retval = H5Tlock(type_id)
    if retval < 0:
        raise DatatypeError("Failed to lock datatype %d" % type_id)

def get_class(hid_t type_id):
    """ (INT type_id) => INT class

        Get <type_id>'s class.
    """

    cdef int classtype
    classtype = <int>H5Tget_class(type_id)
    if classtype < 0:
        raise DatatypeError("Failed to determine class of datatype %d" % type_id)
    return classtype

def get_size(hid_t type_id):
    """ (INT type_id) => INT size

        Determine the total size of a datatype, in bytes.
    """
    cdef size_t retval
    retval = H5Tget_size(type_id)
    if retval == 0:
        raise DatatypeError("Can't determine size of datatype %d (got 0)" % type_id)
    return retval

def get_super(hid_t type_id):
    """ (INT type_id) => INT super_type_id

        Determine the parent type of an array or enumeration datatype.
    """

    cdef hid_t stype
    stype = H5Tget_super(type_id)
    if stype < 0:
        raise DatatypeError("Can't determine base datatype of %d" % type_id)
    return stype

def detect_class(hid_t type_id, int classtype):
    """ (INT type_id, INT class) => BOOL class_is_present

        Determine if a member of class <class> exists in <type_id>
    """

    cdef htri_t retval
    retval = H5Tdetect_class(type_id, <H5T_class_t>classtype)
    if retval < 0:
        raise DatatypeError("Couldn't inspect datatype %d for class %s" % (type_id, str(classtype)))
    return bool(retval)


def close(hid_t type_id, int force=0):
    """ (INT type_id, BOOL force=False)

        Close this datatype.  If "force" is True, ignore any errors.  Useful
        for exception handlers, when you're not sure if you've got an immutable
        datatype.
    """
    cdef herr_t retval
    retval = H5Tclose(type_id)
    if retval < 0 and force:
        raise DatatypeError("Failed to close datatype %d" % type_id)

# === Atomic datatype operations ==============================================
#     H5Tget_size, H5Tset_size, H5Tget_order, H5Tset_order, H5Tget_precision, \
#     H5Tset_precision, H5Tget_offset, H5Tset_offset, H5Tget_sign, H5Tset_sign

def set_size(hid_t type_id, size_t size):
    """ (INT type_id, INT size)

        Set the total size of the datatype, in bytes.  Useful mostly for
        string types.
    """
    cdef herr_t retval
    retval = H5Tset_size(type_id, size)
    if retval < 0:
        raise DatatypeError("Failed to set size of datatype %d to %s" % (type_id, size))
    return retval

def get_order(hid_t type_id):
    """ (INT type_id) => INT order

        Obtain the byte order of the datatype; one of h5t.ORDER_* or
        h5t.pyORDER_NATIVE
    """
    cdef int order
    order = <int>H5Tget_order(type_id)
    if order < 0:
        raise DatatypeError("Failed to determine order of datatype %d" % type_id)
    return order

def set_order(hid_t type_id, int order):
    """ (INT type_id, INT order)

        Set the byte order of the datatype. <order> must be one of 
        h5t.ORDER_* or h5t.pyORDER_NATIVE
    """
    cdef herr_t retval
    retval = H5Tset_order(type_id, <H5T_order_t>order)
    if retval < 0:
        raise DatatypeError("Failed to set order of datatype %d" % type_id)

def get_sign(hid_t type_id):
    """ (INT type_id) => INT sign

        Obtain the "signedness" of the datatype; one of h5t.SIGN_*
    """
    cdef int retval
    retval = <int>H5Tget_sign(type_id)
    if retval < 0:
        raise DatatypeError("Failed to get sign of datatype %d" % type_id)
    return retval

def set_sign(hid_t type_id, int sign):
    """ (INT type_id, INT sign)

        Set the "signedness" of the datatype; one of h5t.SIGN_*
    """
    cdef herr_t retval
    retval = H5Tset_sign(type_id, <H5T_sign_t>sign)
    if retval < 0:
        raise DatatypeError("Failed to set sign of datatype %d" % type_id)

def is_variable_str(hid_t type_id):
    """ (INT type_id) => BOOL is_variable

        Determine if the given string datatype is a variable-length string.
        Please note that reading/writing data in this format is impossible;
        only fixed-length strings are currently supported.
    """
    cdef htri_t retval
    retval = H5Tis_variable_str(type_id)
    if retval < 0:
        raise DatatypeError("Failed to inspect type %d" % type_id)
    return bool(retval)

# === Compound datatype operations ============================================
# get_nmembers
# get_member_class
# get_member_name
# get_member_index
# get_member_offset
# get_member_type
# insert
# pack

def get_nmembers(hid_t type_id):
    """ (INT type_id) => INT number_of_members

        Determine the number of members in a compound or enumerated type.
    """
    cdef int retval
    retval = H5Tget_nmembers(type_id)
    if retval < 0:
        raise DatatypeError("Failed to determine members of datatype %d" % type_id)
    return retval

def get_member_class(hid_t type_id, int member):
    """ (INT type_id, INT member_index) => INT class

        Determine the datatype class of the member of a compound type,
        identified by its index (must be 0 <= idx <= nmembers).
    """

    cdef int retval
    retval = H5Tget_member_class(type_id, member)
    if retval < 0:
        raise DatatypeError()
    
def get_member_name(hid_t type_id, int member):
    """ (INT type_id, INT member_index) => STRING name
    
        Determine the name of a member of a compound or enumerated type,
        identified by its index.
    """

    cdef char* name
    cdef object pyname
    name = NULL
    name = H5Tget_member_name(type_id, member)
    if name != NULL:
        pyname = name
        free(name)
        return pyname
    raise DatatypeError()

def get_member_index(hid_t type_id, char* name):
    """ (INT type_id, STRING name) => INT index

        Determine the index of a member of a compound or enumerated datatype
        identified by a string name.
    """
    cdef int retval
    retval = H5Tget_member_index(type_id, name)
    if retval < 0:
        raise DatatypeError("Failed to determine index of field '%' in datatype %d" % (name, type_id))
    return retval

def get_member_offset(hid_t type_id, int member):
    """ (INT type_id, INT member_index) => INT offset

        Determine the offset, in bytes, of the beginning of the specified
        member of a compound datatype.  Due to a limitation of the HDF5
        library, this function will never raise an exception.  It returns
        0 on failure; be careful as this is also a legal offset value.
    """
    cdef size_t offset
    offset = H5Tget_member_offset(type_id, member)
    return offset

def get_member_type(hid_t type_id, int member):
    """ (INT type_id, INT member_index) => INT type_id

        Create a copy of a member of a compound datatype, identified by its
        index.  You are responsible for closing it when finished.
    """
    cdef hid_t retval
    retval = H5Tget_member_type(type_id, member)
    if retval < 0:
        raise DataTypeError
    return retval


def insert(hid_t type_id, char* name, size_t offset, hid_t field_id):
    """ (INT compound_type_id, STRING name, INT offset, INT member_type)

        Add a member <member_type> named <name> to an existing compound 
        datatype.  <offset> is  the offset in bytes from the beginning of the
        compound type.
    """
    cdef herr_t retval
    retval = H5Tinsert(type_id, name, offset, field_id)
    if retval < 0:
        raise DatatypeError("Failed to insert field %s into compound type" % name)

def pack(hid_t type_id):
    """ (INT type_id)

        Recursively removes padding (introduced on account of e.g. compiler
        alignment rules) from a compound datatype.
    """

    cdef herr_t retval
    retval = H5Tpack(type_id)
    if retval < 0:
        raise DatatypeError("Failed to pack datatype %d" % type_id)
    return retval

# === Array datatype operations ===============================================
# array_create
# get_array_ndims
# get_array_dims



def array_create(hid_t base, object dims_tpl):
    """ (INT base_type_id, TUPLE dimensions)

        Create a new array datatype, of parent type <base_type_id> and
        dimensions given via a tuple of non-negative integers.  "Unlimited" 
        dimensions are not allowed.
    """
    cdef int rank
    cdef hsize_t *dims
    cdef hid_t type_id
    rank = len(dims_tpl)

    dims = tuple_to_dims(dims_tpl)
    if dims == NULL:
        raise ValueError("Invalid dimensions tuple: %s" % str(dims_tpl))
    type_id = H5Tarray_create(base, rank, dims, NULL)
    free(dims)

    if type_id < 0:
        raise DatatypeError("Failed to create datatype based on %d, dimensions %s" % (base, str(dims_tpl)))
    return type_id

def get_array_ndims(hid_t type_id):
    """ (INT type_id) => INT rank

        Get the rank of the given array datatype.
    """
    cdef int n
    n = H5Tget_array_ndims(type_id)
    if n < 0:
        raise DatatypeError("Failed to determine rank of array datatype %d" % type_id)
    return n

def get_array_dims(hid_t type_id):
    """ (INT type_id) => TUPLE dimensions

        Get the dimensions of the given array datatype as a tuple of integers.
    """
    cdef int rank   
    cdef hsize_t* dims
    cdef object dims_tpl
    dims = NULL

    rank = H5Tget_array_dims(type_id, NULL, NULL)
    if rank < 0:
        raise DatatypeError("Failed to determine dimensions of datatype %d" % type_id)

    dims = <hsize_t*>malloc(sizeof(hsize_t)*rank)
    rank = H5Tget_array_dims(type_id, dims, NULL)
    dims_tpl = dims_to_tuple(dims, rank)
    free(dims)
    if dims_tpl is None:
        raise DatatypeError("Failed to determine dimensions of datatype %d: tuple conversion error" % type_id)

    return dims_tpl

# === Enumeration datatypes ===================================================
#  hid_t     H5Tenum_create(hid_t base_id)
#  herr_t    H5Tenum_insert(hid_t type, char *name, void *value)
#  herr_t    H5Tenum_nameof( hid_t type, void *value, char *name, size_t size  )
#  herr_t    H5Tenum_valueof( hid_t type, char *name, void *value  )

#  char*     H5Tget_member_name(hid_t type_id, unsigned field_idx  )
#  int       H5Tget_member_index(hid_t type_id, char * field_name  )

def enum_create(hid_t base_id):
    """ (INT base_type_id) => INT new_type_id

        Create a new enumerated type based on parent type <base_type_id>
    """
    cdef hid_t retval
    retval = H5Tenum_create(base_id)
    if retval < 0:
        raise DatatypeError("Failed to create enum of class %d" % base_id)
    return retval

def enum_insert(hid_t type_id, char* name, long long value):
    """ (INT type_id, STRING name, INT/LONG value)

        Define a new member of an enumerated type.  <value> will be
        automatically converted to the base type defined for this enum.
    """
    cdef herr_t retval
    cdef hid_t ptype
    cdef long long *data_ptr
    ptype = 0

    data_ptr = <long long*>malloc(sizeof(long long))
    try:
        data_ptr[0] = value
        ptype = H5Tget_super(type_id)
        retval = H5Tconvert(H5T_NATIVE_LLONG, ptype, 1, data_ptr, NULL, H5P_DEFAULT)
        if retval < 0:
            raise DatatypeError("Can't preconvert integer for enum insert")
        retval = H5Tenum_insert(type_id, name, data_ptr)
        if retval < 0:
            raise DatatypeError("Failed to insert '%s' (value %d) into enum %d" % (name, value, type_id))
    finally:
        if ptype:
            H5Tclose(ptype)
        free(data_ptr)

#  herr_t    H5Tget_member_value(hid_t type  unsigned memb_no, void *value  )
def get_member_value(hid_t type_id, unsigned int idx):
    """ (INT type_id, UINT index) => LONG value

        Determine the value for the member at <index> of enumerated type
        <type_id>
    """
    cdef herr_t retval
    cdef hid_t ptype
    cdef long long *data_ptr
    data_ptr = <long long*>malloc(sizeof(long long))

    try:
        ptype = H5Tget_super(type_id)
        if ptype < 0:
            raise DatatypeError("Failed to get parent type of enum %d" % type_id)

        retval = H5Tget_member_value(type_id, idx, data_ptr)
        if retval < 0:
            raise DatatypeError("Failed to obtain value of element %d of enum %d" % (idx, type_id))

        retval = H5Tconvert(ptype, H5T_NATIVE_LLONG, 1, data_ptr, NULL, H5P_DEFAULT)
        if retval < 0:
            raise DatatypeError("Failed to postconvert integer for enum retrieval")
    finally:
        H5Tclose(ptype)
        interm = data_ptr[0]
        free(data_ptr)
    return interm

# === Opaque datatypes ========================================================

def set_tag(hid_t type_id, char* tag):
    """ (INT type_id, STRING tag)

        Set the a string describing the contents of an opaque datatype
    """
    cdef herr_t retval
    retval = H5Tset_tag(type_id, tag)
    if retval < 0:
        raise DatatypeError("Failed to set opaque data tag '%s' on type %d" % (tag, type_id))
    return retval

def get_tag(hid_t type_id):
    """ (INT type_id) => STRING tag

        Get the tag associated with an opaque datatype
    """
    cdef char* buf
    cdef object tag
    buf = NULL

    buf = H5Tget_tag(type_id)
    if buf == NULL:
        raise DatatypeError("Failed to get opaque data tag for type %d" % type_id)
    tag = buf
    free(buf)
    return tag


# === Custom Python additions =================================================


# Map array protocol strings to their HDF5 atomic equivalents
# Not sure why LE/BE versions of I8/U8 exist; I'll include them anyway.
_code_map = {"<i1": H5T_STD_I8LE, "<i2": H5T_STD_I16LE, "<i4": H5T_STD_I32LE, "<i8": H5T_STD_I64LE,
            ">i1": H5T_STD_I8BE, ">i2": H5T_STD_I16BE, ">i4": H5T_STD_I32BE, ">i8": H5T_STD_I64BE,
            "|i1": H5T_NATIVE_INT8, "|u1": H5T_NATIVE_UINT8, 
            "<u1": H5T_STD_U8LE, "<u2": H5T_STD_U16LE, "<u4": H5T_STD_U32LE, "<u8": H5T_STD_U64LE,
            ">u1": H5T_STD_U8BE, ">u2": H5T_STD_U16BE, ">u4": H5T_STD_U32BE, ">u8": H5T_STD_U64BE,
            "<f4": H5T_IEEE_F32LE, "<f8": H5T_IEEE_F64LE, ">f4": H5T_IEEE_F32BE, ">f8": H5T_IEEE_F64BE }

# Intermediate mapping which takes complex types to their components
_complex_map = { "<c8": H5T_IEEE_F32LE, "<c16": H5T_IEEE_F64LE, ">c8": H5T_IEEE_F32BE, ">c16": H5T_IEEE_F64BE }

_order_map = { H5T_ORDER_NONE: '|', H5T_ORDER_LE: '<', H5T_ORDER_BE: '>'}
_sign_map  = { H5T_SGN_NONE: 'u', H5T_SGN_2: 'i' }

DEFAULT_COMPLEX_NAMES = ('r','i')
VALID_BYTEORDERS = ('<','>','=')

# For an HDF5 compound object to be considered complex, at least the following 
# must be true:
# (1) Must have exactly two fields
# (2) Both must be IEEE floating-point, of the same precision and byteorder

def _validate_complex(item):
    """ Common validation function for complex names, which must be 2-tuples
        containing strings.
    """
    if not isinstance(item, tuple) or len(item) != 2 or \
      not isinstance(item[0], str) or not isinstance(item[1], str):
        raise ValueError("Complex names must be given a 2-tuples of strings: (real, img)")

def _validate_byteorder(item):
    """ Common validation function for byte orders, which must be <, > or =.
    """
    if not item in VALID_BYTEORDERS:
        raise ValueError("Byte order must be one of "+", ".join(VALID_BYTEORDERS))
    
def _validate_names(names):
    """ Common validation function for compound object field names, which must
        be tuples of strings.
    """
    if isinstance(names, tuple):
        bad = False
        for name in names:
            if not isinstance(name, str):
                bad = True
                break
        if not bad:
            return
    raise ValueError("Compound names must be given as a tuple of strings.")


def py_h5t_to_dtype(hid_t type_id, object byteorder=None, 
                        object compound_names=None, object complex_names=None):
    """ (INT type_id, STRING byteorder=None, TUPLE compound_names=None.
         TUPLE complex_names=None)
        => DTYPE

        Create a Numpy dtype object as similar as possible to the given HDF5
        datatype object.  The result is not guaranteed to be memory-compatible
        with the original datatype object.

        Optional arguments:

        byteorder:  
            None or one of <, >, =.  Coerce the byte order of the resulting
            dtype object.  "None" preserves the original HDF5 byte order.  
            This option IS applied to subtypes of arrays and compound types.

        compound_names:
            None or a tuple indicating which fields of a compound type to 
            preserve in the output.  Fields in the new dtype will be listed 
            in the order that they appear here.  Specifying a field which 
            doesn't appear in the HDF5 type raises ValueError.  This option
            IS NOT applied to subtypes of a compound type, but IS applied to
            subtypes of arrays.

        complex_names:
            Specifies when and how to interpret HDF5 compound datatypes as 
            Python complex numbers.  May be None or a tuple with strings 
            (real name, img name).  "None" indicates the default mapping of
            ("r", "i").  To turn this off, set to the empty tuple "()".
            This option IS applied to subtypes of arrays and compound types.
    """
    cdef int classtype
    cdef int sign
    cdef int size
    cdef int order
    cdef int nfields
    cdef int i
    cdef hid_t tmp_id

    typeobj = None

    # Argument validation and defaults

    if byteorder is not None: 
        _validate_byteorder(byteorder)

    if compound_names is not None:
        _validate_names(compound_names)

    if complex_names is None:
        complex_names = DEFAULT_COMPLEX_NAMES
    elif complex_names != (): 
        _validate_complex(complex_names)

    # End argument validation

    classtype = get_class(type_id)
    
    if classtype == H5T_INTEGER:
        size = get_size(type_id)
        sign = get_sign(type_id)
        order = get_order(type_id)
        typeobj = dtype(_order_map[order] + _sign_map[sign] + str(size))

    elif classtype == H5T_FLOAT:
        size = get_size(type_id)
        order = get_order(type_id)
        typeobj = dtype(_order_map[order] + "f" + str(size))

    elif classtype == H5T_STRING:
        if is_variable_str(type_id):
            raise ConversionError("Variable-length strings are not supported.")
        else:
            size = get_size(type_id)
        typeobj = dtype("|S" + str(size))

    elif classtype == H5T_OPAQUE:
        size = get_size(type_id)
        typeobj = dtype("|V" + str(size))

    elif classtype == H5T_COMPOUND:


        nfields = get_nmembers(type_id)
        field_names = []
        field_types = []

        # First step: read field names and their Numpy dtypes into 
        # two separate arrays.
        for i from 0 <= i < nfields:
            tmp_id = get_member_type(type_id, i)
            try:
                tmp_name = get_member_name(type_id, i)
                field_names.append(tmp_name)
                field_types.append(py_h5t_to_dtype(tmp_id, byteorder,
                                        None, complex_names))
            finally:
                H5Tclose(tmp_id)


        # 1. Only a particular (ordered) subset is requested
        if compound_names is not None:
            dt_list = []
            # Validate the requested fields
            for name in compound_names:
                try:
                    idx = field_names.index(name)
                except ValueError:
                    raise ValueError('Field "%s" not found. Valid fields are:\n%s' % (name, ", ".join(field_names)))
                dt_list.append( (name, field_types[idx]) )
            
            typeobj = dtype(dt_list)

        # 2. Check if it should be converted to a complex number
        elif len(field_names) == 2 and tuple(field_names) == complex_names and \
          field_types[0] == field_types[1] and field_types[0].kind == 'f':

            bstring = field_types[0].str
            blen = int(bstring[2:])
            nstring = bstring[0] + "c" + str(2*blen)

            typeobj = dtype(nstring)

        # 3. Read all fields of the compound type, in HDF5 order.
        else:
            typeobj = dtype(zip(field_names, field_types))

    elif classtype == H5T_ENUM:
        # Enumerated types are treated as their parent type, with an additional
        # enum field entry carrying a dictionary as metadata
        super_tid = H5Tget_super(type_id)
        try:
            edct = py_enum_to_dict(type_id)
            # Superclass must be an integer, so only provide byteorder.
            typeobj = py_attach_enum(edct, py_h5t_to_dtype(super_tid, byteorder))
        finally:
            H5Tclose(super_tid)

    elif classtype == H5T_ARRAY:
        super_tid = get_super(type_id)
        try:
            base_dtype = py_h5t_to_dtype(super_tid, byteorder, compound_names, complex_names)
        finally:
            H5Tclose(super_tid)
        shape = get_array_dims(type_id)
        typeobj = dtype( (base_dtype, shape) )
    else:
        raise ConversionError('Unsupported datatype class "%s"' % PY_NAMES[classtype])

    if byteorder is not None:
        return typeobj.newbyteorder(byteorder)

    return typeobj

def py_dtype_to_h5t(numpy.dtype dtype_in, object complex_names=None):
    """ ( DTYPE dtype_in, TUPLE complex_names=None) => INT type_id

        Given a Numpy dtype object, generate a byte-for-byte memory-compatible
        HDF5 transient datatype object.

        complex_names:
            Specifies when and how to interpret Python complex numbers as
            HDF5 compound datatypes.  May be None or a tuple with strings 
            (real name, img name).  "None" indicates the default mapping of
            ("r", "i"). This option is also applied to subtypes of arrays 
            and compound types.
    """
    cdef hid_t type_out
    cdef hid_t tmp
    cdef hid_t basetype
    cdef int retval

    cdef char* type_str
    cdef char kind
    cdef char byteorder
    cdef int length

    type_out = -1

    if complex_names is None:
        complex_names = DEFAULT_COMPLEX_NAMES
    else: 
        _validate_complex(complex_names)

    type_str = dtype_in.str
    kind = type_str[1]
    byteorder = type_str[0]

    length = int(dtype_in.str[2:])  # is there a better way to do this?

    names = dtype_in.names

    # Anything with field names is considered to be a compound type, except enums
    if names is not None:

        # Check for enumerated type first
        if (kind == c'u' or kind == c'i') and len(names) == 1 and names[0] == 'enum':
            basetype = _code_map[dtype_in.str]
            type_out = py_dict_to_enum(py_recover_enum(dtype_in), basetype)

        # Otherwise it's just a compound type
        else:
            type_out = create(H5T_COMPOUND, length)
            for name in dtype_in.names:
                dt, offset = dtype_in.fields[name]
                tmp = py_dtype_to_h5t(dt, complex_names)
                try:
                    insert(type_out, name, offset, tmp)
                finally:
                    H5Tclose(tmp)

    # Integers and floats map directly to HDF5 atomic types
    elif kind == c'u' or kind  == c'i' or kind == c'f': 
        try:
            type_out =  _code_map[dtype_in.str]
        except KeyError:
            raise ConversionError("Failed to find '%s' in atomic code map" % dtype_in.str)

    # Complex numbers are stored as HDF5 structs, with names defined at runtime
    elif kind == c'c':

        if length == 8:
            type_out = create_ieee_complex64(byteorder, _complex_names[0], _complex_names[1])
        elif length == 16:
            type_out = create_ieee_complex128(byteorder, _complex_names[0], _complex_names[1])
        else:
            raise ConversionError("Unsupported length %d for complex dtype: %s" % (length, repr(dtype_in)))

        if type_out < 0:
            raise ConversionError("Failed to create complex equivalent for dtype: %s" % repr(dtype_in))

    # Opaque/array types are differentiated by the presence of a subdtype
    elif kind == c'V':

        if dtype_in.subdtype:
            basetype = py_dtype_to_h5t(dtype_in.subdtype[0], complex_names)
            try:
                type_out = array_create(basetype, dtype_in.subdtype[1])
            finally:
                H5Tclose(basetype)
        else:
            type_out = create(H5T_OPAQUE, length)
                
    # Strings are assumed to be stored C-style.
    elif kind == c'S':
        type_out = copy(H5T_C_S1)
        set_size(type_out, length)

    else:
        raise ConversionError("No conversion path for dtype: %s" % repr(dtype_in))

    return type_out


def py_enum_to_dict(hid_t type_id):
    """ (INT type_id) => DICT enum
        
        Produce a dictionary in the format [STRING] => LONG from
        an HDF5 enumerated type.
    """
    cdef int nmem
    cdef int i
    nmem = get_nmembers(type_id)

    dictout = {}

    for i from 0 <= i < nmem:
        dictout[get_member_name(type_id, i)] = get_member_value(type_id,i)

    return dictout

def py_dict_to_enum(object enumdict, hid_t basetype):
    """ (DICT enum, INT base_type_id) => INT new_type_id

        Create a new HDF5 enumeration from a Python dictionary in the format
        [string name] => long value, and an HDF5 base type
    """
    cdef hid_t type_id
    type_id = enum_create(basetype)
    for name, value in enumdict.iteritems():
        enum_insert(type_id, str(name), value)

    return type_id

def py_attach_enum(object enumdict, object basetype):
    """ (DICT enum, DTYPE base_dtype) => DTYPE new_dtype

        Convert a Python dictionary in the format [string] => integer to a 
        Numpy dtype with associated enum dictionary.
    """
    return dtype( (basetype, [( (enumdict, 'enum'), basetype )] ) )

def py_recover_enum(numpy.dtype dtype_in):
    """ (DTYPE dtype_with_enum) => DICT enum

        Extract the enum dictionary from a Numpy dtype object
    """
    cdef object names
    names = dtype_in.names

    if names is not None:
        if len(names) == 1 and names[0] == 'enum':
            return dtype_in.fields['enum'][2]

    raise ValueError("Type %s is not an enumerated type" % repr(dtype_in))

def py_list_compound_names(hid_t type_in):
    """ (INT type_id) => LIST compound_names
   
        Obtain a Python list of member names for a compound or enumeration
        type.
    """
    cdef int nmem
    cdef int i

    nmem = get_nmembers(type_in)

    qlist = []
    for i from 0<=i<nmem:
        qlist.append(get_member_name(type_in,i))

    return qlist

def py_can_convert_dtype(object dt, object complex_names=None):
    """ (DTYPE dt, TUPLE complex_names=None) => BOOL can_convert

        Test whether the given Numpy dtype can be converted to the appropriate
        memory-compatible HDF5 datatype.  complex_names works as in the
        function h5t.py_dtype_to_h5t.
    """
    cdef hid_t tid
    tid = 0
    can_convert = False
    try:
        tid = py_dtype_to_h5t(dt, complex_names)
        can_convert = True
    except ConversionError:
        pass

    if tid:
        H5Tclose(tid)

    return can_convert

PY_SIGN = DDict({H5T_SGN_NONE: "UNSIGNED", H5T_SGN_2: "SIGNED"})

PY_CLASS = DDict({ H5T_NO_CLASS: "ERROR", H5T_INTEGER: "INTEGER", 
                    H5T_FLOAT: "FLOAT", H5T_TIME: "TIME", H5T_STRING: "STRING", 
                    H5T_BITFIELD: "BITFIELD", H5T_OPAQUE: "OPAQUE", 
                    H5T_COMPOUND: "COMPOUND", H5T_REFERENCE: "REFERENCE",
                    H5T_ENUM: "ENUM", H5T_VLEN: "VLEN", H5T_ARRAY: "ARRAY" })

PY_ORDER = DDict({ H5T_ORDER_LE: "LITTLE-ENDIAN", H5T_ORDER_BE: "BIG-ENDIAN",
                    H5T_ORDER_VAX: "VAX MIXED-ENDIAN", H5T_ORDER_NONE: "NONE" })











