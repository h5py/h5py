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
from defs_c cimport free
from h5p cimport H5P_DEFAULT
from h5e cimport err_c, pause_errors, resume_errors
from numpy cimport dtype, ndarray

from utils cimport  emalloc, efree, \
                    create_ieee_complex64, create_ieee_complex128, \
                    require_tuple, convert_dims, convert_tuple

# Runtime imports
import h5
from h5 import DDict
from h5e import ArgsError
import sys

# === Custom C API ============================================================

cdef herr_t PY_H5Tclose(hid_t type_id) except *:
    # Unconditionally close a datatype, ignoring all errors.

    cdef err_c cookie
    cdef herr_t retval
    cookie = pause_errors()
    try:
        retval = H5Tclose(type_id)
    finally:
        resume_errors(cookie)

    return retval
    
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

DIR_DEFAULT = H5T_DIR_DEFAULT
DIR_ASCEND  = H5T_DIR_ASCEND
DIR_DESCEND = H5T_DIR_DESCEND

STR_NULLTERM = H5T_STR_NULLTERM
STR_NULLPAD  = H5T_STR_NULLPAD
STR_SPACEPAD = H5T_STR_SPACEPAD


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
C_S1 = H5T_C_S1

# === General datatype operations =============================================

def create(int classtype, size_t size):
    """ (INT class, INT size) => INT type_id
        
        Create a new HDF5 type object.  Legal values are 
        COMPOUND, OPAQUE, and ENUM.
    """
    return H5Tcreate(<H5T_class_t>classtype, size)

def open(hid_t group_id, char* name):
    """ (INT group_id, STRING name) => INT type_id

        Open a named datatype from a file.
    """
    return H5Topen(group_id, name)

def commit(hid_t loc_id, char* name, hid_t type_id):
    """ (INT group_id, STRING name, INT type_id)

        Commit a transient datatype to a named datatype in a file.
    """
    return H5Tcommit(loc_id, name, type_id)

def committed(hid_t type_id):
    """ (INT type_id) => BOOL is_comitted

        Determine if a given type object is named (T) or transient (F).
    """
    return H5Tcommitted(type_id)

def copy(hid_t type_id):
    """ (INT type_id) => INT new_type_id

        Copy an existing HDF type object.
    """
    return H5Tcopy(type_id)

def equal(hid_t typeid_1, hid_t typeid_2):
    """ (INT typeid_1, INT typeid_2) => BOOL types_are_equal

        Test whether two identifiers point to the same datatype object.  
        Note this does NOT perform any kind of logical comparison.
    """
    return bool(H5Tequal(typeid_1, typeid_2))

def lock(hid_t type_id):
    """ (INT type_id)

        Lock a datatype, which makes it immutable and indestructible.
        Once locked, it can't be unlocked.
    """
    H5Tlock(type_id)

def get_class(hid_t type_id):
    """ (INT type_id) => INT class

        Determine the datatype's class.
    """
    return <int>H5Tget_class(type_id)

def get_size(hid_t type_id):
    """ (INT type_id) => INT size

        Determine the total size of a datatype, in bytes.
    """
    return H5Tget_size(type_id)

def get_super(hid_t type_id):
    """ (INT type_id) => INT super_type_id

        Determine the parent type of an array or enumeration datatype.
    """
    return H5Tget_super(type_id)

def get_native_type(hid_t type_id, int direction):
    """ (INT type_id, INT direction) => INT new_type_id

        Determine the native C equivalent for the given datatype.
        Legal values for "direction" are:
          DIR_DEFAULT
          DIR_ASCEND
          DIR_DESCEND
        These determine which direction the list of native datatypes is
        searched; see the HDF5 docs for a definitive list.

        The returned datatype is always a copy one of NATIVE_*, and must
        eventually be closed.
    """
    return H5Tget_native_type(type_id, <H5T_direction_t>direction)

def detect_class(hid_t type_id, int classtype):
    """ (INT type_id, INT class) => BOOL class_is_present

        Determine if a member of the given class exists in a compound
        datatype.  The search is recursive.
    """
    return bool(H5Tdetect_class(type_id, <H5T_class_t>classtype))

def close(hid_t type_id, int force=1):
    """ (INT type_id, BOOL force=True)

        Close this datatype.  If "force" is True (default), ignore errors 
        commonly associated with attempting to close immutable types.
    """
    try:
        H5Tclose(type_id)
    except ArgsError, e:
        if not (force and e.errno == 1005):  # ArgsError, bad value
            raise

# === Atomic datatype operations ==============================================


def set_size(hid_t type_id, size_t size):
    """ (INT type_id, INT size)

        Set the total size of the datatype, in bytes.  Useful mostly for
        string types.
    """
    H5Tset_size(type_id, size)

def get_order(hid_t type_id):
    """ (INT type_id) => INT order

        Obtain the byte order of the datatype; one of:
         ORDER_LE
         ORDER_BE
         ORDER_NATIVE
    """
    return <int>H5Tget_order(type_id)

def set_order(hid_t type_id, int order):
    """ (INT type_id, INT order)

        Set the byte order of the datatype. "order" must be one of
         ORDER_LE
         ORDER_BE
         ORDER_NATIVE
    """
    H5Tset_order(type_id, <H5T_order_t>order)

def get_sign(hid_t type_id):
    """ (INT type_id) => INT sign

        Obtain the "signedness" of the datatype; one of:
          SGN_NONE:  Unsigned
          SGN_2:     Signed 2's complement
    """
    return <int>H5Tget_sign(type_id)

def set_sign(hid_t type_id, int sign):
    """ (INT type_id, INT sign)

        Set the "signedness" of the datatype; one of:
          SGN_NONE:  Unsigned
          SGN_2:     Signed 2's complement
    """
    H5Tset_sign(type_id, <H5T_sign_t>sign)

def is_variable_str(hid_t type_id):
    """ (INT type_id) => BOOL is_variable

        Determine if the given string datatype is a variable-length string.
        Please note that reading/writing data in this format is impossible;
        only fixed-length strings are currently supported.
    """
    return bool(H5Tis_variable_str(type_id))

# === Compound datatype operations ============================================


def get_nmembers(hid_t type_id):
    """ (INT type_id) => INT number_of_members

        Determine the number of members in a compound or enumerated type.
    """
    return H5Tget_nmembers(type_id)

def get_member_class(hid_t type_id, int member):
    """ (INT type_id, INT member) => INT class

        Determine the datatype class of the member of a compound type,
        identified by its index (0 <= member < nmembers).
    """
    if member < 0:
        raise ValueError("Member index must be non-negative.")
    return H5Tget_member_class(type_id, member)

    
def get_member_name(hid_t type_id, int member):
    """ (INT type_id, INT member) => STRING name
    
        Determine the name of a member of a compound or enumerated type,
        identified by its index (0 <= member < nmembers).
    """
    cdef char* name
    name = NULL

    if member < 0:
        raise ValueError("Member index must be non-negative.")

    try:
        name = H5Tget_member_name(type_id, member)
        assert name != NULL
        pyname = name
    finally:
        free(name)

    return pyname

def get_member_index(hid_t type_id, char* name):
    """ (INT type_id, STRING name) => INT index

        Determine the index of a member of a compound or enumerated datatype
        identified by a string name.
    """
    return H5Tget_member_index(type_id, name)

def get_member_offset(hid_t type_id, int member):
    """ (INT type_id, INT member_index) => INT offset

        Determine the offset, in bytes, of the beginning of the specified
        member of a compound datatype.
    """
    return H5Tget_member_offset(type_id, member)


def get_member_type(hid_t type_id, int member):
    """ (INT type_id, INT member_index) => INT type_id

        Create a copy of a member of a compound datatype, identified by its
        index.  You are responsible for closing it when finished.
    """
    if member < 0:
        raise ValueError("Member index must be non-negative.")
    return H5Tget_member_type(type_id, member)

def insert(hid_t type_id, char* name, size_t offset, hid_t field_id):
    """ (INT compound_type_id, STRING name, INT offset, INT member_type)

        Add a named member datatype to a compound datatype.  The parameter
        offset indicates the offset from the start of the compound datatype,
        in bytes.
    """
    H5Tinsert(type_id, name, offset, field_id)

def pack(hid_t type_id):
    """ (INT type_id)

        Recursively removes padding (introduced on account of e.g. compiler
        alignment rules) from a compound datatype.
    """
    H5Tpack(type_id)

# === Array datatype operations ===============================================

def array_create(hid_t base, object dims_tpl):
    """ (INT base_type_id, TUPLE dimensions)

        Create a new array datatype, of parent type <base_type_id> and
        dimensions given via a tuple of non-negative integers.  "Unlimited" 
        dimensions are not allowed.
    """
    cdef hsize_t rank
    cdef hsize_t *dims
    dims = NULL

    require_tuple(dims_tpl, 0, -1, "dims_tpl")
    rank = len(dims_tpl)
    dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)

    try:
        convert_tuple(dims_tpl, dims, rank)
        return H5Tarray_create(base, rank, dims, NULL)
    finally:
        efree(dims)

def get_array_ndims(hid_t type_id):
    """ (INT type_id) => INT rank

        Get the rank of the given array datatype.
    """
    return H5Tget_array_ndims(type_id)

def get_array_dims(hid_t type_id):
    """ (INT type_id) => TUPLE dimensions

        Get the dimensions of the given array datatype as a tuple of integers.
    """
    cdef hsize_t rank   
    cdef hsize_t* dims
    dims = NULL

    rank = H5Tget_array_dims(type_id, NULL, NULL)
    dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
    try:
        H5Tget_array_dims(type_id, dims, NULL)
        return convert_dims(dims, rank)
    finally:
        efree(dims)

# === Enumeration datatypes ===================================================

def enum_create(hid_t base_id):
    """ (INT base_type_id) => INT new_type_id

        Create a new enumerated type based on parent type <base_type_id>
    """
    return H5Tenum_create(base_id)

cdef int enum_convert(hid_t type_id, long long *buf, int reverse) except -1:
    # Convert the long long value in "buf" to the native representation
    # of type_id.  Conversion performed in-place.
    # Reverse: false => llong->type; true => type->llong

    cdef hid_t basetype
    cdef H5T_class_t class_code

    class_code = H5Tget_class(type_id)
    if class_code != H5T_ENUM:
        raise ValueError("Type %d is not of class ENUM" % type_id)

    basetype = H5Tget_super(type_id)
    assert basetype > 0

    try:
        if not reverse:
            H5Tconvert(H5T_NATIVE_LLONG, basetype, 1, buf, NULL, H5P_DEFAULT)
        else:
            H5Tconvert(basetype, H5T_NATIVE_LLONG, 1, buf, NULL, H5P_DEFAULT)
    finally:
        PY_H5Tclose(basetype)

def enum_insert(hid_t type_id, char* name, long long value):
    """ (INT type_id, STRING name, INT/LONG value)

        Define a new member of an enumerated type.  The value will be
        automatically converted to the base type defined for this enum.  If
        the conversion results in overflow, the value will be silently clipped.
    """
    cdef long long buf

    buf = value
    enum_convert(type_id, &buf, 0)
    H5Tenum_insert(type_id, name, &buf)

def enum_nameof(hid_t type_id, long long value):
    """ (INT type_id, LLONG value) => STRING name

        Determine the name associated with the given value.  Due to a
        limitation of the HDF5 library, this can only retrieve names up to
        1023 characters in length.
    """
    cdef herr_t retval
    cdef char name[1024]
    cdef long long buf

    buf = value
    enum_convert(type_id, &buf, 0)
    retval = H5Tenum_nameof(type_id, &buf, name, 1024)
    if retval < 0:  # not sure whether H5Tenum_nameof will actually log an error
        raise RuntimeError("Failed to determine enum name of %d" % value)
    retstring = name
    return retstring

def enum_valueof(hid_t type_id, char* name):
    """ (INT type_id, STRING name) => LONG value)

        Get the value associated with an enum name.
    """
    cdef long long buf

    H5Tenum_valueof(type_id, name, &buf)
    enum_convert(type_id, &buf, 1)
    return buf

def get_member_value(hid_t type_id, int idx):
    """ (INT type_id, UINT index) => LONG value

        Determine the value for the member at the given zero-based index.
    """
    cdef herr_t retval
    cdef hid_t ptype
    cdef long long val
    ptype = 0

    if index < 0:
        raise ValueError("Index must be non-negative.")

    H5Tget_member_value(type_id, idx, &val)
    enum_convert(type_id, &val, 1)
    return val

# === Opaque datatypes ========================================================


def set_tag(hid_t type_id, char* tag):
    """ (INT type_id, STRING tag)

        Set a string describing the contents of an opaque datatype.
    """
    H5Tset_tag(type_id, tag)

def get_tag(hid_t type_id):
    """ (INT type_id) => STRING tag

        Get the tag associated with an opaque datatype.
    """
    cdef char* buf
    buf = NULL

    try:
        buf = H5Tget_tag(type_id)
        assert buf != NULL
        tag = buf
        return tag
    finally:
        free(buf)

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
            raise ValueError("Variable-length strings are not supported.")
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
                PY_H5Tclose(tmp_id)


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
            PY_H5Tclose(super_tid)

    elif classtype == H5T_ARRAY:
        super_tid = get_super(type_id)
        try:
            base_dtype = py_h5t_to_dtype(super_tid, byteorder, compound_names, complex_names)
        finally:
            PY_H5Tclose(super_tid)
        shape = get_array_dims(type_id)
        typeobj = dtype( (base_dtype, shape) )
    else:
        raise ValueError('Unsupported datatype class "%s"' % PY_NAMES[classtype])

    if byteorder is not None:
        return typeobj.newbyteorder(byteorder)

    return typeobj

def py_dtype_to_h5t(dtype dtype_in not None, object complex_names=None):
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
                    PY_H5Tclose(tmp)

    # Integers and floats map directly to HDF5 atomic types
    elif kind == c'u' or kind  == c'i' or kind == c'f': 
        try:
            type_out =  _code_map[dtype_in.str]
        except KeyError:
            raise ValueError("Failed to find '%s' in atomic code map" % dtype_in.str)

    # Complex numbers are stored as HDF5 structs, with names defined at runtime
    elif kind == c'c':

        if length == 8:
            type_out = create_ieee_complex64(byteorder, _complex_names[0], _complex_names[1])
        elif length == 16:
            type_out = create_ieee_complex128(byteorder, _complex_names[0], _complex_names[1])
        else:
            raise ValueError("Unsupported length %d for complex dtype: %s" % (length, repr(dtype_in)))

        if type_out < 0:
            raise ValueError("No complex equivalent for dtype: %s" % repr(dtype_in))

    # Opaque/array types are differentiated by the presence of a subdtype
    elif kind == c'V':

        if dtype_in.subdtype:
            basetype = py_dtype_to_h5t(dtype_in.subdtype[0], complex_names)
            try:
                type_out = array_create(basetype, dtype_in.subdtype[1])
            finally:
                PY_H5Tclose(basetype)
        else:
            type_out = create(H5T_OPAQUE, length)
                
    # Strings are assumed to be stored C-style.
    elif kind == c'S':
        type_out = copy(H5T_C_S1)
        set_size(type_out, length)

    else:
        raise ValueError("No conversion path for dtype: %s" % repr(dtype_in))

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

def py_recover_enum(dtype dtype_in):
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
    except ValueError:
        pass

    if tid:
        PY_H5Tclose(tid)

    return can_convert

PY_SIGN = DDict({H5T_SGN_NONE: "UNSIGNED", H5T_SGN_2: "SIGNED"})

PY_CLASS = DDict({ H5T_NO_CLASS: "ERROR", H5T_INTEGER: "INTEGER", 
                    H5T_FLOAT: "FLOAT", H5T_TIME: "TIME", H5T_STRING: "STRING", 
                    H5T_BITFIELD: "BITFIELD", H5T_OPAQUE: "OPAQUE", 
                    H5T_COMPOUND: "COMPOUND", H5T_REFERENCE: "REFERENCE",
                    H5T_ENUM: "ENUM", H5T_VLEN: "VLEN", H5T_ARRAY: "ARRAY" })

PY_ORDER = DDict({ H5T_ORDER_LE: "LITTLE-ENDIAN", H5T_ORDER_BE: "BIG-ENDIAN",
                    H5T_ORDER_VAX: "VAX MIXED-ENDIAN", H5T_ORDER_NONE: "NONE" })











