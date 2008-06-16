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

    1. Translation

        The functions py_translate_h5t and py_translate_dtype do the heavy
        lifting required to go between HDF5 datatype objects and Numpy dtypes.

        Since the HDF5 library can represent a greater range of types than
        Numpy, the conversion is asymmetric.  Attempting to convert an HDF5
        type to a Numpy dtype will result in a dtype object which matches
        as closely as possible.  In contrast, converting from a Numpy dtype
        to an HDF5 type will always result in a precise, byte-compatible 
        description of the Numpy data layout.

    2. Complex numbers

        Since HDF5 has no native complex types, and the native Numpy
        representation is a struct with two floating-point members, complex
        numbers are saved as HDF5 compound objects.

        These compound objects have exactly two fields, with IEEE 32- or 64-
        bit format, and default names "r" and "i".  Since other conventions
        exist for field naming, and in fact may be essential for compatibility
        with external tools, new names can be specified as arguments to
        both py_translate_* functions.

    3. Enumerations

        There is no native Numpy or Python type for enumerations.  Since an
        enumerated type is simply a mapping between string names and integer
        values, I have implemented enum support through dictionaries.  

        An HDF5 H5T_ENUM type is converted to the appropriate Numpy integer 
        type (e.g. <u4, etc.), and a dictionary mapping names to values is also 
        generated. This dictionary is attached to the dtype object via the
        functions py_enum_attach and py_enum_recover.

        The exact dtype declaration is given below; howeve, the py_enum*
        functions should encapsulate almost all meaningful operations.

        enum_dict = {'RED': 0L, 'GREEN': 1L}

        dtype( ('<i4', [ ( (enum_dict, 'enum'),   '<i4' )] ) )
                  ^             ^         ^         ^
             (main type)  (metadata) (field name) (field type)
"""

# Pyrex compile-time imports
from defs_c cimport free
from h5p cimport H5P_DEFAULT
from h5e cimport err_c, pause_errors, resume_errors
from numpy cimport dtype, ndarray

from utils cimport  emalloc, efree, pybool, \
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
    
cdef object lockid(hid_t id_in):
    cdef TypeID tid
    tid = TypeID(id_in)
    tid._locked = 1
    return tid

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
IEEE_F32LE = lockid(H5T_IEEE_F32LE)
IEEE_F32BE = lockid(H5T_IEEE_F32BE)
IEEE_F64LE = lockid(H5T_IEEE_F64LE)
IEEE_F64BE = lockid(H5T_IEEE_F64BE)

# Signed 2's complement integer types
STD_I8LE  = lockid(H5T_STD_I8LE)
STD_I16LE = lockid(H5T_STD_I16LE)
STD_I32LE = lockid(H5T_STD_I32LE)
STD_I64LE = lockid(H5T_STD_I64LE)

STD_I8BE  = lockid(H5T_STD_I8BE)
STD_I16BE = lockid(H5T_STD_I16BE)
STD_I32BE = lockid(H5T_STD_I32BE)
STD_I64BE = lockid(H5T_STD_I64BE)

# Unsigned integers
STD_U8LE  = lockid(H5T_STD_U8LE)
STD_U16LE = lockid(H5T_STD_U16LE)
STD_U32LE = lockid(H5T_STD_U32LE)
STD_U64LE = lockid(H5T_STD_U64LE)

STD_U8BE  = lockid(H5T_STD_U8BE)
STD_U16BE = lockid(H5T_STD_U16BE)
STD_U32BE = lockid(H5T_STD_U32BE)
STD_U64BE = lockid(H5T_STD_U64BE)

# Native integer types by bytesize
NATIVE_INT8 = lockid(H5T_NATIVE_INT8)
NATIVE_UINT8 = lockid(H5T_NATIVE_UINT8)
NATIVE_INT16 = lockid(H5T_NATIVE_INT16)
NATIVE_UINT16 = lockid(H5T_NATIVE_UINT16)
NATIVE_INT32 = lockid(H5T_NATIVE_INT32)
NATIVE_UINT32 = lockid(H5T_NATIVE_UINT32)
NATIVE_INT64 = lockid(H5T_NATIVE_INT64)
NATIVE_UINT64 = lockid(H5T_NATIVE_UINT64)

# Null terminated (C) string type
C_S1 = lockid(H5T_C_S1)

# === General datatype operations =============================================

def create(int classtype, size_t size):
    """ (INT class, INT size) => INT type_id
        
        Create a new HDF5 type object.  Legal values are 
        COMPOUND, OPAQUE, and ENUM.
    """
    return TypeID(H5Tcreate(<H5T_class_t>classtype, size))

def open(ObjectID group not None, char* name):
    """ (ObjectID group, STRING name) => TypeID

        Open a named datatype from a file.
    """
    return TypeID(H5Topen(group.id, name))

def array_create(TypeID base not None, object dims_tpl):
    """ (TypeID base, TUPLE dimensions)

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
        return H5Tarray_create(base.id, rank, dims, NULL)
    finally:
        efree(dims)

def enum_create(TypeID base not None):
    """ (TypeID base) => INT new_type_id

        Create a new enumerated type based on an (integer) parent type.
    """
    return H5Tenum_create(base.id)

# === XXXX ====

cdef class TypeID(LockableID):

    """
        Represents an HDF5 datatype identifier.
    """

    def commit(self, ObjectID group not None, char* name):
        """ (ObjectID group, STRING name)

            Commit this (transient) datatype to a named datatype in a file.
        """
        return H5Tcommit(group.id, name, self.id)

    def committed(self):
        """ () => BOOL is_comitted

            Determine if a given type object is named (T) or transient (F).
        """
        return pybool(H5Tcommitted(self.id))

    def copy(self):
        """ () => TypeID

            Create a copy of this type object.
        """
        return TypeID(H5Tcopy(self.id))

    def equal(self, TypeID typeid):
        """ (TypeID typeid) => BOOL

            Test whether two identifiers point to the same datatype object.  
            Note this does NOT perform any kind of logical comparison.
        """
        return pybool(H5Tequal(self.id, typeid.id))

    def lock(self):
        """ (self)

            Lock a datatype, which makes it immutable and indestructible.
            Once locked, it can't be unlocked.
        """
        H5Tlock(self.id)
        self._locked = 1

    def get_class():
        """ () => INT classcode

            Determine the datatype's class code.
        """
        return <int>H5Tget_class(self.id)

    def get_size(self):
        """ () => INT size

            Determine the total size of a datatype, in bytes.
        """
        return H5Tget_size(self.id)

    def get_super(self):
        """ () => TypeID

            Determine the parent type of an array or enumeration datatype.
        """
        return TypeID(H5Tget_super(self.id))

    def get_native_type(self, int direction=H5T_DIR_DEFAULT):
        """ (INT direction) => INT new_type_id

            Determine the native C equivalent for the given datatype.
            Legal values for "direction" are:
              DIR_DEFAULT*
              DIR_ASCEND
              DIR_DESCEND
            These determine which direction the list of native datatypes is
            searched; see the HDF5 docs for a definitive list.

            The returned datatype is always an unlocked copy one of NATIVE_*.
        """
        return TypeID(H5Tget_native_type(self.id, <H5T_direction_t>direction))

    def detect_class(self, int classtype):
        """ (INT class) => BOOL class_is_present

            Determine if a member of the given class exists in a compound
            datatype.  The search is recursive.
        """
        return pybool(H5Tdetect_class(self.id, <H5T_class_t>classtype))

    def close(self):
        """ Close this datatype.  If it's locked, nothing happens.
        """
        if not self._locked:
            H5Tclose(type_id)

# === Atomic datatype operations ==============================================


    def set_size(self, size_t size):
        """ (INT size)

            Set the total size of the datatype, in bytes.  Useful mostly for
            string types.
        """
        H5Tset_size(self.id, size)

    def get_order(self)
        """ () => INT order

            Obtain the byte order of the datatype; one of:
             ORDER_LE
             ORDER_BE
             ORDER_NATIVE
        """
        return <int>H5Tget_order(self.id)

    def set_order(self, int order):
        """ (INT type_id, INT order)

            Set the byte order of the datatype. "order" must be one of
             ORDER_LE
             ORDER_BE
             ORDER_NATIVE
        """
        H5Tset_order(self.id, <H5T_order_t>order)

    def get_sign(self):
        """ (INT type_id) => INT sign

            Obtain the "signedness" of the datatype; one of:
              SGN_NONE:  Unsigned
              SGN_2:     Signed 2's complement
        """
        return <int>H5Tget_sign(self.id)

    def set_sign(self, int sign):
        """ (INT sign)

            Set the "signedness" of the datatype; one of:
              SGN_NONE:  Unsigned
              SGN_2:     Signed 2's complement
        """
        H5Tset_sign(self.id <H5T_sign_t>sign)

    def is_variable_str(self):
        """ () => BOOL is_variable

            Determine if the given string datatype is a variable-length string.
            Please note that reading/writing data in this format is impossible;
            only fixed-length strings are currently supported.
        """
        return pybool(H5Tis_variable_str(self.id))

# === Compound datatype operations ============================================


    def get_nmembers(self):
        """ () => INT number_of_members

            Determine the number of members in a compound or enumerated type.
        """
        return H5Tget_nmembers(self.id)

    def get_member_class(self, int member):
        """ (INT member) => INT class

            Determine the datatype class of the member of a compound type,
            identified by its index (0 <= member < nmembers).
        """
        if member < 0:
            raise ValueError("Member index must be non-negative.")
        return H5Tget_member_class(self.id, member)

    
    def get_member_name(self, int member):
        """ (INT member) => STRING name
        
            Determine the name of a member of a compound or enumerated type,
            identified by its index (0 <= member < nmembers).
        """
        cdef char* name
        name = NULL

        if member < 0:
            raise ValueError("Member index must be non-negative.")

        try:
            name = H5Tget_member_name(self.id, member)
            assert name != NULL
            pyname = name
        finally:
            free(name)

        return pyname

    def get_member_index(self, char* name):
        """ (STRING name) => INT index

            Determine the index of a member of a compound or enumerated datatype
            identified by a string name.
        """
        return H5Tget_member_index(self.id, name)

    def get_member_offset(self, int member):
        """ (INT member_index) => INT offset

            Determine the offset, in bytes, of the beginning of the specified
            member of a compound datatype.
        """
        return H5Tget_member_offset(self.id, member)

    def get_member_type(self, int member):
        """ (INT member_index) => INT type_id

            Create a copy of a member of a compound datatype, identified by its
            index.
        """
        if member < 0:
            raise ValueError("Member index must be non-negative.")
        return TypeID(H5Tget_member_type(self.id, member))

    def insert(self, char* name, size_t offset, TypeID field not None):
        """ (STRING name, INT offset, TypeID field)

            Add a named member datatype to a compound datatype.  The parameter
            offset indicates the offset from the start of the compound datatype,
            in bytes.
        """
        H5Tinsert(self.id, name, offset, field.id)

    def pack(self):
        """ ()

            Recursively removes padding (introduced on account of e.g. compiler
            alignment rules) from a compound datatype.
        """
        H5Tpack(self.id)

# === Array datatype operations ===============================================

    def get_array_ndims(self):
        """ () => INT rank

            Get the rank of the given array datatype.
        """
        return H5Tget_array_ndims(self.id)

    def get_array_dims(self):
        """ (INT type_id) => TUPLE dimensions

            Get the dimensions of the given array datatype as a tuple of integers.
        """
        cdef hsize_t rank   
        cdef hsize_t* dims
        dims = NULL

        rank = H5Tget_array_dims(self.id, NULL, NULL)
        dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
        try:
            H5Tget_array_dims(self.id, dims, NULL)
            return convert_dims(dims, rank)
        finally:
            efree(dims)

# === Enumeration datatypes ===================================================


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


def py_translate_h5t(hid_t type_id, object byteorder=None, 
                        object compound_names=None, object complex_names=None):
    """ (INT type_id, STRING byteorder=None, TUPLE compound_names=None.
         TUPLE complex_names=None)
        => DTYPE

        Create a Numpy dtype object as similar as possible to the given HDF5
        datatype object.  The result is guaranteed to be logically compatible
        with the original object, with no loss of precision, but may not
        implement the same memory layout as the HDF5 type.

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
                field_types.append(py_translate_h5t(tmp_id, byteorder,
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
            edct = py_translate_enum(type_id)
            # Superclass must be an integer, so only provide byteorder.
            typeobj = py_enum_attach(edct, py_translate_h5t(super_tid, byteorder))
        finally:
            PY_H5Tclose(super_tid)

    elif classtype == H5T_ARRAY:
        super_tid = get_super(type_id)
        try:
            base_dtype = py_translate_h5t(super_tid, byteorder, compound_names, complex_names)
        finally:
            PY_H5Tclose(super_tid)
        shape = get_array_dims(type_id)
        typeobj = dtype( (base_dtype, shape) )
    else:
        raise ValueError('Unsupported datatype class "%s"' % PY_NAMES[classtype])

    if byteorder is not None:
        return typeobj.newbyteorder(byteorder)

    return typeobj

def py_translate_dtype(dtype dtype_in not None, object complex_names=None):
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
            type_out = py_translate_dict(py_enum_recover(dtype_in), basetype)

        # Otherwise it's just a compound type
        else:
            type_out = create(H5T_COMPOUND, length)
            for name in dtype_in.names:
                dt, offset = dtype_in.fields[name]
                tmp = py_translate_dtype(dt, complex_names)
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
            basetype = py_translate_dtype(dtype_in.subdtype[0], complex_names)
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


def py_translate_enum(hid_t type_id):
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

def py_translate_dict(object enumdict, hid_t basetype):
    """ (DICT enumdict, INT basetype) => INT new_type_id

        Create a new HDF5 enumeration from a Python dictionary in the 
        format [string name] => long value, and an HDF5 base type.
    """
    cdef hid_t type_id
    type_id = enum_create(basetype)
    for name, value in enumdict.iteritems():
        enum_insert(type_id, str(name), value)

    return type_id

def py_enum_attach(object enumdict, object base_dtype):
    """ (DICT enum, DTYPE base_dtype) => DTYPE new_dtype

        Convert a Python dictionary in the format [string] => integer to a 
        Numpy dtype with associated enum dictionary.  Returns a new
        dtype object; does not mutate the original.
    """
    return dtype( (base_dtype, [( (enumdict, 'enum'), base_dtype )] ) )

def py_enum_recover(dtype dtype_in):
    """ (DTYPE dtype_with_enum) => DICT enum

        Get the enum dictionary from a Numpy dtype object.
    """
    cdef object names
    names = dtype_in.names

    if names is not None:
        if len(names) == 1 and names[0] == 'enum':
            return dtype_in.fields['enum'][2]

    raise ValueError("Type %s is not an enumerated type" % repr(dtype_in))

def py_list_compound_names(hid_t type_in):
    """ (INT type_id) => LIST compound_names
   
        Obtain a Python list of member names for a compound or
        enumeration type.
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
        function h5t.py_translate_dtype.
    """
    cdef hid_t tid
    tid = 0

    retval = None
    try:
        tid = py_translate_dtype(dt, complex_names)
        retval = True
    except ValueError:
        retval = False

    if tid:
        PY_H5Tclose(tid)

    return retval

PY_SIGN = DDict({H5T_SGN_NONE: "UNSIGNED", H5T_SGN_2: "SIGNED"})

PY_CLASS = DDict({ H5T_NO_CLASS: "ERROR", H5T_INTEGER: "INTEGER", 
                    H5T_FLOAT: "FLOAT", H5T_TIME: "TIME", H5T_STRING: "STRING", 
                    H5T_BITFIELD: "BITFIELD", H5T_OPAQUE: "OPAQUE", 
                    H5T_COMPOUND: "COMPOUND", H5T_REFERENCE: "REFERENCE",
                    H5T_ENUM: "ENUM", H5T_VLEN: "VLEN", H5T_ARRAY: "ARRAY" })

PY_ORDER = DDict({ H5T_ORDER_LE: "LITTLE-ENDIAN", H5T_ORDER_BE: "BIG-ENDIAN",
                    H5T_ORDER_VAX: "VAX MIXED-ENDIAN", H5T_ORDER_NONE: "NONE" })











