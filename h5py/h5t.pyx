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
from h5 cimport err_c, pause_errors, resume_errors
from numpy cimport dtype, ndarray

from utils cimport  emalloc, efree, pybool, \
                    create_ieee_complex64, create_ieee_complex128, \
                    require_tuple, convert_dims, convert_tuple

# Runtime imports
import h5
from h5 import DDict, ArgsError
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
    # If it's not one of these, the library SEGFAULTS. Thanks, guys.
    if classtype != H5T_COMPOUND and classtype != H5T_OPAQUE and \
        classtype != H5T_ENUM:
        raise ValueError("Class must be COMPOUND, OPAQUE or ENUM")
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

cdef class TypeID(ObjectID):

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

    def get_class(self):
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

            The returned datatype is always an unlocked copy of one of NATIVE_*.
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

    def get_order(self):
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
        H5Tset_sign(self.id, <H5T_sign_t>sign)

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


    cdef int enum_convert(self, long long *buf, int reverse) except -1:
        # Convert the long long value in "buf" to the native representation
        # of this (enumerated type).  Conversion performed in-place.
        # Reverse: false => llong->type; true => type->llong

        cdef hid_t basetype
        cdef H5T_class_t class_code

        class_code = H5Tget_class(self.id)
        if class_code != H5T_ENUM:
            raise ValueError("This type (class %d) is not of class ENUM" % class_code)

        basetype = H5Tget_super(self.id)
        assert basetype > 0

        try:
            if not reverse:
                H5Tconvert(H5T_NATIVE_LLONG, basetype, 1, buf, NULL, H5P_DEFAULT)
            else:
                H5Tconvert(basetype, H5T_NATIVE_LLONG, 1, buf, NULL, H5P_DEFAULT)
        finally:
            PY_H5Tclose(basetype)

    def enum_insert(self, char* name, long long value):
        """ (STRING name, INT/LONG value)

            Define a new member of an enumerated type.  The value will be
            automatically converted to the base type defined for this enum.  If
            the conversion results in overflow, the value will be silently 
            clipped.
        """
        cdef long long buf

        buf = value
        self.enum_convert(&buf, 0)
        H5Tenum_insert(self.id, name, &buf)

    def enum_nameof(self, long long value):
        """ (LLONG value) => STRING name

            Determine the name associated with the given value.  Due to a
            limitation of the HDF5 library, this can only retrieve names up to
            1023 characters in length.
        """
        cdef herr_t retval
        cdef char name[1024]
        cdef long long buf

        buf = value
        self.enum_convert(&buf, 0)
        retval = H5Tenum_nameof(self.id, &buf, name, 1024)
        if retval < 0:  # not sure whether H5Tenum_nameof will actually log an error
            raise RuntimeError("Failed to determine enum name of %d" % value)
        retstring = name
        return retstring

    def enum_valueof(self, char* name):
        """ (STRING name) => LONG value)

            Get the value associated with an enum name.
        """
        cdef long long buf

        H5Tenum_valueof(self.id, name, &buf)
        self.enum_convert(&buf, 1)
        return buf

    def get_member_value(self, int idx):
        """ (UINT index) => LONG value

            Determine the value for the member at the given zero-based index.
        """
        cdef herr_t retval
        cdef hid_t ptype
        cdef long long val
        ptype = 0

        if index < 0:
            raise ValueError("Index must be non-negative.")

        H5Tget_member_value(self.id, idx, &val)
        self.enum_convert(&val, 1)
        return val

# === Opaque datatypes ========================================================


    def set_tag(self, char* tag):
        """ (STRING tag)

            Set a string describing the contents of an opaque datatype.
        """
        H5Tset_tag(self.id, tag)

    def get_tag(self):
        """ () => STRING tag

            Get the tag associated with an opaque datatype.
        """
        cdef char* buf
        buf = NULL

        try:
            buf = H5Tget_tag(self.id)
            assert buf != NULL
            tag = buf
            return tag
        finally:
            free(buf)




