# cython: profile=False

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

# TODO:
# Where a Py object type is the destination, move the logic which selects
# the kind (bytes or unicode) out of the 
"""
    Low-level type-conversion routines.
"""

from h5r cimport Reference, RegionReference, hobj_ref_t, hdset_reg_ref_t

import sys
import h5py

PY3 = sys.version_info[0] == 3

cdef extern from *:

    ctypedef int npy_ucs4
    
# Minimal interface for Python objects immune to Cython refcounting
cdef extern from "Python.h":
    
    # From Cython declarations
    ctypedef void PyTypeObject
    ctypedef struct PyObject:
        Py_ssize_t ob_refcnt
        PyTypeObject *ob_type

    ctypedef int Py_UNICODE
    ctypedef struct PyUnicodeObject:
        Py_UNICODE *str

    PyObject* PyBytes_FromString(char* str) except NULL
    int PyBytes_CheckExact(PyObject* str) except *
    int PyBytes_Size(PyObject* obj) except *
    PyObject* PyString_AsDecodedObject(PyObject* s, char *encoding, char *errors) except NULL

    PyObject* PyUnicode_DecodeUTF8(char *s, Py_ssize_t size, char *errors) except NULL
    int PyUnicode_CheckExact(PyObject* str) except *
    PyObject* PyUnicode_AsUTF8String(PyObject* s) except NULL

    PyObject* PyObject_Str(PyObject* obj) except NULL
    PyObject* PyObject_Unicode(PyObject* obj) except NULL
    char* PyBytes_AsString(PyObject* obj) except NULL

    PyObject* Py_None
    void Py_INCREF(PyObject* obj)
    void Py_DECREF(PyObject* obj)
    void Py_XDECREF(PyObject* obj)

cdef object objectify(PyObject* o):
    Py_INCREF(o)
    return <object>o

cdef hid_t H5PY_OBJ = 0
cdef hid_t H5PY_UTF32_LE = 0
cdef hid_t H5PY_UTF32_BE = 0
cdef hid_t H5PY_UNICODE = 0
cdef hid_t H5PY_BYTES = 0
cdef int H5PY_TYPES_INITED = 0

cpdef init_python_types():
    global H5PY_TYPES_INITED, H5PY_OBJ, H5PY_UTF32_LE, H5PY_UTF32_BE, H5PY_UNICODE, H5PY_BYTES
    
    if H5PY_TYPES_INITED:
        return 

    H5PY_OBJ = H5Tcreate(H5T_OPAQUE, sizeof(PyObject*))
    H5Tset_tag(H5PY_OBJ, "PYTHON:OBJECT")
    H5Tlock(H5PY_OBJ)

    H5PY_UTF32_LE = H5Tcreate(H5T_OPAQUE, 4)
    H5Tset_tag(H5PY_UTF32_LE, "PYTHON:UTF32LE")
    H5Tlock(H5PY_UTF32_LE)

    H5PY_UTF32_BE = H5Tcreate(H5T_OPAQUE, 4)
    H5Tset_tag(H5PY_UTF32_BE, "PYTHON:UTF32BE")
    H5Tlock(H5PY_UTF32_BE)

    H5PY_UNICODE = H5Tcreate(H5T_OPAQUE, sizeof(PyObject*))
    H5Tset_tag(H5PY_UNICODE, "PYTHON:UNICODE")
    H5Tlock(H5PY_UNICODE)

    H5PY_BYTES = H5Tcreate(H5T_OPAQUE, sizeof(PyObject*))
    H5Tset_tag(H5PY_BYTES, "PYTHON:BYTES")
    H5Tlock(H5PY_BYTES)

    H5PY_TYPES_INITED = 1

cpdef hid_t get_python_obj():
    global H5PY_OBJ
    return H5PY_OBJ

cpdef hid_t get_utf32_le():
    global H5PY_UTF32_LE
    return H5PY_UTF32_LE

cpdef hid_t get_utf32_be():
    global H5PY_UTF32_BE
    return H5PY_UTF32_BE

cpdef hid_t get_unicode():
    global H5PY_UNICODE
    return H5PY_UNICODE

cpdef hid_t get_bytes():
    global H5PY_BYTES
    return H5PY_BYTES

ctypedef int (*conv_operator_t)(void* ipt, void* opt, void* bkg, void* priv) except -1
ctypedef herr_t (*init_operator_t)(hid_t src, hid_t dst, void** priv) except -1

# Generic conversion callback
#
# The actual conversion routines are one-liners which plug the appropriate
# operator callback into this function.  This prevents us from having to
# repeat all the conversion boilerplate for every single callback.
#
# While this is somewhat slower than a custom function, the added overhead is
# likely small compared to the cost of the Python-side API calls required to
# implement the conversions.
cdef herr_t generic_converter(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl, conv_operator_t op,
                    init_operator_t initop, H5T_bkg_t need_bkg) except -1:

    cdef int command = cdata[0].command
    cdef conv_size_t *sizes
    cdef int i

    cdef char* buf = <char*>buf_i
    cdef char* bkg = <char*>bkg_i

    if command == H5T_CONV_INIT:

        cdata[0].need_bkg = need_bkg
        return initop(src_id, dst_id, &(cdata[0].priv))

    elif command == H5T_CONV_FREE:
        
        free(cdata[0].priv)
        cdata[0].priv = NULL

    elif command == H5T_CONV_CONV:

        sizes = <conv_size_t*>cdata[0].priv

        if bkg_stride==0: bkg_stride = sizes[0].dst_size;

        if buf_stride == 0:
            # No explicit stride seems to mean that the elements are packed
            # contiguously in the buffer.  In this case we must be careful
            # not to "stomp on" input elements if the output elements are
            # of a larger size.

            if sizes[0].src_size >= sizes[0].dst_size:
                for i from 0<=i<nl:
                    op( buf + (i*sizes[0].src_size),    # input pointer
                        buf + (i*sizes[0].dst_size),    # output pointer
                        bkg + (i*bkg_stride),           # backing buffer
                        cdata[0].priv)                  # conversion context
            else:
                for i from nl>i>=0:
                    op( buf + (i*sizes[0].src_size),
                        buf + (i*sizes[0].dst_size),
                        bkg + (i*bkg_stride),
                        cdata[0].priv)
        else:
            # With explicit strides, we assume that the library knows the
            # alignment better than us.  Therefore we use the given stride
            # offsets exclusively.
            for i from 0<=i<nl:
                op( buf + (i*buf_stride),
                    buf + (i*buf_stride),   # note this is the same!
                    bkg + (i*bkg_stride),
                    cdata[0].priv)
    else:
        return -2   # Unrecognized command.  Note this is NOT an exception.

    return 0

# =============================================================================
# Generic conversion 

ctypedef struct conv_size_t:
    size_t src_size
    size_t dst_size
    int src_cset
    int dst_cset
    int src_pad
    int dst_pad
    int src_byte_order
    int dst_byte_order

cdef herr_t init_generic(hid_t src, hid_t dst, void** priv) except -1:
    
    cdef conv_size_t *sizes
    sizes = <conv_size_t*>malloc(sizeof(conv_size_t))
    priv[0] = sizes
    sizes[0].src_size = H5Tget_size(src)
    sizes[0].dst_size = H5Tget_size(dst)
    sizes[0].src_cset = -1
    sizes[0].dst_cset = -1
    sizes[0].src_pad = -1
    sizes[0].dst_pad = -1
    sizes[0].src_byte_order = 0
    sizes[0].dst_byte_order = 0

    if H5Tget_class(src) == H5T_STRING:
        sizes[0].src_cset = H5Tget_cset(src)
        sizes[0].src_pad = H5Tget_strpad(src)
    if H5Tget_class(dst) == H5T_STRING:
        sizes[0].dst_cset = H5Tget_cset(dst)
        sizes[0].dst_pad = H5Tget_strpad(dst)

    if H5Tequal(src, H5PY_UTF32_LE):
        sizes[0].src_byte_order = -1
    elif H5Tequal(src, H5PY_UTF32_BE):
        sizes[0].src_byte_order = 1

    if H5Tequal(dst, H5PY_UTF32_LE):
        sizes[0].dst_byte_order = -1
    elif H5Tequal(dst, H5PY_UTF32_BE):
        sizes[0].dst_byte_order = 1

    return 0

# These two init functions are necessary as HDF5 does not distinguish between
# fixed and variable length strings when the functions are registered.




# =============================================================================
# Section 1: Conversion from HDF5 vlen to all string types

# HDF5 vlen -> HDF5 vlen
# TODO: Omitted as HDF5 now handles this internally.  We'll eventually want
# this because they claim conversion will eventually be charset sensitive

# HDF5 vlen -> Fixed-width
cdef int conv_vlen2fixed(char** ipt, char* opt, void* bkg, conv_size_t* sizes) except -1:

    cdef char* vlen_str = ipt[0]
    cdef char pad = c'\0'
    cdef size_t ilen

    if sizes.dst_pad == H5T_STR_SPACEPAD:
        pad = c' '

    # Input is empty string
    if vlen_str == NULL:
        memset(opt, pad, sizes.dst_size)

    else:

        ilen = strlen(vlen_str)
            
        if ilen <= sizes[0].dst_size:
            # Input string is smaller, so copy it and pad the remainder
            memcpy(opt, vlen_str, ilen)
            memcpy(opt+ilen, &pad, sizes[0].dst_size - ilen)
        else:
            # Input string is too big, so simply truncate it
            # TODO: make sure utf-8 strings don't get chopped off in the middle of a
            # multibyte sequence
            memcpy(opt, vlen_str, sizes[0].dst_size)

        # TODO: Do we have to overwrite the last element with c'\0' if 
        # the padding is H5T_STR_NULLTERM?

    # TODO: Do we really own this reference?
    free(vlen_str)

    return 0

cdef herr_t init_vlen2fixed(hid_t src, hid_t dst, void** priv) except -1:

    if H5Tis_variable_str(src) and not H5Tis_variable_str(dst):
        return init_generic(src, dst, priv)
    return -2

cdef herr_t vlen2fixed(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl,  <conv_operator_t>conv_vlen2fixed, init_vlen2fixed, H5T_BKG_NO)

# HDF5 vlen -> Py bytes
cdef int conv_vlen2bytes(char** ipt, PyObject** opt, PyObject** bkg, conv_size_t* sizes) except -1:

    # We need this because ipt and opt may be the same pointer!
    cdef PyObject* temp
    
    if ipt[0] == NULL:
        temp = PyBytes_FromString("")
    else:
        temp = PyBytes_FromString(ipt[0])

    free(ipt[0])
    Py_XDECREF(bkg[0])
    opt[0] = temp

cdef herr_t init_vlen2bytes(hid_t src, hid_t dst, void**  priv) except -1:

    cdef int match = 0

    if H5Tis_variable_str(src) and H5Tequal(dst, H5PY_BYTES):
        match = 1

    # Generic object destination means we must consult the rules
    elif H5Tis_variable_str(src) and H5Tequal(dst, H5PY_OBJ):

        # Py2: ASCII vlens always map to byte strings
        if (not PY3) and H5Tget_cset(src) == H5T_CSET_ASCII:
            match = 1

        # Py3: Vlens of any kind map to byte strings only if in bytes mode
        if PY3 and h5py.get_config().read_byte_strings:
            match = 1

    if match:
        print "vlen2bytes matched"
        return init_generic(src, dst, priv)
    return -2

cdef herr_t vlen2bytes(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl,  <conv_operator_t>conv_vlen2bytes, init_vlen2bytes, H5T_BKG_YES)

# HDF5 vlen -> Py unicode
cdef int conv_vlen2unicode(char** ipt, PyObject** opt, PyObject** bkg, conv_size_t* sizes) except -1:

    cdef PyObject* temp
    
    if ipt[0] == NULL:
        temp = PyUnicode_DecodeUTF8("", 0, NULL)
    else:
        try:
            temp = PyUnicode_DecodeUTF8(ipt[0], strlen(ipt[0]), NULL)
        except UnicodeDecodeError:
            temp = PyBytes_FromString(ipt[0])

    free(ipt[0])
    Py_XDECREF(bkg[0])
    opt[0] = temp

cdef herr_t init_vlen2unicode(hid_t src, hid_t dst, void** priv) except -1:

    cdef int match = 0

    if H5Tis_variable_str(src) and H5Tequal(dst, H5PY_UNICODE):
        match = 1

    # Generic object destination means we must consult the rules
    elif H5Tis_variable_str(src) and H5Tequal(dst, H5PY_OBJ):

        # Py2 & Py3: UTF-8 vlens map to byte strings unless in byte string mode
        if (H5Tget_cset(src) == <int>H5T_CSET_UTF8) and (not h5py.get_config().read_byte_strings):
            match = 1

    if match:
        print "vlen2unicode matched"
        return init_generic(src, dst, priv)
    return -2

cdef herr_t vlen2unicode(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl,  <conv_operator_t>conv_vlen2unicode, init_vlen2unicode, H5T_BKG_YES)

# Not implemented for now as this can never happen with the current type translation
# HDF5 vlen -> NumPy UTF32

# =============================================================================
# Section 2: Conversion from HDF5 fixed-length to all destination strings

# HDF5 fixed -> HDF5 fixed
# TODO: Omitted for same reason as above

# HDF5 fixed -> vlen
cdef int conv_fixed2vlen(char* ipt, char** opt, void* bkg, conv_size_t *sizes) except -1:

    cdef char* vlen = NULL

    # TODO: charset validation?

    vlen = <char*>malloc(sizes[0].src_size+1)
    memcpy(vlen, ipt, sizes[0].src_size)
    vlen[sizes[0].src_size] = c'\0'

    opt[0] = vlen

    return 0

cdef herr_t init_fixed2vlen(hid_t src, hid_t dst, void** priv) except -1:

    if H5Tis_variable_str(dst) and not H5Tis_variable_str(dst):
        return init_generic(src, dst, priv)
    return -2

cdef herr_t fixed2vlen(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl,  <conv_operator_t>conv_fixed2vlen, init_fixed2vlen, H5T_BKG_NO)

# Omitted for now as they can't happen with the current type translation
# HDF5 fixed -> Py bytes
# HDF5 fixed -> Py unicode

# HDF5 fixed -> utf32
cdef int conv_fixed2utf32(char* ipt, npy_ucs4* opt, void* bkg, conv_size_t *sizes) except -1:

    raise NotImplementedError("fixed -> utf32le")

cdef herr_t init_fixed2utf32(hid_t src, hid_t dst, void** priv) except -1:

    if (not H5Tis_variable_str(dst)) and (H5Tequal(dst, H5PY_UTF32_LE) or H5Tequal(dst, H5PY_UTF32_BE)):
        return init_generic(src, dst, priv)
    return -2

cdef herr_t fixed2utf32(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl,  <conv_operator_t>conv_fixed2utf32, init_fixed2utf32, H5T_BKG_NO)

# =============================================================================
# Section 3: Conversion from Py bytes to all destination types

# Py bytes -> HDF5 fixed
cdef int conv_bytes2fixed(PyObject** ipt, char* opt, void* bkg, conv_size_t* sizes) except -1:

    cdef PyObject* pystring = NULL
    cdef char* cstring = NULL
    cdef size_t cstring_len = 0
    cdef char pad = c'\0'

    # TODO: Charset validation when dest has H5T_CSET_UTF8?

    if sizes[0].dst_pad == H5T_STR_SPACEPAD:
        pad = c' '

    if ipt[0] == NULL or ipt[0] == Py_None:
        memset(opt, pad, sizes[0].dst_pad)
        return 0

    try:
        if PyBytes_CheckExact(ipt[0]):
            cstring = PyBytes_AsString(ipt[0])
        else:
            pystring = PyObject_Str(ipt[0])
            cstring = PyBytes_AsString(pystring)
        
        cstring_len = strlen(cstring)

        if cstring_len <= sizes[0].dst_size:
            # Input is smaller; copy and pad
            memcpy(opt, cstring, cstring_len)
            memset(opt+cstring_len, pad, sizes[0].dst_size-cstring_len)
        else:
            # Input is bigger; truncate
            memcpy(opt, cstring, sizes[0].dst_size)
    finally:
        Py_XDECREF(pystring)

    return 0

cdef herr_t init_bytes2fixed(hid_t src, hid_t dst, void** priv) except -1:

    if H5Tequal(src, H5PY_BYTES) and (not H5Tis_variable_str(dst)):
        return init_generic(src, dst, priv)
    return -2

cdef herr_t bytes2fixed(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl,  <conv_operator_t>conv_bytes2fixed, init_bytes2fixed, H5T_BKG_NO)


# Py bytes -> HDF5 vlen
cdef int conv_bytes2vlen(PyObject** ipt, char** opt, void* bkg, conv_size_t sizes) except -1:

    cdef PyObject* pystring = NULL
    cdef char* cstring = NULL
    cdef size_t cstring_len = 0

    # TODO: charset validation?

    if ipt[0] == NULL or ipt[0] == Py_None:
        opt[0] = <char*>malloc(1)
        opt[0][0] = c'\0'
        return 0

    try:
        if PyBytes_CheckExact(ipt[0]):
            cstring = PyBytes_AsString(ipt[0])
        else:
            pystring = PyObject_Str(ipt[0])
            cstring = PyBytes_AsString(pystring)

        cstring_len = strlen(cstring)

        opt[0] = <char*>malloc(cstring_len+1)
        memcpy(opt[0], cstring, cstring_len+1)

    finally:
        Py_XDECREF(pystring)

    return 0

cdef herr_t init_bytes2vlen(hid_t src, hid_t dst, void** priv) except -1:

    if H5Tequal(src, H5PY_BYTES) and H5Tis_variable_str(dst):
        return init_generic(src, dst, priv)
    return -2

cdef herr_t bytes2vlen(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl,  <conv_operator_t>conv_bytes2vlen, init_bytes2vlen, H5T_BKG_NO)


# Omitted
# Py bytes -> Py bytes
# Py bytes -> Py unicode
# Py bytes -> utf32

# =============================================================================
# Section 4: Py unicode to all destination types

# Py unicode -> HDF5 fixed
cdef int conv_unicode2fixed(PyObject** ipt, char* opt, void* bkg, conv_size_t* sizes) except -1:

    cdef PyObject* pyunicode = NULL
    cdef PyObject* pybytes = NULL
    cdef char* cstring = NULL
    cdef size_t cstring_len = 0
    cdef char pad = c'\0'

    if sizes[0].dst_pad == H5T_STR_SPACEPAD:
        pad = c' '

    if ipt[0] == NULL or ipt[0] == Py_None:
        memset(opt, pad, sizes[0].dst_size)
        return 0

    try:
        if PyUnicode_CheckExact(ipt[0]):
            pybytes = PyUnicode_AsUTF8String(ipt[0])
            cstring = PyBytes_AsString(pybytes)
        else:
            pyunicode = PyObject_Unicode(ipt[0])
            pybytes = PyUnicode_AsUTF8String(pyunicode)
            cstring = PyBytes_AsString(pybytes)

        cstring_len = strlen(cstring)

        if cstring_len <= sizes[0].dst_size:
            # Input is shorter; copy & pad
            memcpy(opt, cstring, cstring_len)
            memset(opt+cstring_len, pad, sizes[0].dst_size-cstring_len)
        else:
            # Input is longer; truncate
            # TODO: watch out for multibyte utf-8 sequences
            memcpy(opt, cstring, sizes[0].dst_size)
    finally:
        Py_XDECREF(pyunicode)
        Py_XDECREF(pybytes)

    return 0

cdef herr_t init_unicode2fixed(hid_t src, hid_t dst, void** priv) except -1:

    if H5Tequal(src, H5PY_UNICODE) and not H5Tis_variable_str(dst):
        return init_generic(src, dst, priv)
    return -2

cdef herr_t unicode2fixed(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl,  <conv_operator_t>conv_unicode2fixed, init_unicode2fixed, H5T_BKG_NO)


# Py unicode -> HDF5 vlen
cdef int conv_unicode2vlen(PyObject** ipt, char** opt, void* bkg, conv_size_t* sizes) except -1:

    cdef PyObject* pyunicode = NULL
    cdef PyObject* pybytes = NULL
    cdef char* cstring = NULL
    cdef size_t cstring_len = 0

    try:
        if PyUnicode_CheckExact(ipt[0]):
            pybytes = PyUnicode_AsUTF8String(ipt[0])
            cstring = PyBytes_AsString(pybytes)
        else:
            pyunicode = PyObject_Unicode(ipt[0])
            pybytes = PyUnicode_AsUTF8String(pyunicode)
            cstring = PyBytes_AsString(pybytes)

        cstring_len = strlen(cstring)

        opt[0] = <char*>malloc(cstring_len+1)
        memcpy(opt[0], cstring, cstring_len+1)
    finally:
        Py_XDECREF(pyunicode)
        Py_XDECREF(pybytes)

    return 0

cdef herr_t init_unicode2vlen(hid_t src, hid_t dst, void** priv) except -1:

    if H5Tequal(src, H5PY_UNICODE) and H5Tis_variable_str(dst):
        return init_generic(src, dst, priv)
    return -2

cdef herr_t unicode2vlen(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl,  <conv_operator_t>conv_unicode2vlen, init_unicode2vlen, H5T_BKG_NO)


# Omitted
# Py unicode -> Py bytes
# Py unicode -> utf32

# =============================================================================
# Section 5: UTF32 to all destination types

# UTF32 -> HDF5 fixed
cdef int conv_utf322fixed(npy_ucs4* ipt, char* opt, void* bkg, conv_size_t* sizes) except -1:

    raise NotImplementedError("utf32 -> fixed-length")

cdef herr_t init_utf322fixed(hid_t src, hid_t dst, void** priv) except -1:

    if (H5Tequal(src, H5PY_UTF32_LE) or H5Tequal(src, H5PY_UTF32_BE)) and not H5Tis_variable_str(dst):
        return init_generic(src, dst, priv)
    return -2

cdef herr_t utf322fixed(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl,  <conv_operator_t>conv_utf322fixed, init_utf322fixed, H5T_BKG_NO)


# UTF32 -> HDF5 vlen
cdef int conv_utf322vlen(npy_ucs4* ipt, char** opt, void* bkg, conv_size_t* sizes) except -1:

    raise NotImplementedError("utf32 -> vlen")

cdef herr_t init_utf322vlen(hid_t src, hid_t dst, void** priv) except -1:

    if (H5Tequal(src, H5PY_UTF32_LE) or H5Tequal(src, H5PY_UTF32_BE)) and H5Tis_variable_str(dst):
        return init_generic(src, dst, priv)
    return -2

cdef herr_t utf322vlen(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl,  <conv_operator_t>conv_utf322vlen, init_utf322vlen, H5T_BKG_NO)


# Omitted
# utf32 -> bytes
# utf32 -> unicode
# utf32 -> utf32

# =============================================================================
# Section 6: Conversion HDF5 object references and Python Reference objects

# HDF5 object reference -> Py Reference() object
cdef int conv_objref2pyref(void* ipt, void* opt, void* bkg, void* priv) except -1:

    cdef PyObject** buf_obj = <PyObject**>opt
    cdef PyObject** bkg_obj = <PyObject**>bkg
    cdef hobj_ref_t* buf_ref = <hobj_ref_t*>ipt

    cdef Reference ref = Reference()
    cdef PyObject* ref_ptr = NULL

    ref.ref.obj_ref = buf_ref[0]
    ref.typecode = H5R_OBJECT

    ref_ptr = <PyObject*>ref
    Py_INCREF(ref_ptr)  # because Cython discards its reference when the
                        # function exits

    Py_XDECREF(bkg_obj[0])
    buf_obj[0] = ref_ptr

    return 0

cdef herr_t objref2pyref(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl,  <conv_operator_t>conv_objref2pyref, init_generic, H5T_BKG_YES)

# Py Reference() -> HDF5 object reference
cdef int conv_pyref2objref(void* ipt, void* opt, void* bkg, void* priv) except -1:

    cdef PyObject** buf_obj = <PyObject**>ipt
    cdef hobj_ref_t* buf_ref = <hobj_ref_t*>opt

    cdef object obj
    cdef Reference ref

    if buf_obj[0] != NULL and buf_obj[0] != Py_None:
        obj = <object>(buf_obj[0])
        if not isinstance(obj, Reference):
            raise TypeError("Can't convert incompatible object to HDF5 object reference")
        ref = <Reference>(buf_obj[0])
        buf_ref[0] = ref.ref.obj_ref
    else:
        memset(buf_ref, c'\0', sizeof(hobj_ref_t))

    return 0

cdef herr_t pyref2objref(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl,  <conv_operator_t>conv_pyref2objref, init_generic, H5T_BKG_NO)

# =============================================================================
# Section 7: Conversion between HDF5 region references and RegionReference()

# HDF5 region reference -> Py RegionReference()
cdef int conv_regref2pyref(void* ipt, void* opt, void* bkg, void* priv) except -1:

    cdef PyObject** buf_obj = <PyObject**>opt
    cdef PyObject** bkg_obj = <PyObject**>bkg
    cdef hdset_reg_ref_t* buf_ref = <hdset_reg_ref_t*>ipt

    cdef RegionReference ref = RegionReference()
    cdef PyObject* ref_ptr = NULL

    memcpy(ref.ref.reg_ref, buf_ref, sizeof(hdset_reg_ref_t))

    ref.typecode = H5R_DATASET_REGION

    ref_ptr = <PyObject*>ref
    Py_INCREF(ref_ptr)  # because Cython discards its reference when the
                        # function exits

    Py_XDECREF(bkg_obj[0])
    buf_obj[0] = ref_ptr

    return 0

cdef herr_t regref2pyref(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl,  <conv_operator_t>conv_regref2pyref, init_generic, H5T_BKG_YES)

# Py RegionReference() -> HDF5 region reference
cdef int conv_pyref2regref(void* ipt, void* opt, void* bkg, void* priv) except -1:

    cdef PyObject** buf_obj = <PyObject**>ipt
    cdef hdset_reg_ref_t* buf_ref = <hdset_reg_ref_t*>opt

    cdef object obj
    cdef RegionReference ref

    if buf_obj[0] != NULL and buf_obj[0] != Py_None:
        obj = <object>(buf_obj[0])
        if not isinstance(obj, RegionReference):
            raise TypeError("Can't convert incompatible object to HDF5 region reference")
        ref = <RegionReference>(buf_obj[0])
        memcpy(buf_ref, ref.ref.reg_ref, sizeof(hdset_reg_ref_t))
    else:
        memset(buf_ref, c'\0', sizeof(hdset_reg_ref_t))

    return 0

cdef herr_t pyref2regref(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl,  <conv_operator_t>conv_pyref2regref, init_generic, H5T_BKG_NO)

# =============================================================================
# Enum to integer converter

cdef struct conv_enum_t:
    size_t src_size
    size_t dst_size
    hid_t supertype
    int identical

# Direction ("forward"): 1 = enum to int, 0 = int to enum
cdef herr_t enum_int_converter(hid_t src, hid_t dst, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl, int forward) except -1 with gil:

    cdef int command = cdata[0].command
    cdef conv_enum_t *info
    cdef size_t nalloc
    cdef int i
    cdef char* cbuf = NULL
    cdef char* buf = <char*>buf_i

    if command == H5T_CONV_INIT:
        cdata[0].need_bkg = H5T_BKG_NO
        cdata[0].priv = info = <conv_enum_t*>malloc(sizeof(conv_enum_t))
        info[0].src_size = H5Tget_size(src)
        info[0].dst_size = H5Tget_size(dst)
        if forward:
            info[0].supertype = H5Tget_super(src)
            info[0].identical = H5Tequal(info[0].supertype, dst)
        else:
            info[0].supertype = H5Tget_super(dst)
            info[0].identical = H5Tequal(info[0].supertype, src)

    elif command == H5T_CONV_FREE:

        info = <conv_enum_t*>cdata[0].priv
        #H5Tclose(info[0].supertype)
        free(info)
        cdata[0].priv = NULL

    elif command == H5T_CONV_CONV:

        info = <conv_enum_t*>cdata[0].priv

        # Short-circuit success
        if info[0].identical:
            return 0

        if buf_stride == 0:
            # Contiguous case: call H5Tconvert directly
            if forward:
                H5Tconvert(info[0].supertype, dst, nl, buf, NULL, dxpl)
            else:
                H5Tconvert(src, info[0].supertype, nl, buf, NULL, dxpl)
        else:
            # Non-contiguous: gather, convert and then scatter
            if info[0].src_size > info[0].dst_size:
                nalloc = info[0].src_size*nl
            else:
                nalloc = info[0].dst_size*nl
            
            cbuf = <char*>malloc(nalloc)
            if cbuf == NULL:
                raise MemoryError("Can't allocate conversion buffer")
            try:
                for i from 0<=i<nl:
                    memcpy(cbuf + (i*info[0].src_size), buf + (i*buf_stride),
                           info[0].src_size)

                if forward:
                    H5Tconvert(info[0].supertype, dst, nl, cbuf, NULL, dxpl)
                else:
                    H5Tconvert(src, info[0].supertype, nl, cbuf, NULL, dxpl)

                for i from 0<=i<nl:
                    memcpy(buf + (i*buf_stride), cbuf + (i*info[0].dst_size),
                           info[0].dst_size)
            finally:
                free(cbuf)
                cbuf = NULL
    else:
        return -2

    return 0
            
cdef herr_t enum2int(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return enum_int_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl, 1)

cdef herr_t int2enum(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return enum_int_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl, 0)


cpdef int register_converters() except -1:

    cdef hid_t vlstring
    cdef hid_t enum

    vlstring = H5Tcopy(H5T_C_S1)
    H5Tset_size(vlstring, H5T_VARIABLE)
    enum = H5Tenum_create(H5T_STD_I32LE)

    # HDF5 string to HDF5 string
    H5Tregister(H5T_PERS_SOFT, "vlen2fixed", vlstring, H5T_C_S1, vlen2fixed)
    H5Tregister(H5T_PERS_SOFT, "fixed2vlen", H5T_C_S1, vlstring, fixed2vlen)

    # HDF5 string to Python pointer-like
    H5Tregister(H5T_PERS_SOFT, "vlen2bytes", vlstring, H5PY_BYTES, vlen2bytes)
    H5Tregister(H5T_PERS_SOFT, "vlen2unicode", vlstring, H5PY_UNICODE, vlen2unicode)
    H5Tregister(H5T_PERS_SOFT, "fixed2utf32le", H5T_C_S1, H5PY_UTF32_LE, fixed2utf32)
    H5Tregister(H5T_PERS_SOFT, "fixed2utf32be", H5T_C_S1, H5PY_UTF32_BE, fixed2utf32)
    
    # Python pointer-like to HDF5 string
    H5Tregister(H5T_PERS_SOFT, "bytes2fixed", H5PY_BYTES, H5T_C_S1, bytes2fixed)
    H5Tregister(H5T_PERS_SOFT, "bytes2vlen", H5PY_BYTES, vlstring, bytes2vlen)
    H5Tregister(H5T_PERS_SOFT, "unicode2fixed", H5PY_UNICODE, H5T_C_S1, unicode2fixed)
    H5Tregister(H5T_PERS_SOFT, "unicode2vlen", H5PY_UNICODE, vlstring, unicode2vlen)
    H5Tregister(H5T_PERS_SOFT, "utf32le2fixed", H5PY_UTF32_LE, H5T_C_S1, utf322fixed)
    H5Tregister(H5T_PERS_SOFT, "utf32be2fixed", H5PY_UTF32_BE, H5T_C_S1, utf322fixed)
    H5Tregister(H5T_PERS_SOFT, "utf32le2vlen", H5PY_UTF32_LE, vlstring, utf322vlen)
    H5Tregister(H5T_PERS_SOFT, "utf32be2vlen", H5PY_UTF32_BE, vlstring, utf322vlen)

    # Section 6
    H5Tregister(H5T_PERS_HARD, "objref2pyref", H5T_STD_REF_OBJ, H5PY_OBJ, objref2pyref)
    H5Tregister(H5T_PERS_HARD, "pyref2objref", H5PY_OBJ, H5T_STD_REF_OBJ, pyref2objref)

    # Section 7
    H5Tregister(H5T_PERS_HARD, "regref2pyref", H5T_STD_REF_DSETREG, H5PY_OBJ, regref2pyref)
    H5Tregister(H5T_PERS_HARD, "pyref2regref", H5PY_OBJ, H5T_STD_REF_DSETREG, pyref2regref)

    # Enums
    H5Tregister(H5T_PERS_SOFT, "enum2int", enum, H5T_STD_I32LE, enum2int)
    H5Tregister(H5T_PERS_SOFT, "int2enum", H5T_STD_I32LE, enum, int2enum)

    H5Tclose(vlstring)
    H5Tclose(enum)

    return 0

cpdef int unregister_converters() except -1:

    return 0

init_python_types()



