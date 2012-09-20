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

"""
    Low-level type-conversion routines.
"""

from h5r cimport Reference, RegionReference, hobj_ref_t, hdset_reg_ref_t

# Minimal interface for Python objects immune to Cython refcounting
cdef extern from "Python.h":
    
    # From Cython declarations
    ctypedef void PyTypeObject
    ctypedef struct PyObject:
        Py_ssize_t ob_refcnt
        PyTypeObject *ob_type

    PyObject* PyBytes_FromString(char* str) except NULL
    int PyBytes_CheckExact(PyObject* str) except *
    int PyBytes_Size(PyObject* obj) except *
    PyObject* PyString_AsDecodedObject(PyObject* s, char *encoding, char *errors) except NULL

    PyObject* PyUnicode_DecodeUTF8(char *s, Py_ssize_t size, char *errors) except NULL
    int PyUnicode_CheckExact(PyObject* str) except *
    PyObject* PyUnicode_AsUTF8String(PyObject* s) except NULL

    PyObject* PyObject_Str(PyObject* obj) except NULL
    #PyObject* (PyObject* obj) except NULL
    char* PyBytes_AsString(PyObject* obj) except NULL

    PyObject* Py_None
    void Py_INCREF(PyObject* obj)
    void Py_DECREF(PyObject* obj)
    void Py_XDECREF(PyObject* obj)

cdef object objectify(PyObject* o):
    Py_INCREF(o)
    return <object>o

# Create Python object equivalents
cdef hid_t H5PY_OBJ = 0

cpdef hid_t get_python_obj():
    global H5PY_OBJ
    if H5PY_OBJ <= 0:
        H5PY_OBJ = H5Tcreate(H5T_OPAQUE, sizeof(PyObject*))
        H5Tset_tag(H5PY_OBJ, "PYTHON:OBJECT")
        H5Tlock(H5PY_OBJ)
    return H5PY_OBJ

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

        if H5Tis_variable_str(src_id):
            sizes.cset = H5Tget_cset(src_id)
        elif H5Tis_variable_str(dst_id):
            sizes.cset = H5Tget_cset(dst_id)

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
    int cset

cdef herr_t init_generic(hid_t src, hid_t dst, void** priv) except -1:
    
    cdef conv_size_t *sizes
    sizes = <conv_size_t*>malloc(sizeof(conv_size_t))
    priv[0] = sizes
    sizes[0].src_size = H5Tget_size(src)
    sizes[0].dst_size = H5Tget_size(dst)

    return 0

# =============================================================================
# Vlen string conversion

cdef int conv_vlen2str(void* ipt, void* opt, void* bkg, void* priv) except -1:

    cdef PyObject** buf_obj = <PyObject**>opt
    cdef PyObject** bkg_obj = <PyObject**>bkg
    cdef char** buf_cstring = <char**>ipt
    cdef PyObject* temp_obj = NULL
    cdef conv_size_t *sizes = <conv_size_t*>priv

    # When reading we identify H5T_CSET_ASCII as a byte string and
    # H5T_CSET_UTF8 as a utf8-encoded unicode string
    if sizes.cset == H5T_CSET_ASCII:
        if buf_cstring[0] == NULL:
            temp_obj = PyBytes_FromString("")
        else:
            temp_obj = PyBytes_FromString(buf_cstring[0])
    elif sizes.cset == H5T_CSET_UTF8:
        if buf_cstring[0] == NULL:
            temp_obj = PyUnicode_DecodeUTF8("", 0, NULL)
        else:
            temp_obj = PyUnicode_DecodeUTF8(buf_cstring[0], strlen(buf_cstring[0]), NULL)

    # Since all data conversions are by definition in-place, it
    # is our responsibility to free the memory used by the vlens.
    free(buf_cstring[0])

    # HDF5 will eventuallly overwrite this target location, so we
    # make sure to decref the object there.
    Py_XDECREF(bkg_obj[0])

    # Write the new string object to the buffer in-place
    buf_obj[0] = temp_obj

    return 0

cdef int conv_str2vlen(void* ipt, void* opt, void* bkg, void* priv) except -1:

    cdef PyObject** buf_obj = <PyObject**>ipt
    cdef char** buf_cstring = <char**>opt
    cdef conv_size_t* sizes = <conv_size_t*>priv

    cdef PyObject* temp_object = NULL
    cdef PyObject* temp_encoded = NULL

    cdef char* temp_string = NULL
    cdef size_t temp_string_len = 0  # Not including null term

    try:
        if buf_obj[0] == NULL or buf_obj[0] == Py_None:
            temp_string = ""
            temp_string_len = 0
        else:
            if PyBytes_CheckExact(buf_obj[0]):

                # Input is a byte string.  If we're using CSET_UTF8, make sure
                # it's valid UTF-8.  Otherwise just store it.
                temp_object = buf_obj[0]
                Py_INCREF(temp_object)
                if sizes.cset == H5T_CSET_UTF8:
                    try:
                        pass # disabled for Python 3 compatibility
                        #temp_encoded = PyString_AsDecodedObject(temp_object, "utf8", NULL)
                    except:
                        raise ValueError("Byte string is not valid utf-8 and can't be stored in a utf-8 dataset")
                temp_string = PyBytes_AsString(temp_object)
                temp_string_len = PyBytes_Size(temp_object)

            # We are given a Unicode object.  Encode it to utf-8 regardless of
            # the HDF5 character set.
            elif PyUnicode_CheckExact(buf_obj[0]):
                temp_object = buf_obj[0]
                Py_INCREF(temp_object)
                temp_encoded = PyUnicode_AsUTF8String(temp_object)
                Py_INCREF(temp_encoded)
                temp_string = PyBytes_AsString(temp_encoded)
                temp_string_len = PyBytes_Size(temp_encoded)

            else:
                if sizes.cset == H5T_CSET_ASCII:
                    temp_object = PyObject_Str(buf_obj[0])
                    temp_string = PyBytes_AsString(temp_object)
                    temp_string_len = PyBytes_Size(temp_object)
                elif sizes.cset == H5T_CSET_UTF8:
                    temp_object = PyObject_Str(buf_obj[0])
                    Py_INCREF(temp_object)
                    temp_encoded = PyUnicode_AsUTF8String(temp_object)
                    Py_INCREF(temp_encoded)
                    temp_string = PyBytes_AsString(temp_encoded)
                    temp_string_len = PyBytes_Size(temp_encoded)
                else:
                    raise TypeError("Unrecognized dataset encoding")
                    
        if strlen(temp_string) != temp_string_len:
            raise ValueError("VLEN strings do not support embedded NULLs")

        buf_cstring[0] = <char*>malloc(temp_string_len+1)
        memcpy(buf_cstring[0], temp_string, temp_string_len+1)

        return 0
    finally:
        Py_XDECREF(temp_object)
        Py_XDECREF(temp_encoded)

# =============================================================================
# VLEN to fixed-width strings

cdef herr_t init_vlen2fixed(hid_t src, hid_t dst, void** priv) except -1:

    cdef conv_size_t *sizes

    if not (H5Tis_variable_str(src) and (not H5Tis_variable_str(dst))):
        return -2

    sizes = <conv_size_t*>malloc(sizeof(conv_size_t))
    priv[0] = sizes
    sizes[0].src_size = H5Tget_size(src)
    sizes[0].dst_size = H5Tget_size(dst)

    return 0

cdef herr_t init_fixed2vlen(hid_t src, hid_t dst, void** priv) except -1:

    cdef conv_size_t *sizes

    if not (H5Tis_variable_str(dst) and (not H5Tis_variable_str(src))):
        return -2

    sizes = <conv_size_t*>malloc(sizeof(conv_size_t))
    priv[0] = sizes
    sizes[0].src_size = H5Tget_size(src)
    sizes[0].dst_size = H5Tget_size(dst)

    return 0

cdef int conv_vlen2fixed(void* ipt, void* opt, void* bkg, void* priv) except -1:

    cdef char** buf_vlen = <char**>ipt
    cdef char* buf_fixed = <char*>opt
    cdef char* temp_string = NULL
    cdef size_t temp_string_len = 0  # Without null term
    cdef conv_size_t *sizes = <conv_size_t*>priv

    if buf_vlen[0] != NULL:
        temp_string = buf_vlen[0]
        temp_string_len = strlen(temp_string)

        if temp_string_len <= sizes[0].dst_size:
            # Pad with zeros
            memcpy(buf_fixed, temp_string, temp_string_len)
            memset(buf_fixed + temp_string_len, c'\0', sizes[0].dst_size - temp_string_len)
        else:
            # Simply truncate the string
            memcpy(buf_fixed, temp_string, sizes[0].dst_size)
    else:
        memset(buf_fixed, c'\0', sizes[0].dst_size)

    return 0

cdef int conv_fixed2vlen(void* ipt, void* opt, void* bkg, void* priv) except -1:

    cdef char** buf_vlen = <char**>opt
    cdef char* buf_fixed = <char*>ipt
    cdef char* temp_string = NULL
    cdef conv_size_t *sizes = <conv_size_t*>priv

    temp_string = <char*>malloc(sizes[0].src_size+1)
    memcpy(temp_string, buf_fixed, sizes[0].src_size)
    temp_string[sizes[0].src_size] = c'\0'

    buf_vlen[0] = temp_string

    return 0

# =============================================================================
# HDF5 references to Python instances of h5r.Reference

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

# =============================================================================
# Conversion functions


cdef herr_t vlen2str(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl,  conv_vlen2str, init_generic, H5T_BKG_YES)

cdef herr_t str2vlen(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl, conv_str2vlen, init_generic, H5T_BKG_NO)

cdef herr_t vlen2fixed(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl, conv_vlen2fixed, init_vlen2fixed, H5T_BKG_NO)

cdef herr_t fixed2vlen(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl, conv_fixed2vlen, init_fixed2vlen, H5T_BKG_NO)

cdef herr_t objref2pyref(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl, conv_objref2pyref, init_generic, H5T_BKG_YES)

cdef herr_t pyref2objref(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl, conv_pyref2objref, init_generic, H5T_BKG_NO)

cdef herr_t regref2pyref(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl, conv_regref2pyref, init_generic, H5T_BKG_YES)

cdef herr_t pyref2regref(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:
    return generic_converter(src_id, dst_id, cdata, nl, buf_stride, bkg_stride,
             buf_i, bkg_i, dxpl, conv_pyref2regref, init_generic, H5T_BKG_NO)

# =============================================================================
# Enum to integer converter

cdef struct conv_enum_t:
    size_t src_size
    size_t dst_size
    hid_t supertype
    int identical


cdef int enum_int_converter_init(hid_t src, hid_t dst,
                                 H5T_cdata_t *cdata, int forward) except -1 with gil:
    cdef conv_enum_t *info

    cdata[0].need_bkg = H5T_BKG_NO
    cdata[0].priv = info = <conv_enum_t*>malloc(sizeof(conv_enum_t))
    info[0].src_size = H5Tget_size(src)
    info[0].dst_size = H5Tget_size(dst)

cdef void enum_int_converter_free(H5T_cdata_t *cdata):
    cdef conv_enum_t *info

    info = <conv_enum_t*>cdata[0].priv
    free(info)
    cdata[0].priv = NULL


cdef int enum_int_converter_conv(hid_t src, hid_t dst, H5T_cdata_t *cdata,
                                  size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                                 void *bkg_i, hid_t dxpl, int forward) except -1 with gil:
    cdef conv_enum_t *info
    cdef size_t nalloc
    cdef int i
    cdef char* cbuf = NULL
    cdef char* buf = <char*>buf_i

    info = <conv_enum_t*>cdata[0].priv
    
    if forward:
        info[0].supertype = H5Tget_super(src)
        info[0].identical = H5Tequal(info[0].supertype, dst)
    else:
        info[0].supertype = H5Tget_super(dst)
        info[0].identical = H5Tequal(info[0].supertype, src)
   
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
            raise MemoryError()
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
    return 0


# Direction ("forward"): 1 = enum to int, 0 = int to enum
cdef herr_t enum_int_converter(hid_t src, hid_t dst, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                               void *bkg_i, hid_t dxpl, int forward) except -1:

    cdef int command = cdata[0].command

    if command == H5T_CONV_INIT:
        enum_int_converter_init(src, dst, cdata, forward)
    elif command == H5T_CONV_FREE:
        enum_int_converter_free(cdata)
    elif command == H5T_CONV_CONV:
        return enum_int_converter_conv(src, dst, cdata, nl, buf_stride,
                                       bkg_stride, buf_i, bkg_i, dxpl, forward)
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
    cdef hid_t pyobj
    cdef hid_t enum

    vlstring = H5Tcopy(H5T_C_S1)
    H5Tset_size(vlstring, H5T_VARIABLE)
    
    enum = H5Tenum_create(H5T_STD_I32LE)

    pyobj = get_python_obj()

    H5Tregister(H5T_PERS_HARD, "vlen2str", vlstring, pyobj, vlen2str)
    H5Tregister(H5T_PERS_HARD, "str2vlen", pyobj, vlstring, str2vlen)

    H5Tregister(H5T_PERS_SOFT, "vlen2fixed", vlstring, H5T_C_S1, vlen2fixed)
    H5Tregister(H5T_PERS_SOFT, "fixed2vlen", H5T_C_S1, vlstring, fixed2vlen)

    H5Tregister(H5T_PERS_HARD, "objref2pyref", H5T_STD_REF_OBJ, pyobj, objref2pyref)
    H5Tregister(H5T_PERS_HARD, "pyref2objref", pyobj, H5T_STD_REF_OBJ, pyref2objref)

    H5Tregister(H5T_PERS_HARD, "regref2pyref", H5T_STD_REF_DSETREG, pyobj, regref2pyref)
    H5Tregister(H5T_PERS_HARD, "pyref2regref", pyobj, H5T_STD_REF_DSETREG, pyref2regref)

    H5Tregister(H5T_PERS_SOFT, "enum2int", enum, H5T_STD_I32LE, enum2int)
    H5Tregister(H5T_PERS_SOFT, "int2enum", H5T_STD_I32LE, enum, int2enum)

    H5Tclose(vlstring)
    H5Tclose(enum)

    return 0

cpdef int unregister_converters() except -1:

    H5Tunregister(H5T_PERS_HARD, "vlen2str", -1, -1, vlen2str)
    H5Tunregister(H5T_PERS_HARD, "str2vlen", -1, -1, str2vlen)

    H5Tunregister(H5T_PERS_SOFT, "vlen2fixed", -1, -1, vlen2fixed)
    H5Tunregister(H5T_PERS_SOFT, "fixed2vlen", -1, -1, fixed2vlen)

    H5Tunregister(H5T_PERS_HARD, "objref2pyref", -1, -1, objref2pyref)
    H5Tunregister(H5T_PERS_HARD, "pyref2objref", -1, -1, pyref2objref)

    H5Tunregister(H5T_PERS_HARD, "regref2pyref", -1, -1, regref2pyref)
    H5Tunregister(H5T_PERS_HARD, "pyref2regref", -1, -1, pyref2regref)

    H5Tunregister(H5T_PERS_SOFT, "enum2int", -1, -1, enum2int)
    H5Tunregister(H5T_PERS_SOFT, "int2enum", -1, -1, int2enum)

    return 0





