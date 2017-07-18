# cython: profile=False

# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Low-level type-conversion routines.
"""

from h5r cimport Reference, RegionReference, hobj_ref_t, hdset_reg_ref_t
from h5t cimport H5PY_OBJ, typewrap, py_create, TypeID
cimport numpy as np
from libc.stdlib cimport realloc

# Initialization
np.import_array()

# Minimal interface for Python objects immune to Cython refcounting
cdef extern from "Python.h":
    
    # From Cython declarations
    ctypedef int PyTypeObject
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

cdef extern from "numpy/arrayobject.h":
    PyTypeObject PyArray_Type
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr, int nd, np.npy_intp* dims, np.npy_intp* strides, void* data, int flags, object obj)


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
    cdef PyObject* bkg_obj0
    cdef char* buf_cstring0

    memcpy(&bkg_obj0, bkg_obj, sizeof(bkg_obj0))
    memcpy(&buf_cstring0, buf_cstring, sizeof(buf_cstring0))

    # When reading we identify H5T_CSET_ASCII as a byte string and
    # H5T_CSET_UTF8 as a utf8-encoded unicode string
    if sizes.cset == H5T_CSET_ASCII:
        if buf_cstring0 == NULL:
            temp_obj = PyBytes_FromString("")
        else:
            temp_obj = PyBytes_FromString(buf_cstring0)
    elif sizes.cset == H5T_CSET_UTF8:
        if buf_cstring0 == NULL:
            temp_obj = PyUnicode_DecodeUTF8("", 0, NULL)
        else:
            temp_obj = PyUnicode_DecodeUTF8(buf_cstring0, strlen(buf_cstring0), NULL)

    # Since all data conversions are by definition in-place, it
    # is our responsibility to free the memory used by the vlens.
    free(buf_cstring0)

    # HDF5 will eventuallly overwrite this target location, so we
    # make sure to decref the object there.
    Py_XDECREF(bkg_obj0)

    # Write the new string object to the buffer in-place
    memcpy(buf_obj, &temp_obj, sizeof(temp_obj));

    return 0

cdef int conv_str2vlen(void* ipt, void* opt, void* bkg, void* priv) except -1:

    cdef PyObject** buf_obj = <PyObject**>ipt
    cdef char** buf_cstring = <char**>opt
    cdef conv_size_t* sizes = <conv_size_t*>priv

    cdef PyObject* temp_object = NULL
    cdef PyObject* temp_encoded = NULL

    cdef char* temp_string = NULL
    cdef size_t temp_string_len = 0  # Not including null term

    cdef PyObject* buf_obj0
    cdef char* buf_cstring0

    memcpy(&buf_obj0, buf_obj, sizeof(buf_obj0))

    try:
        if buf_obj0 == NULL or buf_obj0 == Py_None:
            temp_string = ""
            temp_string_len = 0
        else:
            if PyBytes_CheckExact(buf_obj0):

                # Input is a byte string.  If we're using CSET_UTF8, make sure
                # it's valid UTF-8.  Otherwise just store it.
                temp_object = buf_obj0
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
            elif PyUnicode_CheckExact(buf_obj0):
                temp_object = buf_obj0
                Py_INCREF(temp_object)
                temp_encoded = PyUnicode_AsUTF8String(temp_object)
                temp_string = PyBytes_AsString(temp_encoded)
                temp_string_len = PyBytes_Size(temp_encoded)

            else:
                if sizes.cset == H5T_CSET_ASCII:
                    temp_object = PyObject_Str(buf_obj0)
                    temp_string = PyBytes_AsString(temp_object)
                    temp_string_len = PyBytes_Size(temp_object)
                elif sizes.cset == H5T_CSET_UTF8:
                    temp_object = PyObject_Str(buf_obj0)
                    Py_INCREF(temp_object)
                    temp_encoded = PyUnicode_AsUTF8String(temp_object)
                    Py_INCREF(temp_encoded)
                    temp_string = PyBytes_AsString(temp_encoded)
                    temp_string_len = PyBytes_Size(temp_encoded)
                else:
                    raise TypeError("Unrecognized dataset encoding")
                    
        if strlen(temp_string) != temp_string_len:
            raise ValueError("VLEN strings do not support embedded NULLs")

        buf_cstring0 = <char*>malloc(temp_string_len+1)
        memcpy(buf_cstring0, temp_string, temp_string_len+1)
        memcpy(buf_cstring, &buf_cstring0, sizeof(buf_cstring0));

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
    cdef char* buf_vlen0

    memcpy(&buf_vlen0, buf_vlen, sizeof(buf_vlen0));

    if buf_vlen0 != NULL:
        temp_string = buf_vlen0
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

    memcpy(buf_vlen, &temp_string, sizeof(temp_string));

    return 0

# =============================================================================
# HDF5 references to Python instances of h5r.Reference

cdef int conv_objref2pyref(void* ipt, void* opt, void* bkg, void* priv) except -1:

    cdef PyObject** buf_obj = <PyObject**>opt
    cdef hobj_ref_t* buf_ref = <hobj_ref_t*>ipt

    cdef Reference ref = Reference()
    cdef PyObject* ref_ptr = NULL

    memcpy(&ref.ref.obj_ref, buf_ref, sizeof(ref.ref.obj_ref))
    ref.typecode = H5R_OBJECT

    ref_ptr = <PyObject*>ref
    Py_INCREF(ref_ptr)  # because Cython discards its reference when the
                        # function exits

    memcpy(buf_obj, &ref_ptr, sizeof(ref_ptr))

    return 0

cdef int conv_pyref2objref(void* ipt, void* opt, void* bkg, void* priv) except -1:

    cdef PyObject** buf_obj = <PyObject**>ipt
    cdef hobj_ref_t* buf_ref = <hobj_ref_t*>opt

    cdef object obj
    cdef Reference ref

    cdef PyObject* buf_obj0

    memcpy(&buf_obj0, buf_obj, sizeof(buf_obj0));

    if buf_obj0 != NULL and buf_obj0 != Py_None:
        obj = <object>(buf_obj0)
        if not isinstance(obj, Reference):
            raise TypeError("Can't convert incompatible object to HDF5 object reference")
        ref = <Reference>(buf_obj0)
        memcpy(buf_ref, &ref.ref.obj_ref, sizeof(ref.ref.obj_ref))
    else:
        memset(buf_ref, c'\0', sizeof(hobj_ref_t))

    return 0

cdef int conv_regref2pyref(void* ipt, void* opt, void* bkg, void* priv) except -1:

    cdef PyObject** buf_obj = <PyObject**>opt
    cdef PyObject** bkg_obj = <PyObject**>bkg
    cdef hdset_reg_ref_t* buf_ref = <hdset_reg_ref_t*>ipt

    cdef RegionReference ref = RegionReference()
    cdef PyObject* ref_ptr = NULL

    cdef PyObject* bkg_obj0

    memcpy(&bkg_obj0, bkg_obj, sizeof(bkg_obj0));
    memcpy(ref.ref.reg_ref, buf_ref, sizeof(hdset_reg_ref_t))

    ref.typecode = H5R_DATASET_REGION

    ref_ptr = <PyObject*>ref
    Py_INCREF(ref_ptr)  # because Cython discards its reference when the
                        # function exits

    Py_XDECREF(bkg_obj0)
    memcpy(buf_obj, &ref_ptr, sizeof(ref_ptr))

    return 0

cdef int conv_pyref2regref(void* ipt, void* opt, void* bkg, void* priv) except -1:

    cdef PyObject** buf_obj = <PyObject**>ipt
    cdef hdset_reg_ref_t* buf_ref = <hdset_reg_ref_t*>opt

    cdef object obj
    cdef RegionReference ref

    cdef PyObject* buf_obj0

    memcpy(&buf_obj0, buf_obj, sizeof(buf_obj0));

    if buf_obj0 != NULL and buf_obj0 != Py_None:
        obj = <object>(buf_obj0)
        if not isinstance(obj, RegionReference):
            raise TypeError("Can't convert incompatible object to HDF5 region reference")
        ref = <RegionReference>(buf_obj0)
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
             buf_i, bkg_i, dxpl, conv_objref2pyref, init_generic, H5T_BKG_NO)

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
    cdef int identical
    cdef hid_t supertype = -1
    
    info = <conv_enum_t*>cdata[0].priv
    
    try:
        if forward:
            supertype = H5Tget_super(src)
            identical = H5Tequal(supertype, dst)
        else:
            supertype = H5Tget_super(dst)
            identical = H5Tequal(supertype, src)
   
        # Short-circuit success
        if identical:
            return 0

        if buf_stride == 0:
            # Contiguous case: call H5Tconvert directly
            if forward:
                H5Tconvert(supertype, dst, nl, buf, NULL, dxpl)
            else:
                H5Tconvert(src, supertype, nl, buf, NULL, dxpl)
        else:
            # Non-contiguous: gather, convert and then scatter
            if info[0].src_size > info[0].dst_size:
                nalloc = info[0].src_size*nl
            else:
                nalloc = info[0].dst_size*nl

            cbuf = <char*>malloc(nalloc)
            if cbuf == NULL:
                raise MemoryError()

            for i from 0<=i<nl:
                memcpy(cbuf + (i*info[0].src_size), buf + (i*buf_stride),
                        info[0].src_size)

            if forward:
                H5Tconvert(supertype, dst, nl, cbuf, NULL, dxpl)
            else:
                H5Tconvert(src, supertype, nl, cbuf, NULL, dxpl)

            for i from 0<=i<nl:
                memcpy(buf + (i*buf_stride), cbuf + (i*info[0].dst_size),
                        info[0].dst_size)

    finally:
        free(cbuf)
        cbuf = NULL
        if supertype > 0:
            H5Tclose(supertype)

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

# =============================================================================
# ndarray to VLEN routines

cdef herr_t vlen2ndarray(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:

    cdef int command = cdata[0].command
    cdef size_t src_size, dst_size
    cdef TypeID supertype
    cdef TypeID outtype
    cdef np.dtype dt
    cdef int i

    cdef char* buf = <char*>buf_i

    if command == H5T_CONV_INIT:

        cdata[0].need_bkg = H5T_BKG_NO
        if H5Tget_class(src_id) != H5T_VLEN or H5Tget_class(dst_id) != H5T_OPAQUE:
            return -2

    elif command == H5T_CONV_FREE:
        
        pass

    elif command == H5T_CONV_CONV:

        # need to pass element dtype to converter
        supertype = typewrap(H5Tget_super(src_id))
        dt = supertype.dtype
        outtype = py_create(dt)

        if buf_stride == 0:
            # No explicit stride seems to mean that the elements are packed
            # contiguously in the buffer.  In this case we must be careful
            # not to "stomp on" input elements if the output elements are
            # of a larger size.

            src_size = H5Tget_size(src_id)
            dst_size = H5Tget_size(dst_id)

            if src_size >= dst_size:
                for i from 0<=i<nl:
                    conv_vlen2ndarray(buf + (i*src_size), buf + (i*dst_size),
                                      dt, supertype, outtype)
            else:
                for i from nl>i>=0:
                    conv_vlen2ndarray(buf + (i*src_size), buf + (i*dst_size),
                                      dt, supertype, outtype)
        else:
            # With explicit strides, we assume that the library knows the
            # alignment better than us.  Therefore we use the given stride
            # offsets exclusively.
            for i from 0<=i<nl:
                conv_vlen2ndarray(buf + (i*buf_stride), buf + (i*buf_stride),
                                  dt, supertype, outtype)

    else:
        return -2   # Unrecognized command.  Note this is NOT an exception.

    return 0


cdef struct vlen_t:
    size_t len
    void* ptr

cdef int conv_vlen2ndarray(void* ipt, void* opt, np.dtype elem_dtype,
        TypeID intype, TypeID outtype) except -1:

    cdef PyObject** buf_obj = <PyObject**>opt
    cdef vlen_t* in_vlen = <vlen_t*>ipt
    cdef int flags = np.NPY_WRITEABLE | np.NPY_C_CONTIGUOUS
    cdef np.npy_intp dims[1]
    cdef void* data
    cdef np.ndarray ndarray
    cdef PyObject* ndarray_obj
    cdef vlen_t in_vlen0

    memcpy(&in_vlen0, in_vlen, sizeof(in_vlen0))

    dims[0] = in_vlen0.len
    data = in_vlen0.ptr
    if outtype.get_size() > intype.get_size():
        data = realloc(data, outtype.get_size() * in_vlen0.len)
    H5Tconvert(intype.id, outtype.id, in_vlen0.len, data, NULL, H5P_DEFAULT)
    
    Py_INCREF(<PyObject*>elem_dtype)
    ndarray = PyArray_NewFromDescr(&PyArray_Type, elem_dtype, 1,
                dims, NULL, data, flags, <object>NULL)
    ndarray.flags |= np.NPY_OWNDATA
    ndarray_obj = <PyObject*>ndarray
    Py_INCREF(ndarray_obj)

    # Write the new object to the buffer in-place
    in_vlen0.ptr = NULL
    memcpy(buf_obj, &ndarray_obj, sizeof(ndarray_obj))
    
    return 0

cdef herr_t ndarray2vlen(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl) except -1:

    cdef int command = cdata[0].command
    cdef size_t src_size, dst_size
    cdef TypeID supertype
    cdef TypeID outtype
    cdef np.dtype dt
    cdef int i
    cdef PyObject **pdata = <PyObject **> buf_i
    cdef PyObject *pdata_elem

    cdef char* buf = <char*>buf_i

    if command == H5T_CONV_INIT:

        cdata[0].need_bkg = H5T_BKG_NO
        if not H5Tequal(src_id, H5PY_OBJ) or H5Tget_class(dst_id) != H5T_VLEN:
            return -2
        supertype = typewrap(H5Tget_super(dst_id))
        for i from 0 <= i < nl:
            memcpy(&pdata_elem, pdata+i, sizeof(pdata_elem))
            if supertype != py_create((<np.ndarray> pdata_elem).dtype, 1):
                return -2
            if (<np.ndarray> pdata_elem).ndim != 1:
                return -2

    elif command == H5T_CONV_FREE:
        
        pass

    elif command == H5T_CONV_CONV:

        # need to pass element dtype to converter
        memcpy(&pdata_elem, pdata, sizeof(pdata_elem))
        supertype = py_create((<np.ndarray> pdata_elem).dtype)
        outtype = typewrap(H5Tget_super(dst_id))

        if buf_stride == 0:
            # No explicit stride seems to mean that the elements are packed
            # contiguously in the buffer.  In this case we must be careful
            # not to "stomp on" input elements if the output elements are
            # of a larger size.

            src_size = H5Tget_size(src_id)
            dst_size = H5Tget_size(dst_id)

            if src_size >= dst_size:
                for i from 0<=i<nl:
                    conv_ndarray2vlen(buf + (i*src_size), buf + (i*dst_size),
                                      supertype, outtype)
            else:
                for i from nl>i>=0:
                    conv_ndarray2vlen(buf + (i*src_size), buf + (i*dst_size),
                                      supertype, outtype)
        else:
            # With explicit strides, we assume that the library knows the
            # alignment better than us.  Therefore we use the given stride
            # offsets exclusively.
            for i from 0<=i<nl:
                conv_ndarray2vlen(buf + (i*buf_stride), buf + (i*buf_stride),
                                  supertype, outtype)

    else:
        return -2   # Unrecognized command.  Note this is NOT an exception.

    return 0


cdef int conv_ndarray2vlen(void* ipt, void* opt,
        TypeID intype, TypeID outtype) except -1:

    cdef PyObject** buf_obj = <PyObject**>ipt
    cdef vlen_t* in_vlen = <vlen_t*>opt
    cdef int flags = np.NPY_WRITEABLE | np.NPY_C_CONTIGUOUS
    cdef np.npy_intp dims[1]
    cdef void* data
    cdef np.ndarray ndarray
    cdef size_t len
    cdef PyObject* buf_obj0

    memcpy(&buf_obj0, buf_obj, sizeof(buf_obj0))
    ndarray = <np.ndarray> buf_obj0
    len = ndarray.shape[0]

    if outtype.get_size() > intype.get_size():
        data = malloc(outtype.get_size() * len)
    else:
        data = malloc(intype.get_size() * len)
    memcpy(data, ndarray.data, intype.get_size() * len)
    H5Tconvert(intype.id, outtype.id, len, data, NULL, H5P_DEFAULT)

    memcpy(&in_vlen[0].len, &len, sizeof(len))
    memcpy(&in_vlen[0].ptr, &data, sizeof(data))
    
    return 0
            
# =============================================================================

cpdef int register_converters() except -1:

    cdef hid_t vlstring
    cdef hid_t vlentype
    cdef hid_t pyobj
    cdef hid_t enum

    vlstring = H5Tcopy(H5T_C_S1)
    H5Tset_size(vlstring, H5T_VARIABLE)
    
    enum = H5Tenum_create(H5T_STD_I32LE)

    vlentype = H5Tvlen_create(H5T_STD_I32LE)

    pyobj = H5PY_OBJ

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

    H5Tregister(H5T_PERS_SOFT, "vlen2ndarray", vlentype, pyobj, vlen2ndarray)
    H5Tregister(H5T_PERS_SOFT, "ndarray2vlen", pyobj, vlentype, ndarray2vlen)

    H5Tclose(vlstring)
    H5Tclose(vlentype)
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

    H5Tunregister(H5T_PERS_SOFT, "vlen2ndarray", -1, -1, vlen2ndarray)
    H5Tunregister(H5T_PERS_SOFT, "ndarray2vlen", -1, -1, ndarray2vlen)

    return 0
