/***** Preamble block *********************************************************
* 
* This file is part of h5py, a low-level Python interface to the HDF5 library.
* 
* Copyright (C) 2008 Andrew Collette
* http://h5py.alfven.org
* License: BSD  (See LICENSE.txt for full license)
* 
* $Date$
* 
****** End preamble block ****************************************************/

#include <stdio.h>
#include <string.h>
#include "Python.h"
#include "hdf5.h"
#include "typeconv.h"

hid_t _H5PY_OBJ = 0;

/*  Return the (locked) canonical type corresponding to a Python object
    pointer in memory.  This is an ephemeral type; it should never be stored
    in a file. */
hid_t h5py_object_type(void){
    if(_H5PY_OBJ == 0){
        _H5PY_OBJ = H5Tcreate(H5T_OPAQUE, sizeof(PyObject*));
        H5Tset_tag(_H5PY_OBJ, "PYTHON:OBJECT");
        H5Tlock(_H5PY_OBJ);
    }
    return _H5PY_OBJ;
}


/* === Type-conversion callbacks & support === */

/* Check types for Python string/vlen conversion */
htri_t h5py_match_str_ptr(hid_t str, hid_t pyptr){

    htri_t is_var_str = 0;
    htri_t is_pyptr = 0;
    char* tagval;

    is_var_str = H5Tis_variable_str(str);
    if(is_var_str<0) goto failed;

    tagval = H5Tget_tag(pyptr);
    if(tagval != NULL){
        is_pyptr = !strcmp(tagval, "PYTHON:OBJECT");
    }
    free(tagval);

    return is_var_str && is_pyptr;

    failed:     /* Error target */

    return -1;
}

typedef struct {
    size_t src_size;
    size_t dst_size;
} conv_size_t;

/*  Convert from HDF5 variable-length strings to Python string objects.
*/
herr_t vlen_to_str(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dxpl){

    PyGILState_STATE gil;

    char** str;
    PyObject** obj;
    PyObject** obj_bkg;
    PyObject* obj_tmp;

    char *buf;
    char *bkg;
    
    conv_size_t *sizes = NULL;

    herr_t retval = -1;
    int i;

    buf = (char*)buf_i;
    bkg = (char*)bkg_i;
    
    switch(cdata->command){

    /*  Determine if we can convert between src_id and dst_id; return 0 if
        possible, -1 otherwise */
    case H5T_CONV_INIT:  

        /*  Only accept the case of vlen H5T_STRING to Python string */
        if(h5py_match_str_ptr(src_id, dst_id) <= 0) goto init_failed;

        cdata->need_bkg = H5T_BKG_YES;
        if((cdata->priv = sizes = (conv_size_t*)malloc(sizeof(conv_size_t))) == NULL) goto init_failed;

        if((sizes->src_size = H5Tget_size(src_id)) == 0) goto init_failed;
        if((sizes->dst_size = H5Tget_size(dst_id)) == 0) goto init_failed;

        return 0;

        init_failed:    /* Error target */
        free(sizes);
        return -1;

    case H5T_CONV_CONV:

        gil = PyGILState_Ensure();

        sizes = (conv_size_t*)(cdata->priv);

        if(buf_stride==0) buf_stride = sizes->src_size;
        if(bkg_stride==0) bkg_stride = sizes->dst_size;

        for(i=0;i<nl;i++){

            obj = (PyObject**)(buf+(i*buf_stride));
            str = (char**)(buf+(i*buf_stride));
            obj_bkg = (PyObject**)(bkg+(i*bkg_stride));

            if((*str)==NULL){
                obj_tmp = PyString_FromString("");
            } else {
                obj_tmp = PyString_FromString(*str);
            }
            if(obj_tmp==NULL) goto conv_failed;

            /* Since all data conversions are by convention in-place, it
               is our responsibility to free the memory used by the vlens. */
            free(*str);

            Py_XDECREF(*obj_bkg);
            *obj = obj_tmp;
        }

        PyGILState_Release(gil);
        return 0;

        conv_failed:    /* Error target */
        
        PyGILState_Release(gil);
        return -1;
        
    case H5T_CONV_FREE:

        free(cdata->priv);
        return 0;

    default:

        return -1;
    }
}


/*  Convert from Python strings to HDF5 vlens.
*/
herr_t str_to_vlen(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dset_xfer_plist){

    PyGILState_STATE gil;

    PyObject**   obj;
    PyObject*    inter_string = NULL;
    char**       str;
    char*        str_tmp;
    Py_ssize_t   len;

    conv_size_t *sizes = NULL;

    herr_t retval = -1;
    int i;

    char* buf = (char*)buf_i;
    char* bkg = (char*)bkg_i;
    
    switch(cdata->command){

    case H5T_CONV_INIT:

        /*  Only accept Python string -> HDF5 vlen */
        if(h5py_match_str_ptr(dst_id, src_id) <= 0) goto init_failed;

        cdata->need_bkg = H5T_BKG_NO;
        if((cdata->priv = sizes = (conv_size_t*)malloc(sizeof(conv_size_t))) == NULL) goto init_failed;

        if((sizes->src_size = H5Tget_size(src_id)) == 0) goto init_failed;
        if((sizes->dst_size = H5Tget_size(dst_id)) == 0) goto init_failed;
        
        return 0;

        init_failed:    /* Error target */
        free(sizes);
        return -1;

    case H5T_CONV_CONV:


        gil = PyGILState_Ensure();
        sizes = (conv_size_t*)(cdata->priv);

        if(buf_stride==0) buf_stride = sizes->src_size;

        for(i=0;i<nl;i++){

            obj = (PyObject**)(buf+(i*buf_stride));
            str = (char**)(buf+(i*buf_stride));

            if(*obj == NULL || *obj == Py_None){
                len = 1;
                str_tmp = "";

            } else { /* If it's not a string, take the result of str(obj) */

                if(PyString_CheckExact(*obj)) {
                    len = PyString_Size(*obj)+1;
                    str_tmp = PyString_AsString(*obj);
                } else {
                    inter_string = PyObject_Str(*obj);
                    if(inter_string == NULL) goto conv_failed;
                    len = PyString_Size(inter_string)+1;
                    str_tmp = PyString_AsString(inter_string);
                }

            }

            *str = (char*)malloc(len);  /* len already includes null term */
            memcpy(*str, str_tmp, len);

        }            

        retval = 0;

        conv_out:

        /* Note we do NOT decref obj, as it is a borrowed reference */
        Py_XDECREF(inter_string);
        PyGILState_Release(gil);
        return retval;

        conv_failed:    /* Error target */
        retval = -1;
        goto conv_out;
        
    case H5T_CONV_FREE:

        free(cdata->priv);
        return 0;

    default:

        return -1;
    }

}

typedef struct {
    size_t src_size;
    size_t dst_size;
    int vlen_to_fixed;
} h5py_vlfix_conv_t;

/* Convert back & forth between fixed and vlen strings.  When converting from
    vlen to fixed, if the string is shorted, the space will be padded with
    nulls; when longer, it will simply be truncated with no null termination.
 */
herr_t vlen_fixed(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dset_xfer_plist){

    htri_t svlen, dvlen;
    h5py_vlfix_conv_t *info = NULL;
    char* buf_ptr;
    char* str_tmp;
    size_t str_tmp_len;

    int i, start, stop, incr;

    char* buf = (char*)buf_i;
    char* bkg = (char*)bkg_i;
    
    switch(cdata->command){

    case H5T_CONV_INIT:

        if((svlen = H5Tis_variable_str(src_id)) < 0) goto init_failed;
        if((dvlen = H5Tis_variable_str(dst_id)) < 0) goto init_failed;

        /* Exactly one must be variable-length */
        if((svlen && dvlen) || !(svlen || dvlen)) goto init_failed;

        if((cdata->priv = info = malloc(sizeof(h5py_vlfix_conv_t))) == NULL) goto init_failed;

        if((info->src_size = H5Tget_size(src_id)) < 0) goto init_failed;
        if((info->dst_size = H5Tget_size(dst_id)) < 0) goto init_failed;

        info->vlen_to_fixed = svlen;

        return 0;

        init_failed:
        free(info);
        return -1;

    case H5T_CONV_CONV:

        info = (h5py_vlfix_conv_t*)(cdata->priv);

        if(buf_stride==0) buf_stride = info->src_size;

        if(info->src_size >= info->dst_size){
            start = 0;
            stop = nl;
            incr = 1;
        } else {
            start = nl-1;
            stop = -1;
            incr = -1;
        }

        if(info->vlen_to_fixed){

            for(i=start; i!=stop; i+=incr){
                buf_ptr = buf + (i*buf_stride);
                str_tmp = *((char**)buf_ptr);
                str_tmp_len = strlen(str_tmp);
                if(str_tmp_len <= info->dst_size){
                    memcpy(buf_ptr, str_tmp, str_tmp_len);
                    memset(buf_ptr + str_tmp_len, info->dst_size - str_tmp_len, '\0');
                } else {
                    memcpy(buf_ptr, str_tmp, info->dst_size);
                }
                free(str_tmp);
            }

        } else {

            for(i=start; i!=stop; i+=incr){
                buf_ptr = buf + (i*buf_stride);
                if((str_tmp = (char*)malloc(info->src_size + 1))==NULL) goto conv_failed;
                memcpy(str_tmp, buf_ptr, info->src_size);
                str_tmp[info->src_size] = '\0';
                *((char**)buf_ptr) = str_tmp;
            }
            
        }

        return 0;

        conv_failed:
        return -1;

    case H5T_CONV_FREE:

        return 0;

    default:

        return -1;
    }
}

/* Convert back & forth between enums and ints */

typedef struct {
    H5T_class_t src_cls;
    size_t src_size;
    size_t dst_size;
    hid_t int_src_id;   /* Integer type appropriate for source */
    hid_t int_dst_id;   /* Integer type appropriate for destination */
    int identical;      /* Tells if the above types are the same */
} h5py_enum_conv_t;

/* This function is registered on both paths ENUM -> INT and INT -> ENUM */
herr_t enum_int(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
                    size_t nl, size_t buf_stride, size_t bkg_stride, void *buf_i,
                    void *bkg_i, hid_t dset_xfer_plist){

    h5py_enum_conv_t* info = NULL;
    hid_t conv_src_id, conv_dst_id;

    char* conv_buf = NULL;
    size_t nalloc;

    herr_t cresult;
    int i;

    char* buf = (char*)buf_i;
    char* bkg = (char*)bkg_i;
    
    switch(cdata->command){

        case H5T_CONV_INIT:
            
            cdata->need_bkg = H5T_BKG_NO;
            if((cdata->priv = info = (h5py_enum_conv_t*)malloc(sizeof(h5py_enum_conv_t))) == NULL) goto init_failed;

            info->int_src_id = 0;
            info->int_dst_id = 0;

            if((info->src_cls = H5Tget_class(src_id)) < 0) goto init_failed;

            if((info->src_size = H5Tget_size(src_id)) == 0) goto init_failed;
            if((info->dst_size = H5Tget_size(dst_id)) == 0) goto init_failed;

            if(info->src_cls == H5T_ENUM){
                /* We're trying to convert an ENUM to an INT */
                info->int_src_id = H5Tget_super(src_id);
                info->int_dst_id = dst_id;
                if(H5Iinc_ref(dst_id) < 0) goto init_failed;
            } else {
                /* We're trying to convert an INT to an ENUM */
                info->int_src_id = src_id;
                info->int_dst_id = H5Tget_super(dst_id);
                if(H5Iinc_ref(src_id) < 0) goto init_failed;
            }
            if(info->int_src_id<0) goto init_failed;
            if(info->int_dst_id<0) goto init_failed;

            if((info->identical = H5Tequal(info->int_src_id, info->int_dst_id)) < 0) goto init_failed;

            return 0;

            init_failed:

            if(info!=NULL){
                if(info->int_src_id>0)  H5Idec_ref(info->int_src_id);
                if(info->int_dst_id>0)  H5Idec_ref(info->int_dst_id);
            }
            free(info);

            return -1;

        case H5T_CONV_CONV:

            info = (h5py_enum_conv_t*)(cdata->priv);

            /* Shortcut */
            if(info->identical) return 0;

            if(buf_stride==0){
                /*  Contiguous data: H5Tconvert can do this directly */

                if(H5Tconvert(info->int_src_id, info->int_dst_id,
                   nl, buf, NULL, dset_xfer_plist) < 0) goto conv_failed;

            } else {
                /*  Can't tell H5Tconvert about strides; use a buffer */

                if( (info->src_size) > (info->dst_size)){
                    nalloc = (info->src_size)*nl;
                } else {
                    nalloc = (info->dst_size)*nl;
                }
                if((conv_buf = malloc(nalloc)) == NULL) goto conv_failed;

                /* Copy into temp buffer */
                for(i=0;i<nl;i++){
                    memcpy(conv_buf+(i*(info->src_size)), buf+(i*buf_stride), 
                           info->src_size);
                }
    
                /* Convert in-place */
                if(H5Tconvert(info->int_src_id, info->int_dst_id,
                     nl, conv_buf, NULL, dset_xfer_plist) < 0) goto conv_failed;

                /*  Copy back out to source buffer.  Remember these elements
                    are now of size info->dst_size. */
                for(i=0;i<nl;i++){
                    memcpy(buf+(i*buf_stride), conv_buf+(i*(info->dst_size)),
                           info->dst_size);
                }

            } /* if ... else */

            free(conv_buf);
            return 0;

            conv_failed:
            free(conv_buf);
            return -1;

        case H5T_CONV_FREE:

            /* Segfault on cleanup; something's wrong with cdata->priv */
            return 0;

        default:

            return -1;

    } /* case */

}        

int h5py_register_conv(void){

    hid_t h5py_enum = H5Tenum_create(H5T_NATIVE_INT);
    hid_t h5py_obj = h5py_object_type();
    hid_t vlen_str = H5Tcopy(H5T_C_S1);
    H5Tset_size(vlen_str, H5T_VARIABLE);

    /*  "Soft" registration means the conversion is tested for any two types
        which match the given classes (in this case H5T_STRING and H55_OPAQUE) */
    H5Tregister(H5T_PERS_SOFT, "vlen_to_str", vlen_str, h5py_obj, vlen_to_str);
    H5Tregister(H5T_PERS_SOFT, "str_to_vlen", h5py_obj, vlen_str, str_to_vlen);

    H5Tregister(H5T_PERS_SOFT, "enum to int", h5py_enum, H5T_NATIVE_INT, enum_int);
    H5Tregister(H5T_PERS_SOFT, "int to enum", H5T_NATIVE_INT, h5py_enum, enum_int);

    H5Tregister(H5T_PERS_SOFT, "fix to vlen", H5T_C_S1, vlen_str, vlen_fixed);
    H5Tregister(H5T_PERS_SOFT, "vlen to fix", vlen_str, H5T_C_S1, vlen_fixed);

    H5Tclose(vlen_str);

    return 0;
}








