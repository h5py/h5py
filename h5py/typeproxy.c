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

/*
    Contains proxy functions for reading and writing data from datasets and
    attributes.  Importantly, these functions implement the proper workarounds
    required for variable-length type support, as implemented in typeconv.c.
*/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "hdf5.h"
#include "typeproxy.h"

/* ------- Private function prototypes ------ */

herr_t h5py_resolve_spaces(hid_t dset_id, hid_t ifspace, hid_t imspace,
                           hid_t *ofspace, hid_t *omspace);

void* h5py_setup_buffer(hid_t itype, hid_t otype, hid_t space_id, hsize_t* nl);

htri_t h5py_detect_vlen(hid_t type_id);


/* ------- Attribute read/write support ------- */

herr_t H5PY_attr_rw(hid_t attr, hid_t mtype, void *buf, h5py_rw_t dir){

    hid_t   atype = 0;      /* Attribute data type */
    hid_t   aspace = 0;     /* Attribute data space */

    htri_t  vlen_present;
    herr_t  status;         /* API function result */
    herr_t  retval;         /* Return value for this function */

    hsize_t mtype_size;
    hsize_t  nl;
    void*   bkg_buf = NULL;
    void*   conv_buf = NULL;

    atype = H5Aget_type(attr);
    if(atype<0) goto failed;

    vlen_present = h5py_detect_vlen(atype);
    if(vlen_present<0) goto failed;

    if(!vlen_present){
        /* Direct read/write */
        
        switch(dir){
        case H5PY_READ:
            status = H5Aread(attr, mtype, buf);
            break;
        case H5PY_WRITE:
            status = H5Awrite(attr, mtype, buf);
            break;
        default:
            goto failed;
        }
        if(status<0) goto failed;
    
    } else {
        /* Buffered read/write */

        aspace = H5Aget_space(attr);
        if(aspace<0) goto failed;

        conv_buf = h5py_setup_buffer(atype, mtype, aspace, &nl);
        if(conv_buf==NULL) goto failed;

        mtype_size = H5Tget_size(mtype);
        if(mtype_size==0) goto failed;

        bkg_buf = malloc(mtype_size*nl);
        if(bkg_buf==NULL) goto failed;

        memcpy(bkg_buf, buf, mtype_size*nl);

        switch(dir){

        case H5PY_READ:
            status = H5Aread(attr, atype, conv_buf);
            if(status<0) goto failed;
            status = H5Tconvert(atype, mtype, nl, conv_buf, bkg_buf, H5P_DEFAULT);
            if(status<0) goto failed;
            memcpy(buf, conv_buf, mtype_size*nl);
            break;

        case H5PY_WRITE:
            memcpy(conv_buf, buf, mtype_size*nl);
            status = H5Tconvert(mtype, atype, nl, conv_buf, bkg_buf, H5P_DEFAULT);
            if(status<0) goto failed;
            status = H5Awrite(attr, atype, conv_buf);
            if(status<0) goto failed;
            break;

        default:
            goto failed;
        }
    }

    retval = 0;

    out:        /* Cleanup */

    free(bkg_buf);
    free(conv_buf);
    if(atype>0)     H5Tclose(atype);
    if(aspace>0)    H5Sclose(aspace);

    return retval;

    failed:     /* Error target */

    retval = -1;
    goto out;

}



/*  H5PY_dset_rw

    Read & write datasets with proxy support for vlen bug.  "Direction"
    determines whether to read or write data.
*/
herr_t H5PY_dset_rw(hid_t dset, hid_t mtype, hid_t mspace_in, hid_t fspace_in,
                   hid_t xfer_plist, void* buf, h5py_rw_t dir){

    hid_t   dstype = 0; 
    hid_t   mspace = 0, fspace =0;
    htri_t  vlen_present;
    herr_t  status;             /* Status flag for API calls */
    herr_t  retval;             /* Return value for this function */

    hsize_t nl;                 /* Number of elements for read/write */
    size_t  mtype_size;
    hid_t cspace = 0;           /* Dataspace for conversion buffer */
    void* conv_buf = NULL;      /* Conversion buffer */
    void* bkg_buf = NULL;       /* Backing buffer */


    dstype = H5Dget_type(dset);
    if(dstype<0) goto failed;

    vlen_present = h5py_detect_vlen(dstype);
    if(vlen_present<0) goto failed;

    if(!vlen_present){
        /* Standard read/write */

        switch(dir){
        case H5PY_READ:
            status = H5Dread(dset, mtype, mspace_in, fspace_in, xfer_plist, buf);
            break;
        case H5PY_WRITE:
            status = H5Dwrite(dset, mtype, mspace_in, fspace_in, xfer_plist, buf);
            break;
        default:
            goto failed;
        }
        if(status<0) goto failed;

    } else {
        /* Buffered read/write */

        status = h5py_resolve_spaces(dset, fspace_in, mspace_in, &fspace, &mspace);
        if(status<0) goto failed;

        conv_buf = h5py_setup_buffer(dstype, mtype, fspace, &nl);
        if(conv_buf==NULL) goto failed;

        cspace = H5Screate_simple(1, &nl, NULL);
        if(cspace<0) goto failed;

        /* Populate the backing buffer with in-memory data */
        /* TODO: skip unless (1) reading (any type), or (2) writing compound */
        mtype_size = H5Tget_size(mtype);
        if(mtype_size==0) goto failed;

        bkg_buf = malloc(mtype_size*nl);

        status = h5py_copy(mtype, mspace, bkg_buf, buf, H5PY_GATHER);
        if(status<0) goto failed;

        switch(dir){

        case H5PY_READ:
            status = H5Dread(dset, dstype, cspace, fspace, xfer_plist, conv_buf);
            if(status<0) goto failed;
            status = H5Tconvert(dstype, mtype, nl, conv_buf, bkg_buf, xfer_plist);
            if(status<0) goto failed;
            status = h5py_copy(mtype, mspace, conv_buf, buf, H5PY_SCATTER);
            if(status<0) goto failed;
            break;

        case H5PY_WRITE:
            status = h5py_copy(mtype, mspace, conv_buf, buf, H5PY_GATHER);
            if(status<0) goto failed;
            status = H5Tconvert(mtype, dstype, nl, conv_buf, bkg_buf, xfer_plist);
            if(status<0) goto failed;
            status = H5Dwrite(dset, dstype, cspace, fspace, xfer_plist, conv_buf);
            if(status<0) goto failed;
            break;

        default:
            goto failed;
        }

    }

    retval = 0;

    out:        /* Cleanup */

    free(conv_buf);
    free(bkg_buf);

    if(dstype>0)    H5Tclose(dstype);
    if(fspace>0)    H5Sclose(fspace);
    if(mspace>0)    H5Sclose(mspace);
    if(cspace>0)    H5Sclose(cspace);

    return retval;

    failed:     /* Error target */

    retval = -1;
    goto out;

}

/* ------- Support functions ------- */


/*  Normalize a pair of file and memory dataspaces to get rid of H5S_ALL's.
    The new dataspaces returned via ofspace and omspace must be closed. */
herr_t h5py_resolve_spaces(hid_t dset_id, hid_t ifspace, hid_t imspace,
                           hid_t *ofspace, hid_t *omspace){

    hid_t of_tmp, om_tmp;

    if(ifspace==H5S_ALL){
        of_tmp = H5Dget_space(dset_id);
    } else {
        of_tmp = H5Scopy(ifspace);
    }
    if(of_tmp<0) goto failed;

    if(imspace==H5S_ALL){
        om_tmp = H5Scopy(of_tmp);
    } else {
        om_tmp = H5Scopy(imspace);
    }
    if(om_tmp<0) goto failed;

    *ofspace = of_tmp;
    *omspace = om_tmp;

    return 0;

    failed:

    return -1;
}

void* h5py_setup_buffer(hid_t itype, hid_t otype, hid_t space_id, hsize_t* nl){

    void*       buf = NULL;
    size_t      isize, osize, buflen;
    hssize_t    nelements;

    isize = H5Tget_size(itype);
    if(isize==0) goto failed;

    osize = H5Tget_size(otype);
    if(osize==0) goto failed;

    if(isize>osize){
        buflen = isize;
    } else {
        buflen = osize;
    }

    nelements = H5Sget_select_npoints(space_id);
    if(nelements<0) goto failed;

    buf = malloc(nelements*buflen);
    if(buf==NULL) goto failed;

    *nl = nelements;
    return buf;

    failed:

    free(buf);
    return NULL;
    
}


/*  
    Determine if a type is variable-length (H5T_STRING or H5T_VLEN) or in the
    case of compound or array types, contains one.
*/
htri_t h5py_detect_vlen(hid_t type_id){

    H5T_class_t  typeclass;
    htri_t       retval;

    htri_t  is_vlen;
    hid_t   stype=0;
    int     nmembers;
    int     i;

    typeclass = H5Tget_class(type_id);
    if(typeclass<0) goto failed;

    switch(typeclass){

        case H5T_STRING:
            retval = H5Tis_variable_str(type_id);
            break;

        case H5T_VLEN:
            retval = 1;
            break;

        case H5T_ARRAY:
            stype = H5Tget_super(type_id);
            if(stype<0){
                retval = -1;
                break;
            }
            retval = h5py_detect_vlen(stype);
            break;

        case H5T_COMPOUND:
            nmembers = H5Tget_nmembers(type_id);
            if(nmembers<0){
                retval = -1;
                break;
            }
            for(i=0;i<nmembers;i++){
                stype = H5Tget_member_type(type_id, i);
                if(stype<0){
                    retval = -1;
                    break;
                }
                retval = h5py_detect_vlen(stype);
                if(retval!=0){
                    break;
                }
            }
            break;

        default:
            retval = 0;

    } /* switch */


    out:        /* cleanup */

    if(stype>0)     H5Tclose(stype);
    
    return retval;

    failed:     /* error target */

    retval = -1;
    goto out;

}


/* ------ Implements buffer-to-buffer scatter/gather operations ------- */

typedef struct {
    size_t  i;
    size_t  el_size;
    void*   buf;
} h5py_scatter_t;

herr_t h5py_scatter_cb(void* elem, hid_t type_id, unsigned ndim,
                             const hsize_t *point, void *operator_data){

    h5py_scatter_t* info = (h5py_scatter_t*)operator_data;
   
    memcpy(elem, (info->buf)+((info->i)*(info->el_size)), info->el_size);
    
    info->i++;

    return 0;
}

herr_t h5py_gather_cb(void* elem, hid_t type_id, unsigned ndim,
                             const hsize_t *point, void *operator_data){

    h5py_scatter_t* info = (h5py_scatter_t*)operator_data;
   
    memcpy((info->buf)+((info->i)*(info->el_size)), elem, info->el_size);
    
    info->i++;

    return 0;
}

herr_t h5py_copy(hid_t type_id, hid_t space_id, void* contig_buf, 
                 void* scatter_buf, h5py_copy_t op){

    size_t      el_size;
    hssize_t    nl;
    herr_t      call_result;

    h5py_scatter_t info;
    H5D_operator_t cb;

    el_size = H5Tget_size(type_id);
    if(el_size==0) goto failed;

    nl = H5Sget_select_npoints(space_id);
    if(nl<0) goto failed;

    info.i = 0;
    info.el_size = el_size;
    info.buf = contig_buf;
    
    switch(op){
        case H5PY_SCATTER:
            cb = h5py_scatter_cb;
            break;
        case H5PY_GATHER:
            cb = h5py_gather_cb;
            break;
        default:
            goto failed;
    }

    call_result = H5Diterate(scatter_buf, type_id, space_id, cb, &info);
    if(call_result<0) goto failed;

    return 0;

    failed:

    return -1;
}





    
