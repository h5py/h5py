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
    Implements an LZF filter module for HDF5, using the BSD-licensed library
    by Marc Alexander Lehmann (http://www.goof.com/pcg/marc/liblzf.html).

    No Python-specific code is used.  The filter behaves like the DEFLATE
    filter, in that it is called for every type and space, and returns 0
    if the data cannot be compressed.

    This can be used to generate an .so file to load by HDF5's dynamic filter loading:

    1. h5cc -O2 -fPIC -shlib -shared  lzf/*.c lzf_plugin.c -o libhdf5lzf.so 
    2. copy libhdf5lzf.so to /usr/local/hdf5/lib/plugin or to the corresponding folder defined by HDF5_PLUGIN_PATH environment variable
 
*/
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include "lzf/lzf.h"
#include "H5PLextern.h"

#define H5PY_FILTER_LZF_VERSION 4

#define H5Z_FILTER_LZF      32000


/* Our own versions of H5Epush_sim, as it changed in 1.8 */
#if H5_VERS_MAJOR == 1 && H5_VERS_MINOR < 7

#define PUSH_ERR(func, minor, str)  H5Epush(__FILE__, func, __LINE__, H5E_PLINE, minor, str)
#define H5PY_GET_FILTER H5Pget_filter_by_id

#else

#define PUSH_ERR(func, minor, str)  H5Epush1(__FILE__, func, __LINE__, H5E_PLINE, minor, str)
#define H5PY_GET_FILTER(a,b,c,d,e,f,g) H5Pget_filter_by_id2(a,b,c,d,e,f,g,NULL)

#endif



static size_t H5Z_filter_lzf(unsigned int flags, size_t cd_nelmts,
                const unsigned int *cd_values, size_t nbytes, size_t *buf_size, void **buf);

static herr_t lzf_set_local(hid_t dcpl, hid_t type, hid_t space);

/* This message derives from H5Z */
const H5Z_class2_t H5Z_LZF[1] = {{
    H5Z_CLASS_T_VERS,                /* H5Z_class_t version             */
    H5Z_FILTER_LZF,		     /* Filter id number		*/
    1, 1,                            /* Encoding and decoding enabled   */
    "LZF",			     /* Filter name for debugging	*/
    NULL,                            /* The "can apply" callback        */
    (H5Z_set_local_func_t)(lzf_set_local),                            /* The "set local" callback        */
    (H5Z_func_t)H5Z_filter_lzf,    /* The actual filter function	*/
}};

H5PL_type_t   H5PLget_plugin_type(void) {return H5PL_TYPE_FILTER;}
const void    *H5PLget_plugin_info(void) {return H5Z_LZF;}


static herr_t lzf_set_local(hid_t dcpl, hid_t type, hid_t space){

    int ndims;
    int i;
    herr_t r;

    unsigned int bufsize;
    hsize_t chunkdims[32];

    unsigned int flags;
    size_t nelements = 8;
    unsigned values[] = {0,0,0,0,0,0,0,0};

    r = H5PY_GET_FILTER(dcpl, H5Z_FILTER_LZF, &flags, &nelements, values, 0, NULL);
    if(r<0) return -1;

    if(nelements < 3) nelements = 3;  /* First 3 slots reserved.  If any higher
                                      slots are used, preserve the contents. */

    /* It seems the H5Z_FLAG_REVERSE flag doesn't work here, so we have to be
       careful not to clobber any existing version info */
    if(values[0]==0) values[0] = H5PY_FILTER_LZF_VERSION;
    if(values[1]==0) values[1] = LZF_VERSION;

    ndims = H5Pget_chunk(dcpl, 32, chunkdims);
    if(ndims<0) return -1;
    if(ndims>32){
        PUSH_ERR("lzf_set_local", H5E_CALLBACK, "Chunk rank exceeds limit");
        return -1;
    }

    bufsize = H5Tget_size(type);
    if(bufsize==0) return -1;

    for(i=0;i<ndims;i++){
        bufsize *= chunkdims[i];
    }

    values[2] = bufsize;

#ifdef H5PY_LZF_DEBUG
    fprintf(stderr, "LZF: Computed buffer size %d\n", bufsize);
#endif

    r = H5Pmodify_filter(dcpl, H5Z_FILTER_LZF, flags, nelements, values);
    if(r<0) return -1;
    return 1;
}

static size_t
H5Z_filter_lzf(unsigned int flags, size_t cd_nelmts,
      const unsigned int *cd_values, size_t nbytes,
      size_t *buf_size, void **buf)
{
    void* outbuf = NULL;
    size_t outbuf_size = 0;

    unsigned int status = 0;        /* Return code from lzf routines */

    /* We're compressing */
    if(!(flags & H5Z_FLAG_REVERSE)){

        /* Allocate an output buffer exactly as long as the input data; if
           the result is larger, we simply return 0.  The filter is flagged
           as optional, so HDF5 marks the chunk as uncompressed and
           proceeds.
        */

        outbuf_size = (*buf_size);
        outbuf = malloc(outbuf_size);

        if(outbuf == NULL){
            PUSH_ERR("lzf_filter", H5E_CALLBACK, "Can't allocate compression buffer");
            goto failed;
        }
        status = lzf_compress(*buf, nbytes, outbuf, outbuf_size);

    /* We're decompressing */
    } else {

        if((cd_nelmts>=3)&&(cd_values[2]!=0)){
            outbuf_size = cd_values[2];   /* Precomputed buffer guess */
        }else{
            outbuf_size = (*buf_size);
        }

#ifdef H5PY_LZF_DEBUG
        fprintf(stderr, "Decompress %zu chunk w/buffer %zu\n", nbytes, outbuf_size);
#endif

        while(!status){
            
            free(outbuf);
            outbuf = malloc(outbuf_size);

            if(outbuf == NULL){
                PUSH_ERR("lzf_filter", H5E_CALLBACK, "Can't allocate decompression buffer");
                goto failed;
            }

            status = lzf_decompress(*buf, nbytes, outbuf, outbuf_size);

            if(!status){    /* compression failed */

                if(errno == E2BIG){
                    outbuf_size += (*buf_size);
#ifdef H5PY_LZF_DEBUG
                    fprintf(stderr, "    Too small: %zu\n", outbuf_size);
#endif
                } else if(errno == EINVAL) {

                    PUSH_ERR("lzf_filter", H5E_CALLBACK, "Invalid data for LZF decompression");
                    goto failed;

                } else {
                    PUSH_ERR("lzf_filter", H5E_CALLBACK, "Unknown LZF decompression error");
                    goto failed;
                }

            } /* if !status */

        } /* while !status */

    } /* compressing vs decompressing */

    if(status != 0){

        free(*buf);
        *buf = outbuf;
        *buf_size = outbuf_size;

        return status;  /* Size of compressed/decompressed data */
    } 

    failed:

    free(outbuf);
    return 0;
} /* End filter function */



