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

    The only public function is (int) register_lzf(void), which passes on
    the result from H5Zregister.
*/

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include "hdf5.h"
#include "lzf/lzf.h"
#include "lzf_filter.h"

/* Max size of compress/decompress buffer */
#define H5PY_LZF_MAX_BUF (100L*1024L*1024L)

#if H5_VERS_MAJOR == 1 && H5_VERS_MINOR < 7

#define H5PY_LZF_16API 1
#define PUSH_ERR(func, minor, str)  H5Epush(__FILE__, func, __LINE__, H5E_PLINE, minor, str)

#else

#define H5PY_LZF_16API 0
#define PUSH_ERR(func, minor, str)  H5Epush1(__FILE__, func, __LINE__, H5E_PLINE, minor, str)

#endif

/* In HDF5, one filter function handles both compression and decompression */
size_t lzf_filter(unsigned flags, size_t cd_nelmts,
		    const unsigned cd_values[], size_t nbytes,
		    size_t *buf_size, void **buf);


/* Try to register the filter, passing on the HDF5 return value */
int register_lzf(void){

    int retval;

#if H5PY_LZF_16API
    H5Z_class_t filter_class = {
        (H5Z_filter_t)(H5PY_FILTER_LZF),
        "lzf",
        NULL,
        NULL,
        (H5Z_func_t)(lzf_filter)
    };
#else
    H5Z_class_t filter_class = {
        H5Z_CLASS_T_VERS,
        (H5Z_filter_t)(H5PY_FILTER_LZF),
        1, 1,
        "lzf",
        NULL,
        NULL,
        (H5Z_func_t)(lzf_filter)
    };
#endif

    retval = H5Zregister(&filter_class);
    if(retval<0){
        PUSH_ERR("register_lzf", H5E_CANTREGISTER, "Can't register LZF filter");
    }
    return retval;
}

/* The filter function */
size_t lzf_filter(unsigned flags, size_t cd_nelmts,
		    const unsigned cd_values[], size_t nbytes,
		    size_t *buf_size, void **buf){

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

        outbuf_size = nbytes;
        outbuf = malloc(outbuf_size);

        if(outbuf == NULL){
            PUSH_ERR("lzf_filter", H5E_CALLBACK, "Can't allocate compression buffer");
            goto failed;
        }

        status = lzf_compress(*buf, nbytes, outbuf, outbuf_size);

    /* We're decompressing */
    } else {

        outbuf_size = (*buf_size);

        while(!status){
        
            free(outbuf);
            outbuf = malloc(outbuf_size);

            if(outbuf == NULL){
                PUSH_ERR("lzf_filter", H5E_CALLBACK, "Can't allocate decompression buffer");
                goto failed;
            }

            status = lzf_decompress(*buf, nbytes, outbuf, outbuf_size);

            /* compression failed */
            if(!status){

                /* Output buffer too small; make it bigger */
                if(errno == E2BIG){
#ifdef H5PY_LZF_DEBUG
                    fprintf(stderr, "LZF filter: Buffer guess too small: %d", outbuf_size);
#endif
                    outbuf_size += (*buf_size);
                    if(outbuf_size > H5PY_LZF_MAX_BUF){
                        PUSH_ERR("lzf_filter", H5E_CALLBACK, "Requested LZF buffer too big");
                        goto failed;
                    }

                /* Horrible internal error (data corruption) */
                } else if(errno == EINVAL) {

                    PUSH_ERR("lzf_filter", H5E_CALLBACK, "Invalid data for LZF decompression");
                    goto failed;

                /* Unknown error */
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













