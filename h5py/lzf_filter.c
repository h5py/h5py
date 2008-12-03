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

/* In HDF5, one filter function handles both compression and decompression */
size_t lzf_filter(unsigned flags, size_t cd_nelmts,
		    const unsigned cd_values[], size_t nbytes,
		    size_t *buf_size, void **buf);


/* Try to register the filter, passing on the HDF5 return value */
int register_lzf(void){

/* Thanks to PyTables for this */
#if H5_VERS_MAJOR == 1 && H5_VERS_MINOR < 7
   /* 1.6.x */
    H5Z_class_t filter_class = {
        (H5Z_filter_t)(H5PY_FILTER_LZF),    /* filter_id */
        "lzf",                         /* comment */
        NULL,                          /* can_apply_func */
        NULL,                          /* set_local_func */
        (H5Z_func_t)(lzf_filter)      /* filter_func */
    };
#else
   /* 1.7.x */
    H5Z_class_t filter_class = {
        H5Z_CLASS_T_VERS,             /* H5Z_class_t version */
        (H5Z_filter_t)(H5PY_FILTER_LZF),   /* filter_id */
        1, 1,                         /* Encoding and decoding enabled */
        "lzf",	 		  /* comment */
        NULL,                         /* can_apply_func */
        NULL,                         /* set_local_func */
        (H5Z_func_t)(lzf_filter)     /* filter_func */
    };
#endif /* if H5_VERSION < "1.7" */

    return H5Zregister(&filter_class);
}

#define H5PY_LZF_MAX_BUF (100L*1024L*1024L) /* 100MB chunks are outrageous */

static size_t historical_buf_size = 0;

/* The filter function */
size_t lzf_filter(unsigned flags, size_t cd_nelmts,
		    const unsigned cd_values[], size_t nbytes,
		    size_t *buf_size, void **buf){

    void* outbuf = NULL;
    size_t outbuf_size = 0;
    unsigned int status = 0;        /* Return code from lzf routines */


    /* If we're compressing */
    if(!(flags & H5Z_FLAG_REVERSE)){

        /* Allocate an output buffer exactly as long as the input data; if
           the result is larger, we simply return 0.
        */
        outbuf_size = nbytes;
        outbuf = malloc(outbuf_size);

        status = lzf_compress(*buf, nbytes, outbuf, outbuf_size);

    /* If we're decompressing */
    } else {

        /* Initialize to our last guess */
        if(historical_buf_size == 0){
            historical_buf_size = *buf_size;
        }
        outbuf_size = historical_buf_size;
        
        while(!status){
        
            free(outbuf);
            outbuf = malloc(outbuf_size);

            status = lzf_decompress(*buf, nbytes, outbuf, outbuf_size);

            /* compression failed */
            if(!status){

                /* Output buffer too small */
                if(errno == E2BIG){
                    outbuf_size += (*buf_size);
                    if(outbuf_size > H5PY_LZF_MAX_BUF){
                        fprintf(stderr, "Can't allocate buffer for LZF decompression");
                        goto failed;
                    }
                    historical_buf_size = outbuf_size;

                /* Horrible internal error */
                } else if(errno == EINVAL) {
                    fprintf(stderr, "LZF decompression error");
                    goto failed;

                /* Unknown error */
                } else {
                    fprintf(stderr, "Unspecified LZF error %d", errno);
                    goto failed;
                }

            } /* if !status */

        } /* while !status */

    } /* if decompressing */
    

    /* If compression/decompression successful, swap buffers */
    if(status){

        free(*buf);
        *buf = outbuf;
        *buf_size = outbuf_size;

        return status;  /* Size of compressed/decompressed data */
    } 

    failed:
        /* Could put a Python exception call here */
        free(outbuf);
        return 0;

} /* End filter function */













