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

from h5 cimport herr_t, htri_t

cdef extern from "hdf5.h":

    ctypedef int H5Z_filter_t

    int H5Z_FILTER_ERROR
    int H5Z_FILTER_NONE
    int H5Z_FILTER_ALL
    int H5Z_FILTER_DEFLATE
    int H5Z_FILTER_SHUFFLE 
    int H5Z_FILTER_FLETCHER32
    int H5Z_FILTER_SZIP
    int H5Z_FILTER_RESERVED
    int H5Z_FILTER_MAX
    int H5Z_MAX_NFILTERS

    int H5Z_FLAG_DEFMASK
    int H5Z_FLAG_MANDATORY
    int H5Z_FLAG_OPTIONAL

    int H5Z_FLAG_INVMASK
    int H5Z_FLAG_REVERSE
    int H5Z_FLAG_SKIP_EDC

    int H5_SZIP_ALLOW_K13_OPTION_MASK   #1
    int H5_SZIP_CHIP_OPTION_MASK        #2
    int H5_SZIP_EC_OPTION_MASK          #4
    int H5_SZIP_NN_OPTION_MASK          #32
    int H5_SZIP_MAX_PIXELS_PER_BLOCK    #32

    int H5Z_FILTER_CONFIG_ENCODE_ENABLED #(0x0001)
    int H5Z_FILTER_CONFIG_DECODE_ENABLED #(0x0002)

    cdef enum H5Z_EDC_t:
        H5Z_ERROR_EDC       = -1,
        H5Z_DISABLE_EDC     = 0,
        H5Z_ENABLE_EDC      = 1,
        H5Z_NO_EDC          = 2 

    # --- Filter API ----------------------------------------------------------
    htri_t H5Zfilter_avail(H5Z_filter_t id_) except *
    herr_t H5Zget_filter_info(H5Z_filter_t filter_, unsigned int *filter_config_flags) except *







