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
    Conversion routines designed to support the use of variable-length,
    reference and other types which suffer from the HDF5 type-conversion
    bug.
*/

#include "hdf5.h"

#ifndef H5PY_TYPEPROXY_H
#define H5PY_TYPEPROXY_H

/* Proxy functions for reading and writing datasets and attributes */

typedef enum {
    H5PY_WRITE = 0,
    H5PY_READ
} h5py_rw_t;

herr_t H5PY_dset_rw(hid_t dset, hid_t mtype, hid_t mspace_in, hid_t fspace_in,
                   hid_t xfer_plist, void* buf, h5py_rw_t dir);

herr_t H5PY_attr_rw(hid_t attr, hid_t mtype, void* buf, h5py_rw_t dir);


/*  Copy data back & forth between a contiguous buffer and a dataspace 
    selection.  The dataspace must be a "real" dataspace; the value
    H5S_ALL is not acceptable. */

typedef enum {
    H5PY_SCATTER = 0,
    H5PY_GATHER
} h5py_copy_t;

herr_t h5py_copy(hid_t type_id, hid_t space_id, void* contig_buf, 
                 void* scatter_buf, h5py_copy_t op);


#endif





