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
    Implements low-level infrastructure for vlen and enum types
*/

#include "hdf5.h"

#ifndef H5PY_TYPECONV_H
#define H5PY_TYPECONV_H

/* Register all new conversion functions */
int h5py_register_conv(void);

/* Return the canonical Python object pointer type */
hid_t h5py_object_type(void);

#endif
