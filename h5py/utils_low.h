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
   This file contains code based on utils.c from the PyTables project.  The
   complete PyTables license is available under licenses/pytables.txt in the
   distribution root directory.
*/

#ifndef H5PY_UTILS_LOW
#define H5PY_UTILS_LOW

#include "Python.h"
#include "hdf5.h"
#include "numpy/arrayobject.h"

hid_t create_ieee_complex64(const char byteorder, const char* real_name, const char* img_name);
hid_t create_ieee_complex128(const char byteorder, const char* real_name, const char* img_name);

int check_numpy_read(PyArrayObject* arr, hid_t space_id);
int check_numpy_write(PyArrayObject* arr, hid_t space_id);

void* emalloc(size_t size);
void efree(void* ptr);

#endif

