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

#include "Python.h"
#include "hdf5.h"
#include "numpy/arrayobject.h"

hid_t create_ieee_complex64(const char byteorder, const char* real_name, const char* img_name);
hid_t create_ieee_complex128(const char byteorder, const char* real_name, const char* img_name);


hsize_t* tuple_to_dims(PyObject* tpl);
PyObject* dims_to_tuple(hsize_t* dims, hsize_t rank);

int check_numpy_read(PyArrayObject* arr, hid_t space_id);
int check_numpy_write(PyArrayObject* arr, hid_t space_id);

