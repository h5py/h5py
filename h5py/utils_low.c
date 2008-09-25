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
   Utilities which are difficult or impossible to implement in pure Pyrex, 
   such as functions requiring HDF5 C macros.

   This file contains code based on utils.c from the PyTables project.  The
   complete PyTables license is available under licenses/pytables.txt in the
   distribution root directory.
*/

#include "Python.h"
#include "numpy/arrayobject.h"
#include "utils_low.h"
#include "hdf5.h"
#include "pythread.h"


/* Rewritten versions of create_ieee_complex64/128 from Pytables, to support 
   standard array-interface typecodes and variable names for real/imag parts.  
   Also removed unneeded datatype copying.
   Both return -1 on failure, and raise Python exceptions.

   These must be written in C as they use the HOFFSET macro.
*/
hid_t create_ieee_complex64(const char byteorder, const char* real_name, const char* img_name) {
  hid_t float_id = -1;
  hid_t complex_id = -1;
  herr_t retval = -1;

  complex_id = H5Tcreate(H5T_COMPOUND, sizeof(npy_complex64));
  if(complex_id < 0) goto err;

  if (byteorder == '<')
    float_id = H5T_IEEE_F32LE;
  else if (byteorder == '>')
    float_id = H5T_IEEE_F32BE;
  else if (byteorder == '=' || byteorder == '|')
    float_id = H5T_NATIVE_FLOAT;
  else {
    PyErr_SetString(PyExc_ValueError, "Byte order must be one of <, > or |");
    goto err;
  }

  retval = H5Tinsert(complex_id, real_name, HOFFSET(npy_complex64, real), float_id);
  if(retval<0) goto err;

  retval = H5Tinsert(complex_id, img_name, HOFFSET(npy_complex64, imag), float_id);
  if(retval<0) goto err;

  return complex_id;

  err:
    if(!PyErr_Occurred()){
        PyErr_SetString(PyExc_RuntimeError, "Failed to propagate exception at create_ieee_complex64.");
    }
    if(complex_id > 0)
        H5Tclose(complex_id);
    return -1;
}

hid_t create_ieee_complex128(const char byteorder, const char* real_name, const char* img_name) {
  hid_t float_id = -1;
  hid_t complex_id = -1;
  herr_t retval = -1;

  complex_id = H5Tcreate(H5T_COMPOUND, sizeof(npy_complex128));
  if(complex_id < 0) goto err;

  if (byteorder == '<')
    float_id = H5T_IEEE_F64LE;
  else if (byteorder == '>')
    float_id = H5T_IEEE_F64BE;
  else if (byteorder == '=' || byteorder == '|')
    float_id = H5T_NATIVE_DOUBLE;
  else {
    PyErr_SetString(PyExc_ValueError, "Byte order must be one of <, > or |");
    goto err;
  }

  retval = H5Tinsert(complex_id, real_name, HOFFSET(npy_complex128, real), float_id);
  if(retval<0) goto err;

  retval = H5Tinsert(complex_id, img_name, HOFFSET(npy_complex128, imag), float_id);
  if(retval<0) goto err;

  return complex_id;

  err:
    if(!PyErr_Occurred()){
        PyErr_SetString(PyExc_RuntimeError, "Failed to propagate exception at create_ieee_complex128.");
    }
    if(complex_id > 0)
        H5Tclose(complex_id);
    return -1;
}











