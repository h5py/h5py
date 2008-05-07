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
#include "utils.h"
#include "hdf5.h"


/* Convert an hsize_t array to a Python tuple of long ints.
   Returns None on failure
*/
PyObject* dims_to_tuple(hsize_t* dims, hsize_t rank) {

    PyObject* tpl;
    PyObject* plong;
    int i;
    tpl = NULL;
    plong = NULL;
    
    tpl = PyTuple_New(rank);
    if(tpl == NULL) goto err;

    for(i=0; i<rank; i++){
        plong = PyLong_FromLong((long) dims[i]);
        if(plong == NULL) goto err;
        PyTuple_SET_ITEM(tpl, i, plong); /* steals reference */
    }
    
    Py_INCREF(tpl);
    return tpl;

    err:
    Py_XDECREF(tpl);
    Py_INCREF(Py_None);
    return Py_None;
}

/* Convert a Python tuple to a malloc'ed hsize_t array 
   Returns NULL on failure
*/
hsize_t* tuple_to_dims(PyObject* tpl){

    int rank;
    hsize_t* dims;
    PyObject* temp;
    int i;
    dims = NULL;
    temp = NULL;

    if(tpl == NULL) goto err;
    if(!PyTuple_Check(tpl)) goto err;

    rank = (int)PyTuple_GET_SIZE(tpl);

    dims = (hsize_t*)malloc(sizeof(hsize_t)*rank);

    for(i=0; i<rank; i++){
        temp = PyTuple_GetItem(tpl, i);
        if(temp == NULL) goto err;
        
        if PyLong_Check(temp)
            dims[i] = (hsize_t)PyLong_AsLong(temp);
        else if PyInt_Check(temp)
            dims[i] = (hsize_t)PyLong_AsLong(temp);
        else if PyFloat_Check(temp)
            dims[i] = (hsize_t)PyFloat_AsDouble(temp);
        else
            goto err;
    }

    return dims;

    err:
      if(dims!=NULL) free(dims);
      return NULL;
}

/* The functions

    - check_numpy_write(PyObject* arr, hid_t dataspace)
    - check_numpy_read(PyObject* arr, hid_t dataspace)

   test whether or not a given array object is suitable for reading or writing.
   If dataspace id is positive, it will be checked for compatibility with
   the array object's shape.

   Return values:
    1:  Can read/write
    0:  Can't read/write
   -1:  Failed to determine (i.e. either the array or the space object is bad)
*/
int check_numpy(PyArrayObject* arr, hid_t space_id, int write){

    int required_flags;
    hsize_t arr_rank;
    hsize_t space_rank;
    hsize_t *space_dims = NULL;
    int i;

    required_flags = NPY_C_CONTIGUOUS | NPY_OWNDATA;
    /* That's not how you spell "writable" */
    if(write) required_flags = required_flags | NPY_WRITEABLE;  

    int retval = 0;  /* Default = not OK */

    if(!(arr->flags & required_flags)) goto out;

    if(space_id > 0){

        arr_rank = arr->nd;
        space_rank = H5Sget_simple_extent_ndims(space_id);

        if(space_rank < 0) goto failed;
        if( arr_rank != space_rank) goto out;

        space_dims = (hsize_t*)malloc(sizeof(hsize_t)*space_rank);
        space_rank = H5Sget_simple_extent_dims(space_id, space_dims, NULL);
        if(space_rank < 0) goto failed;

        for(i=0; i<space_rank; i++){
            if(write){
                if(PyArray_DIM(arr,i) < space_dims[i]) goto out;
            } else {
                if(PyArray_DIM(arr,i) > space_dims[i]) goto out;
            }
        }

    }

    retval = 1;  /* got here == success */

  out:
    if(space_dims != NULL) free(space_dims);
    return retval; 

  failed:
    /* could optionally print an error message */
    if(space_dims != NULL) free(space_dims);
    return -1;
}

int check_numpy_write(PyArrayObject* arr, hid_t space_id){
    return check_numpy(arr, space_id, 1);
}

int check_numpy_read(PyArrayObject* arr, hid_t space_id){
    return check_numpy(arr, space_id, 0);
}


/* Rewritten versions of create_ieee_complex64/128 from Pytables, to support 
   standard array-interface typecodes and variable names for real/imag parts.  
   Also removed unneeded datatype copying.
   Both return -1 on failure.
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
  else
    goto err;

  retval = H5Tinsert(complex_id, real_name, HOFFSET(npy_complex64, real), float_id);
  if(retval<0) goto err;

  retval = H5Tinsert(complex_id, img_name, HOFFSET(npy_complex64, imag), float_id);
  if(retval<0) goto err;

  return complex_id;

  err:
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
  else
    goto err;

  retval = H5Tinsert(complex_id, real_name, HOFFSET(npy_complex128, real), float_id);
  if(retval<0) goto err;

  retval = H5Tinsert(complex_id, img_name, HOFFSET(npy_complex128, imag), float_id);
  if(retval<0) goto err;

  return complex_id;

  err:
    if(complex_id > 0)
        H5Tclose(complex_id);
    return -1;
}
