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

/* Wrapper for malloc(size) with the following behavior:
   1. Always returns NULL for emalloc(0)
   2. Raises RuntimeError for emalloc(size<0) and returns NULL
   3. Raises RuntimeError if allocation fails and returns NULL
*/
void* emalloc(size_t size){

    void *retval = NULL;

    if(size==0) return NULL;
    if(size<0){
		PyErr_SetString(PyExc_RuntimeError, "Attempted negative malloc (h5py emalloc)");
    }

    retval = malloc(size);
    if(retval == NULL){
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed (h5py emalloc)");
    }

    return retval;
}

/* Counterpart to emalloc.  For the moment, just a wrapper for free().
*/
void efree(void* ptr){
    free(ptr);
}

/* Convert an hsize_t array to a Python tuple of long ints.
   Returns NULL on failure, and raises an exception (either propagates
   an exception from the conversion, or raises RuntimeError).
*/
PyObject* convert_dims(hsize_t* dims, hsize_t rank) {

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
    
    return tpl;

    err:
    Py_XDECREF(tpl);
    if(!PyErr_Occurred()){
        PyErr_SetString(PyExc_RuntimeError, "Failed to convert hsize_t array to tuple.");
    }
    return NULL;
}

/* Convert a Python tuple to an hsize_t array.  You must allocate
   the array yourself and pass both it and the size to this function.
   Returns 0 on success, -1 on failure and raises an exception.
*/
int convert_tuple(PyObject* tpl, hsize_t *dims, hsize_t rank_in){

    PyObject* temp = NULL;
    int rank;
    int i;

    if(tpl == NULL) goto err;
    if(!PyTuple_Check(tpl)) goto err;

    rank = (int)PyTuple_GET_SIZE(tpl);
    if(rank != rank_in) {
        PyErr_SetString(PyExc_RuntimeError, "Allocated space does not match tuple length");
        goto err;
    }

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

    return 0;

    err:
    if(!PyErr_Occurred()){
        PyErr_SetString(PyExc_ValueError, "Illegal argument (must be a tuple of numbers).");
    }
    return -1;
}

/* The functions

    - check_numpy_write(PyObject* arr, hid_t dataspace)
    - check_numpy_read(PyObject* arr, hid_t dataspace)

   test whether or not a given array object is suitable for reading or writing.
   If dataspace id is positive, it will be checked for compatibility with
   the array object's shape.

   Return values:
    1:  Can read/write
    0: Failed (Python error raised.)
*/
int check_numpy(PyArrayObject* arr, hid_t space_id, int write){

    int required_flags;
    hsize_t arr_rank;
    hsize_t space_rank;
    hsize_t *space_dims = NULL;
    int i;

    /* Validate array flags */

    if(write){
        if(!(arr->flags & (NPY_C_CONTIGUOUS | NPY_OWNDATA | NPY_WRITEABLE))){
            PyErr_SetString(PyExc_ValueError, "Array must be writable, C-contiguous and own its data.");
            goto failed;
        } 
    } else {
        if(!(arr->flags & (NPY_C_CONTIGUOUS | NPY_OWNDATA))){
            PyErr_SetString(PyExc_ValueError, "Array must be C-contiguous and own its data.");
            goto failed;
        }
    }

    /* Validate dataspace compatibility, if it's provided. */

    if(space_id > 0){

        arr_rank = arr->nd;
        space_rank = H5Sget_simple_extent_ndims(space_id);
        if(space_rank < 0) goto failed;

        if( arr_rank != space_rank){
            PyErr_SetString(PyExc_ValueError, "Numpy array rank must match dataspace rank.");
            goto failed;
        }

        space_dims = (hsize_t*)malloc(sizeof(hsize_t)*space_rank);
        space_rank = H5Sget_simple_extent_dims(space_id, space_dims, NULL);
        if(space_rank < 0) goto failed;

        for(i=0; i<space_rank; i++){
            if(write){
                if(PyArray_DIM(arr,i) < space_dims[i]){
                    PyErr_SetString(PyExc_ValueError, "Array dimensions incompatible with dataspace.");
                    goto failed;
                }
            } else {
                if(PyArray_DIM(arr,i) > space_dims[i]) {
                    PyErr_SetString(PyExc_ValueError, "Array dimensions incompatible with dataspace.");
                    goto failed;
                }
            } /* if(write) */
        } /* for */
    } /* if(space_id > 0) */

  free(space_dims);
  return 1;

  failed:
    free(space_dims);
    if(!PyErr_Occurred()){
        PyErr_SetString(PyExc_ValueError, "Numpy array is incompatible.");
    }
    return 0;
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
   Both return -1 on failure, and raise Python exceptions.
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
