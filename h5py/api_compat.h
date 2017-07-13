/***** Preamble block *********************************************************
*
* This file is part of h5py, a Python interface to the HDF5 library.
*
* http://www.h5py.org
*
* Copyright 2008-2013 Andrew Collette and contributors
*
* License:  Standard 3-clause BSD; see "license.txt" for full license terms
*           and contributor agreement.
*
****** End preamble block ****************************************************/

/* Contains compatibility macros and definitions for use by Cython code */

#ifndef H5PY_COMPAT
#define H5PY_COMPAT

#if defined(MPI_VERSION) && (MPI_VERSION < 3) && !defined(PyMPI_HAVE_MPI_Message)
typedef void *PyMPI_MPI_Message;
#define MPI_Message PyMPI_MPI_Message
#endif

#include <stddef.h>
#include "Python.h"
#include "numpy/arrayobject.h"
#include "hdf5.h"

/* The HOFFSET macro can't be used from Cython. */

#define h5py_size_n64 (sizeof(npy_complex64))
#define h5py_size_n128 (sizeof(npy_complex128))

#ifdef NPY_COMPLEX256
#define h5py_size_n256 (sizeof(npy_complex256))
#endif

#define h5py_offset_n64_real (HOFFSET(npy_complex64, real))
#define h5py_offset_n64_imag (HOFFSET(npy_complex64, imag))
#define h5py_offset_n128_real (HOFFSET(npy_complex128, real))
#define h5py_offset_n128_imag (HOFFSET(npy_complex128, imag))

#ifdef NPY_COMPLEX256
#define h5py_offset_n256_real (HOFFSET(npy_complex256, real))
#define h5py_offset_n256_imag (HOFFSET(npy_complex256, imag))
#endif

#endif
