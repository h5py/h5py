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

include "defs.pxd"

from h5r cimport Reference, RegionReference, hobj_ref_t, hdset_reg_ref_t

# Minimal interface for Python objects immune to Cython refcounting
cdef extern from "Python.h":
    
    # From Cython declarations
    ctypedef void PyTypeObject
    ctypedef struct PyObject:
        Py_ssize_t ob_refcnt
        PyTypeObject *ob_type

    PyObject* PyString_FromString(char* str) except NULL
    int PyString_CheckExact(PyObject* str) except *
    int PyString_Size(PyObject* obj) except *
    PyObject* PyObject_Str(PyObject* obj) except NULL
    char* PyString_AsString(PyObject* obj) except NULL

    PyObject* Py_None
    void Py_INCREF(PyObject* obj)
    void Py_DECREF(PyObject* obj)
    void Py_XDECREF(PyObject* obj)


