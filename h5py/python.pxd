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

# This file is based on code from the PyTables project.  The complete PyTables
# license is available at licenses/pytables.txt, in the distribution root
# directory.

# Some helper routines from the Python API
cdef extern from "Python.h":

  # special types
  ctypedef int Py_ssize_t

  # references
  void Py_INCREF(object)
  void Py_DECREF(object)

  # To release global interpreter lock (GIL) for threading
  void Py_BEGIN_ALLOW_THREADS()
  void Py_END_ALLOW_THREADS()

  # Functions for integers
  object PyInt_FromLong(long)
  long PyInt_AsLong(object)
  object PyLong_FromLongLong(long long)
  long long PyLong_AsLongLong(object)

  # Functions for floating points
  object PyFloat_FromDouble(double)

  # Functions for strings
  object PyString_FromStringAndSize(char *s, int len)
  char *PyString_AsString(object string)
  object PyString_FromString(char *)

  # Functions for lists
  int PyList_Append(object list, object item)
  int PyList_Check(object list_)

  # Functions for tuples
  object PyTuple_New(int)
  int PyTuple_SetItem(object, int, object)
  object PyTuple_GetItem(object, int)
  int PyTuple_Size(object tuple)
  int PyTuple_Check(object tpl)

  # Functions for dicts
  int PyDict_Contains(object p, object key)
  object PyDict_GetItem(object p, object key)

  # Functions for objects
  object PyObject_GetItem(object o, object key)
  int PyObject_SetItem(object o, object key, object v)
  int PyObject_DelItem(object o, object key)
  long PyObject_Length(object o)
  int PyObject_Compare(object o1, object o2)
  int PyObject_AsReadBuffer(object obj, void **buffer, Py_ssize_t *buffer_len)

  # Exception handling (manual)
  ctypedef extern class __builtin__.BaseException [object PyBaseExceptionObject]:
    cdef object dict
    cdef object args
    cdef object message

  void* PyExc_Exception   # Not allowed to declare objects "extern C" (why not?)
  void PyErr_SetString(object type_, char* msg)
  void PyErr_SetNone(object type_)
  void PyErr_SetObject(object type_, object args)
  object PyErr_NewException(char* name, object base, object dict_)





