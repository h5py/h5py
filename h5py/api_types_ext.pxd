# === Standard C library types and functions ==================================


cdef extern from "stdlib.h":
  ctypedef long size_t
  void *malloc(size_t size)
  void free(void *ptr)

cdef extern from "string.h":
  size_t strlen(char* s)
  char *strchr(char *s, int c)
  char *strcpy(char *dest, char *src)
  char *strncpy(char *dest, char *src, size_t n)
  int strcmp(char *s1, char *s2)
  char *strdup(char *s)
  void *memcpy(void *dest, void *src, size_t n)
  void *memset(void *s, int c, size_t n)

cdef extern from "time.h":
  ctypedef int time_t

IF UNAME_SYSNAME != "Windows":
  cdef extern from "unistd.h":
    ctypedef long ssize_t
ELSE:
  ctypedef long ssize_t

cdef extern from "stdint.h":
  ctypedef signed char int8_t
  ctypedef unsigned char uint8_t
  ctypedef signed int int16_t
  ctypedef unsigned int uint16_t
  ctypedef signed long int int32_t
  ctypedef unsigned long int uint32_t
  ctypedef signed long long int int64_t
  ctypedef signed long long int uint64_t 

# Can't use Cython defs because they keep moving them around
cdef extern from "Python.h":
  ctypedef void PyObject
  ctypedef ssize_t Py_ssize_t

  PyObject* PyErr_Occurred()
  void PyErr_SetString(object type, char *message)
  object PyBytes_FromStringAndSize(char *v, Py_ssize_t len)

# === Compatibility definitions and macros for h5py ===========================

cdef extern from "api_compat.h":

  size_t h5py_size_n64
  size_t h5py_size_n128
  size_t h5py_offset_n64_real
  size_t h5py_offset_n64_imag
  size_t h5py_offset_n128_real
  size_t h5py_offset_n128_imag

cdef extern from "lzf_filter.h":

  int H5PY_FILTER_LZF
  int register_lzf() except *
