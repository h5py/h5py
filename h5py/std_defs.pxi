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

# "Boilerplate" definitions which are used in every .pxd file.  Also includes
# the dynamically-generated config.pxi interface file.  A companion file
# "std_code.pxi" goes in every .pyx file.

include "config.pxi"

# === Standard C functions and definitions ===

cdef extern from "stdlib.h":
  ctypedef long size_t
  void *malloc(size_t size)
  void free(void *ptr)

cdef extern from "string.h":
  char *strchr(char *s, int c)
  char *strcpy(char *dest, char *src)
  char *strncpy(char *dest, char *src, size_t n)
  int strcmp(char *s1, char *s2)
  char *strdup(char *s)
  void *memcpy(void *dest, void *src, size_t n)

cdef extern from "time.h":
  ctypedef int time_t

cdef extern from "unistd.h":
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

# === HDF5 types ===

cdef extern from "hdf5.h":

  ctypedef int hid_t  # In H5Ipublic.h
  ctypedef int hbool_t
  ctypedef int herr_t
  ctypedef int htri_t
  # hsize_t should be unsigned, but Windows platform does not support
  # such an unsigned long long type.
  ctypedef long long hsize_t
  ctypedef signed long long hssize_t
  ctypedef signed long long haddr_t  # I suppose this must be signed as well...












