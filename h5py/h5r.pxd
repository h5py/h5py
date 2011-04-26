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

from defs cimport *

cdef extern from "hdf5.h":

  ctypedef haddr_t hobj_ref_t
  ctypedef unsigned char hdset_reg_ref_t[12]

cdef union ref_u:
    hobj_ref_t         obj_ref
    hdset_reg_ref_t    reg_ref

cdef class Reference:

    cdef ref_u ref
    cdef readonly int typecode
    cdef readonly size_t typesize

cdef class RegionReference(Reference):
    pass

