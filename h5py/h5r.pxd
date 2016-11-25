# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

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

