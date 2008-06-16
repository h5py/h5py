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

include "std_defs.pxi"
from h5 cimport ObjectID

cdef class AttrID(ObjectID):
    pass

cdef extern from "hdf5.h":

  # --- Attribute operations --------------------------------------------------
  hid_t     H5Acreate(hid_t loc_id, char *name, hid_t type_id, hid_t space_id, hid_t create_plist) except *
  hid_t     H5Aopen_idx(hid_t loc_id, unsigned int idx) except *
  hid_t     H5Aopen_name(hid_t loc_id, char *name) except *
  herr_t    H5Aclose(hid_t attr_id) except *
  herr_t    H5Adelete(hid_t loc_id, char *name) except *

  herr_t    H5Aread(hid_t attr_id, hid_t mem_type_id, void *buf) except *
  herr_t    H5Awrite(hid_t attr_id, hid_t mem_type_id, void *buf  ) except *

  int       H5Aget_num_attrs(hid_t loc_id) except *
  ssize_t   H5Aget_name(hid_t attr_id, size_t buf_size, char *buf) except *
  hid_t     H5Aget_space(hid_t attr_id) except *
  hid_t     H5Aget_type(hid_t attr_id) except *

  ctypedef herr_t (*H5A_operator_t)(hid_t loc_id, char *attr_name, operator_data)
  herr_t    H5Aiterate(hid_t loc_id, unsigned * idx, H5A_operator_t op, op_data) except *






