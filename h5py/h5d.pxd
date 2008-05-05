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

from defs_c cimport size_t, time_t
from h5 cimport hid_t, hbool_t, herr_t, htri_t, hsize_t, hssize_t, hvl_t

cdef extern from "hdf5.h":

  # HDF5 layouts
  cdef enum H5D_layout_t:
    H5D_LAYOUT_ERROR    = -1,
    H5D_COMPACT         = 0,    # raw data is very small
    H5D_CONTIGUOUS      = 1,    # the default
    H5D_CHUNKED         = 2,    # slow and fancy
    H5D_NLAYOUTS        = 3     # this one must be last!

  cdef enum H5D_alloc_time_t:
    H5D_ALLOC_TIME_ERROR	=-1,
    H5D_ALLOC_TIME_DEFAULT  =0,
    H5D_ALLOC_TIME_EARLY	=1,
    H5D_ALLOC_TIME_LATE	    =2,
    H5D_ALLOC_TIME_INCR	    =3

  cdef enum H5D_space_status_t:
    H5D_SPACE_STATUS_ERROR	        =-1,
    H5D_SPACE_STATUS_NOT_ALLOCATED	=0,
    H5D_SPACE_STATUS_PART_ALLOCATED	=1,
    H5D_SPACE_STATUS_ALLOCATED		=2

  cdef enum H5D_fill_time_t:
    H5D_FILL_TIME_ERROR	=-1,
    H5D_FILL_TIME_ALLOC =0,
    H5D_FILL_TIME_NEVER	=1,
    H5D_FILL_TIME_IFSET	=2

  cdef enum H5D_fill_value_t:
    H5D_FILL_VALUE_ERROR        =-1,
    H5D_FILL_VALUE_UNDEFINED    =0,
    H5D_FILL_VALUE_DEFAULT      =1,
    H5D_FILL_VALUE_USER_DEFINED =2


  # --- Dataset operations ----------------------------------------------------
  hid_t     H5Dcreate(hid_t loc, char* name, hid_t type_id, hid_t space_id, hid_t create_plist_id)
  hid_t     H5Dopen(hid_t file_id, char *name)
  herr_t    H5Dclose(hid_t dset_id)

  hid_t     H5Dget_space(hid_t dset_id)
  hid_t     H5Dget_type(hid_t dset_id)
  hid_t     H5Dget_create_plist(hid_t dataset_id)

  herr_t    H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id,
                  hid_t file_space_id, hid_t plist_id, void *buf)
  herr_t    H5Dwrite(hid_t dset_id, hid_t mem_type, hid_t mem_space, hid_t file_space, hid_t xfer_plist, void* buf)

  herr_t H5Dvlen_reclaim(hid_t type_id, hid_t space_id, hid_t plist_id,
                         void *buf)

