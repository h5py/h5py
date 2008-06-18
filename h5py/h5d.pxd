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
from h5 cimport class ObjectID

cdef class DatasetID(ObjectID):
    pass

from h5t cimport class TypeID, typewrap
from h5s cimport class SpaceID
from h5p cimport class PropID, PropDCID, PropDXID, pdefault
from numpy cimport class ndarray



cdef extern from "hdf5.h":

  # HDF5 layouts
  ctypedef enum H5D_layout_t:
    H5D_LAYOUT_ERROR    = -1,
    H5D_COMPACT         = 0,    # raw data is very small
    H5D_CONTIGUOUS      = 1,    # the default
    H5D_CHUNKED         = 2,    # slow and fancy
    H5D_NLAYOUTS        = 3     # this one must be last!

  ctypedef enum H5D_alloc_time_t:
    H5D_ALLOC_TIME_ERROR	=-1,
    H5D_ALLOC_TIME_DEFAULT  =0,
    H5D_ALLOC_TIME_EARLY	=1,
    H5D_ALLOC_TIME_LATE	    =2,
    H5D_ALLOC_TIME_INCR	    =3

  ctypedef enum H5D_space_status_t:
    H5D_SPACE_STATUS_ERROR	        =-1,
    H5D_SPACE_STATUS_NOT_ALLOCATED	=0,
    H5D_SPACE_STATUS_PART_ALLOCATED	=1,
    H5D_SPACE_STATUS_ALLOCATED		=2

  ctypedef enum H5D_fill_time_t:
    H5D_FILL_TIME_ERROR	=-1,
    H5D_FILL_TIME_ALLOC =0,
    H5D_FILL_TIME_NEVER	=1,
    H5D_FILL_TIME_IFSET	=2

  ctypedef enum H5D_fill_value_t:
    H5D_FILL_VALUE_ERROR        =-1,
    H5D_FILL_VALUE_UNDEFINED    =0,
    H5D_FILL_VALUE_DEFAULT      =1,
    H5D_FILL_VALUE_USER_DEFINED =2


  hid_t     H5Dcreate(hid_t loc, char* name, hid_t type_id, hid_t space_id, 
                        hid_t create_plist_id) except *
  hid_t     H5Dopen(hid_t file_id, char *name) except *
  herr_t    H5Dclose(hid_t dset_id) except *

  hid_t     H5Dget_space(hid_t dset_id) except *
  herr_t    H5Dget_space_status(hid_t dset_id, 
                                H5D_space_status_t *status) except *
  hid_t     H5Dget_type(hid_t dset_id) except *
  hid_t     H5Dget_create_plist(hid_t dataset_id) except *
  
  haddr_t   H5Dget_offset(hid_t dset_id) except *
  hsize_t   H5Dget_storage_size(hid_t dset_id) except? 0

  herr_t    H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id,
                  hid_t file_space_id, hid_t plist_id, void *buf) except *
  herr_t    H5Dwrite(hid_t dset_id, hid_t mem_type, hid_t mem_space, hid_t 
                        file_space, hid_t xfer_plist, void* buf) except *

  herr_t    H5Dextend(hid_t dataset_id, hsize_t *size) except *

  herr_t    H5Dfill(void *fill, hid_t fill_type_id, void *buf, 
                    hid_t buf_type_id, hid_t space_id  ) except *
  herr_t    H5Dvlen_get_buf_size(hid_t dset_id, hid_t type_id, 
                                    hid_t space_id, hsize_t *size) except *
  herr_t    H5Dvlen_reclaim(hid_t type_id, hid_t space_id, 
                            hid_t plist, void *buf) except *
  ctypedef  herr_t (*H5D_operator_t)(void *elem, hid_t type_id, unsigned ndim,
				    hsize_t *point, void *operator_data)
  herr_t    H5Diterate(void *buf, hid_t type_id, hid_t space_id, 
                        H5D_operator_t operator, void* operator_data) except *




