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

cdef class FileID(ObjectID):
    pass

cdef extern from "hdf5.h":

  # File constants
  int H5F_ACC_TRUNC, H5F_ACC_RDONLY, H5F_ACC_RDWR, H5F_ACC_EXCL
  int H5F_ACC_DEBUG, H5F_ACC_CREAT

  # The difference between a single file and a set of mounted files
  cdef enum H5F_scope_t:
    H5F_SCOPE_LOCAL     = 0,    # specified file handle only
    H5F_SCOPE_GLOBAL    = 1,    # entire virtual file
    H5F_SCOPE_DOWN      = 2     # for internal use only

  cdef enum H5F_close_degree_t:
    H5F_CLOSE_WEAK  = 0,
    H5F_CLOSE_SEMI  = 1,
    H5F_CLOSE_STRONG = 2,
    H5F_CLOSE_DEFAULT = 3

  int H5F_OBJ_FILE
  int H5F_OBJ_DATASET
  int H5F_OBJ_GROUP
  int H5F_OBJ_DATATYPE
  int H5F_OBJ_ATTR
  int H5F_OBJ_ALL
  int H5F_OBJ_LOCAL

  # --- File operations -------------------------------------------------------
  hid_t  H5Fcreate(char *filename, unsigned int flags,
                   hid_t create_plist, hid_t access_plist) except *
  hid_t  H5Fopen(char *name, unsigned flags, hid_t access_id) except *
  herr_t H5Fclose (hid_t file_id) except *
  htri_t H5Fis_hdf5(char *name) except *
  herr_t H5Fflush(hid_t object_id, H5F_scope_t scope) except *

  hid_t     H5Freopen(hid_t file_id) except *
  herr_t    H5Fmount(hid_t loc_id, char *name, hid_t child_id, hid_t plist_id) except *
  herr_t    H5Funmount(hid_t loc_id, char *name) except *
  herr_t    H5Fget_filesize(hid_t file_id, hsize_t *size) except *
  hid_t     H5Fget_create_plist(hid_t file_id  ) except *
  hid_t     H5Fget_access_plist(hid_t file_id)  except *
  hssize_t  H5Fget_freespace(hid_t file_id) except *
  ssize_t   H5Fget_name(hid_t obj_id, char *name, size_t size) except *
  int       H5Fget_obj_count(hid_t file_id, unsigned int types) except *
  int       H5Fget_obj_ids(hid_t file_id, unsigned int types, int max_objs, hid_t *obj_id_list) except *










