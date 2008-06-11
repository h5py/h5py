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

cdef extern from "hdf5.h":

  ctypedef enum H5G_link_t:
    H5G_LINK_ERROR      = -1,
    H5G_LINK_HARD       = 0,
    H5G_LINK_SOFT       = 1

  cdef enum H5G_obj_t:
    H5G_UNKNOWN = -1,           # Unknown object type
    H5G_LINK,                   # Object is a symbolic link
    H5G_GROUP,                  # Object is a group
    H5G_DATASET,                # Object is a dataset
    H5G_TYPE,                   # Object is a named data type

  ctypedef struct H5G_stat_t:
    unsigned long fileno[2]
    unsigned long objno[2]
    unsigned int nlink
    H5G_obj_t type              # new in HDF5 1.6
    time_t mtime
    size_t linklen
    #H5O_stat_t ohdr            # Object header information. New in HDF5 1.6

  # --- Group operations ------------------------------------------------------
  hid_t  H5Gcreate(hid_t loc_id, char *name, size_t size_hint ) except *
  hid_t  H5Gopen(hid_t loc_id, char *name ) except *
  herr_t H5Gclose(hid_t group_id) except *
#  herr_t H5Glink (hid_t file_id, H5G_link_t link_type,
#                  char *current_name, char *new_name) except *
  herr_t H5Glink2( hid_t curr_loc_id, char *current_name, 
                   H5G_link_t link_type, hid_t new_loc_id, char *new_name ) except *

  herr_t H5Gunlink (hid_t file_id, char *name) except *
#  herr_t H5Gmove(hid_t loc_id, char *src, char *dst) except *
  herr_t H5Gmove2(hid_t src_loc_id, char *src_name,
                  hid_t dst_loc_id, char *dst_name ) except *
  herr_t H5Gget_num_objs(hid_t loc_id, hsize_t*  num_obj) except *
  int    H5Gget_objname_by_idx(hid_t loc_id, hsize_t idx, char *name, size_t size ) except *
  int    H5Gget_objtype_by_idx(hid_t loc_id, hsize_t idx ) except *

  ctypedef herr_t (*H5G_iterate_t)(hid_t group, char *name, op_data)
  herr_t H5Giterate(hid_t loc_id, char *name, int *idx, H5G_iterate_t operator, operator_data  ) except *
  herr_t H5Gget_objinfo(hid_t loc_id, char* name, int follow_link, H5G_stat_t *statbuf) except *

  herr_t H5Gget_linkval(hid_t loc_id, char *name, size_t size, char *value) except *
  herr_t H5Gset_comment(hid_t loc_id, char *name, char *comment ) except *
  int H5Gget_comment(hid_t loc_id, char *name, size_t bufsize, char *comment ) except *


