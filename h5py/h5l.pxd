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

include "std_defs.pxi"

from h5 cimport ObjectID

# Public classes
cdef class LinkProxy(ObjectID):
  pass

cdef extern from "hdf5.h":

  # TODO: put both versions in h5t.pxd
  ctypedef enum H5T_cset_t:
    H5T_CSET_ERROR       = -1,  #
    H5T_CSET_ASCII       = 0,   # US ASCII
    H5T_CSET_UTF8        = 1,   # UTF-8 Unicode encoding

  unsigned int H5L_MAX_LINK_NAME_LEN #  ((uint32_t) (-1)) (4GB - 1)

  # Link class types.
  # * Values less than 64 are reserved for the HDF5 library's internal use.
  # * Values 64 to 255 are for "user-defined" link class types; these types are
  # * defined by HDF5 but their behavior can be overridden by users.
  # * Users who want to create new classes of links should contact the HDF5
  # * development team at hdfhelp@ncsa.uiuc.edu .
  # * These values can never change because they appear in HDF5 files. 
  # 
  ctypedef enum H5L_type_t:
    H5L_TYPE_ERROR = (-1),      #  Invalid link type id         
    H5L_TYPE_HARD = 0,          #  Hard link id                 
    H5L_TYPE_SOFT = 1,          #  Soft link id                 
    H5L_TYPE_EXTERNAL = 64,     #  External link id             
    H5L_TYPE_MAX = 255          #  Maximum link type id         

  #  Information struct for link (for H5Lget_info/H5Lget_info_by_idx)
  cdef union _add_u:
    haddr_t address   #  Address hard link points to    
    size_t val_size   #  Size of a soft link or UD link value 

  ctypedef struct H5L_info_t:
    H5L_type_t  type            #  Type of link                   
    hbool_t     corder_valid    #  Indicate if creation order is valid 
    int64_t     corder          #  Creation order                 
    H5T_cset_t  cset            #  Character set of link name     
    _add_u u

  #  Prototype for H5Literate/H5Literate_by_name() operator 
  ctypedef herr_t (*H5L_iterate_t) (hid_t group, char *name, H5L_info_t *info,
                    void *op_data)

  ctypedef enum H5_index_t:
    H5_INDEX_NAME,
    H5_INDEX_CRT_ORDER

  ctypedef enum H5_iter_order_t:
     H5_ITER_INC,      # Increasing order
     H5_ITER_DEC,     # Decreasing order
     H5_ITER_NATIVE  # Fastest available order

 # API

  herr_t H5Lmove(hid_t src_loc, char *src_name, hid_t dst_loc,
    char *dst_name, hid_t lcpl_id, hid_t lapl_id) except *

  herr_t H5Lcopy(hid_t src_loc, char *src_name, hid_t dst_loc,
    char *dst_name, hid_t lcpl_id, hid_t lapl_id) except *

  herr_t H5Lcreate_hard(hid_t cur_loc, char *cur_name,
    hid_t dst_loc, char *dst_name, hid_t lcpl_id, hid_t lapl_id) except *

  herr_t H5Lcreate_soft(char *link_target, hid_t link_loc_id,
    char *link_name, hid_t lcpl_id, hid_t lapl_id) except *

  herr_t H5Ldelete(hid_t loc_id, char *name, hid_t lapl_id) except *

  herr_t H5Ldelete_by_idx(hid_t loc_id, char *group_name,
    H5_index_t idx_type, H5_iter_order_t order, hsize_t n, hid_t lapl_id) except *

  herr_t H5Lget_val(hid_t loc_id, char *name, void *bufout,
    size_t size, hid_t lapl_id) except *

  herr_t H5Lget_val_by_idx(hid_t loc_id, char *group_name,
    H5_index_t idx_type, H5_iter_order_t order, hsize_t n,
    void *bufout, size_t size, hid_t lapl_id) except *

  htri_t H5Lexists(hid_t loc_id, char *name, hid_t lapl_id) except *

  herr_t H5Lget_info(hid_t loc_id, char *name,
    H5L_info_t *linfo, hid_t lapl_id) except *

  herr_t H5Lget_info_by_idx(hid_t loc_id, char *group_name,
    H5_index_t idx_type, H5_iter_order_t order, hsize_t n,
    H5L_info_t *linfo, hid_t lapl_id) except *

  ssize_t H5Lget_name_by_idx(hid_t loc_id, char *group_name,
    H5_index_t idx_type, H5_iter_order_t order, hsize_t n,
    char *name, size_t size, hid_t lapl_id) except *

  herr_t H5Literate(hid_t grp_id, H5_index_t idx_type,
    H5_iter_order_t order, hsize_t *idx, H5L_iterate_t op, void *op_data) except *

  herr_t H5Literate_by_name(hid_t loc_id, char *group_name,
    H5_index_t idx_type, H5_iter_order_t order, hsize_t *idx,
    H5L_iterate_t op, void *op_data, hid_t lapl_id) except *

  herr_t H5Lvisit(hid_t grp_id, H5_index_t idx_type, H5_iter_order_t order,
    H5L_iterate_t op, void *op_data) except *

  herr_t H5Lvisit_by_name(hid_t loc_id, char *group_name,
    H5_index_t idx_type, H5_iter_order_t order, H5L_iterate_t op,
    void *op_data, hid_t lapl_id) except *



