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

# This file provides all external libraries for h5py.

# Originally each HDF5 subsection was attached to its own .pxd file, but this
# proved much too complicated as the definitions are interdependent.

# This file contains code or comments from the HDF5 library, as well as some
# PyTables definitions.  Licenses for both these packages are located in 
# the "licenses" folder in the distribution root directory.

include "config.pxi"  # Needed for H5PY_*API defines


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

# === Compatibility definitions and macros for h5py ===========================

cdef extern from "compat.h":

  size_t h5py_size_n64
  size_t h5py_size_n128
  size_t h5py_offset_n64_real
  size_t h5py_offset_n64_imag
  size_t h5py_offset_n128_real
  size_t h5py_offset_n128_imag

cdef extern from "lzf_filter.h":

  int H5PY_FILTER_LZF
  int register_lzf() except *

# === H5 - Common definitions and library functions ===========================

cdef extern from "hdf5.h":

  # Basic types
  ctypedef int hid_t
  ctypedef int hbool_t
  ctypedef int herr_t
  ctypedef int htri_t
  ctypedef long long hsize_t
  ctypedef signed long long hssize_t
  ctypedef signed long long haddr_t

  ctypedef struct hvl_t:
    size_t len                 # Length of VL data (in base type units)
    void *p                    # Pointer to VL data

  int HADDR_UNDEF

  # H5 API
  herr_t H5open() except *
  herr_t H5close() except *

  herr_t H5get_libversion(unsigned *majnum, unsigned *minnum,
                          unsigned *relnum) except *

  # New in 1.8.X
  IF H5PY_18API:

    ctypedef enum H5_iter_order_t:
      H5_ITER_UNKNOWN = -1,       # Unknown order
      H5_ITER_INC,                # Increasing order
      H5_ITER_DEC,                # Decreasing order
      H5_ITER_NATIVE,             # No particular order, whatever is fastest
      H5_ITER_N                   # Number of iteration orders

    ctypedef enum H5_index_t:
      H5_INDEX_UNKNOWN = -1,      # Unknown index type     
      H5_INDEX_NAME,              # Index on names      
      H5_INDEX_CRT_ORDER,         # Index on creation order    
      H5_INDEX_N                  # Number of indices defined    


# === H5D - Dataset API =======================================================

cdef extern from "hdf5.h":

  ctypedef enum H5D_layout_t:
    H5D_LAYOUT_ERROR    = -1,
    H5D_COMPACT         = 0,
    H5D_CONTIGUOUS      = 1,
    H5D_CHUNKED         = 2,
    H5D_NLAYOUTS        = 3

  ctypedef enum H5D_alloc_time_t:
    H5D_ALLOC_TIME_ERROR    =-1,
    H5D_ALLOC_TIME_DEFAULT  =0,
    H5D_ALLOC_TIME_EARLY    =1,
    H5D_ALLOC_TIME_LATE        =2,
    H5D_ALLOC_TIME_INCR        =3

  ctypedef enum H5D_space_status_t:
    H5D_SPACE_STATUS_ERROR            =-1,
    H5D_SPACE_STATUS_NOT_ALLOCATED    =0,
    H5D_SPACE_STATUS_PART_ALLOCATED    =1,
    H5D_SPACE_STATUS_ALLOCATED        =2

  ctypedef enum H5D_fill_time_t:
    H5D_FILL_TIME_ERROR    =-1,
    H5D_FILL_TIME_ALLOC =0,
    H5D_FILL_TIME_NEVER    =1,
    H5D_FILL_TIME_IFSET    =2

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

  # These must have their return values checked manually, in order to
  # allow the GIL to be released during reading and writing.
  herr_t    H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id,
                  hid_t file_space_id, hid_t plist_id, void *buf) nogil
  herr_t    H5Dwrite(hid_t dset_id, hid_t mem_type, hid_t mem_space, hid_t 
                        file_space, hid_t xfer_plist, void* buf) nogil

  herr_t    H5Dextend(hid_t dataset_id, hsize_t *size) except *

  herr_t    H5Dfill(void *fill, hid_t fill_type_id, void *buf, 
                    hid_t buf_type_id, hid_t space_id ) except *
  herr_t    H5Dvlen_get_buf_size(hid_t dset_id, hid_t type_id, 
                                    hid_t space_id, hsize_t *size) except *
  herr_t    H5Dvlen_reclaim(hid_t type_id, hid_t space_id, 
                            hid_t plist, void *buf) except *

  ctypedef  herr_t (*H5D_operator_t)(void *elem, hid_t type_id, unsigned ndim,
                    hsize_t *point, void *operator_data) except -1
  herr_t    H5Diterate(void *buf, hid_t type_id, hid_t space_id, 
                        H5D_operator_t operator, void* operator_data) except *
  herr_t    H5Dset_extent(hid_t dset_id, hsize_t* size)

  IF H5PY_18API:
    hid_t H5Dcreate2(hid_t loc_id, char *name, hid_t type_id, hid_t space_id,
                     hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id) except *
    hid_t H5Dcreate_anon(hid_t file_id, hid_t type_id, hid_t space_id,
                     hid_t plist_id, hid_t dapl_id) except *


# === H5F - File API ==========================================================

cdef extern from "hdf5.h":

  # File constants
  cdef enum:
    H5F_ACC_TRUNC
    H5F_ACC_RDONLY
    H5F_ACC_RDWR
    H5F_ACC_EXCL
    H5F_ACC_DEBUG
    H5F_ACC_CREAT

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
  hid_t     H5Fget_create_plist(hid_t file_id ) except *
  hid_t     H5Fget_access_plist(hid_t file_id)  except *
  hssize_t  H5Fget_freespace(hid_t file_id) except *
  ssize_t   H5Fget_name(hid_t obj_id, char *name, size_t size) except *
  int       H5Fget_obj_count(hid_t file_id, unsigned int types) except *
  int       H5Fget_obj_ids(hid_t file_id, unsigned int types, int max_objs, hid_t *obj_id_list) except *

  IF H5PY_18API:
    herr_t H5Fget_intent(hid_t file_id, unsigned int *intent) except *

# === H5FD - Low-level file descriptor API ====================================

cdef extern from "hdf5.h":

  ctypedef enum H5FD_mem_t:
    H5FD_MEM_NOLIST    = -1,
    H5FD_MEM_DEFAULT    = 0,
    H5FD_MEM_SUPER      = 1,
    H5FD_MEM_BTREE      = 2,
    H5FD_MEM_DRAW       = 3,
    H5FD_MEM_GHEAP      = 4,
    H5FD_MEM_LHEAP      = 5,
    H5FD_MEM_OHDR       = 6,
    H5FD_MEM_NTYPES

  # HDF5 uses a clever scheme wherein these are actually init() calls
  # Hopefully Cython won't have a problem with this.
  # Thankfully they are defined but -1 if unavailable
  hid_t H5FD_CORE
  hid_t H5FD_FAMILY
# hid_t H5FD_GASS  not in 1.8.X
  hid_t H5FD_LOG
  hid_t H5FD_MPIO
  hid_t H5FD_MULTI
  hid_t H5FD_SEC2
  hid_t H5FD_STDIO  

  int H5FD_LOG_LOC_READ   # 0x0001
  int H5FD_LOG_LOC_WRITE  # 0x0002
  int H5FD_LOG_LOC_SEEK   # 0x0004
  int H5FD_LOG_LOC_IO     # (H5FD_LOG_LOC_READ|H5FD_LOG_LOC_WRITE|H5FD_LOG_LOC_SEEK)

  # Flags for tracking number of times each byte is read/written
  int H5FD_LOG_FILE_READ  # 0x0008
  int H5FD_LOG_FILE_WRITE # 0x0010
  int H5FD_LOG_FILE_IO    # (H5FD_LOG_FILE_READ|H5FD_LOG_FILE_WRITE)

  # Flag for tracking "flavor" (type) of information stored at each byte
  int H5FD_LOG_FLAVOR     # 0x0020

  # Flags for tracking total number of reads/writes/seeks
  int H5FD_LOG_NUM_READ   # 0x0040
  int H5FD_LOG_NUM_WRITE  # 0x0080
  int H5FD_LOG_NUM_SEEK   # 0x0100
  int H5FD_LOG_NUM_IO     # (H5FD_LOG_NUM_READ|H5FD_LOG_NUM_WRITE|H5FD_LOG_NUM_SEEK)

  # Flags for tracking time spent in open/read/write/seek/close
  int H5FD_LOG_TIME_OPEN  # 0x0200        # Not implemented yet
  int H5FD_LOG_TIME_READ  # 0x0400        # Not implemented yet
  int H5FD_LOG_TIME_WRITE # 0x0800        # Partially implemented (need to track total time)
  int H5FD_LOG_TIME_SEEK  # 0x1000        # Partially implemented (need to track total time & track time for seeks during reading)
  int H5FD_LOG_TIME_CLOSE # 0x2000        # Fully implemented
  int H5FD_LOG_TIME_IO    # (H5FD_LOG_TIME_OPEN|H5FD_LOG_TIME_READ|H5FD_LOG_TIME_WRITE|H5FD_LOG_TIME_SEEK|H5FD_LOG_TIME_CLOSE)

  # Flag for tracking allocation of space in file
  int H5FD_LOG_ALLOC      # 0x4000
  int H5FD_LOG_ALL        # (H5FD_LOG_ALLOC|H5FD_LOG_TIME_IO|H5FD_LOG_NUM_IO|H5FD_LOG_FLAVOR|H5FD_LOG_FILE_IO|H5FD_LOG_LOC_IO)


# === H5G - Groups API ========================================================

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

  hid_t  H5Gcreate(hid_t loc_id, char *name, size_t size_hint) except *
  hid_t  H5Gopen(hid_t loc_id, char *name) except *
  herr_t H5Gclose(hid_t group_id) except *
  herr_t H5Glink2( hid_t curr_loc_id, char *current_name, 
                   H5G_link_t link_type, hid_t new_loc_id, char *new_name) except *

  herr_t H5Gunlink (hid_t file_id, char *name) except *
  herr_t H5Gmove2(hid_t src_loc_id, char *src_name,
                  hid_t dst_loc_id, char *dst_name) except *
  herr_t H5Gget_num_objs(hid_t loc_id, hsize_t*  num_obj) except *
  int    H5Gget_objname_by_idx(hid_t loc_id, hsize_t idx, char *name, size_t size) except *
  int    H5Gget_objtype_by_idx(hid_t loc_id, hsize_t idx) except *

  ctypedef herr_t (*H5G_iterate_t)(hid_t group, char *name, void* op_data) except 2
  herr_t H5Giterate(hid_t loc_id, char *name, int *idx, H5G_iterate_t operator, void* data) except *
  herr_t H5Gget_objinfo(hid_t loc_id, char* name, int follow_link, H5G_stat_t *statbuf) except *

  herr_t H5Gget_linkval(hid_t loc_id, char *name, size_t size, char *value) except *
  herr_t H5Gset_comment(hid_t loc_id, char *name, char *comment) except *
  int H5Gget_comment(hid_t loc_id, char *name, size_t bufsize, char *comment) except *

  # New extensions in 1.8.X
  IF H5PY_18API:

    ctypedef enum H5G_storage_type_t:
        H5G_STORAGE_TYPE_UNKNOWN = -1,
        H5G_STORAGE_TYPE_SYMBOL_TABLE,
        H5G_STORAGE_TYPE_COMPACT,
        H5G_STORAGE_TYPE_DENSE 
   
    ctypedef struct H5G_info_t:
        H5G_storage_type_t     storage_type
        hsize_t     nlinks
        int64_t     max_corder

    hid_t   H5Gcreate_anon( hid_t loc_id, hid_t gcpl_id, hid_t gapl_id) except *
    hid_t   H5Gcreate2(hid_t loc_id, char *name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id) except *
    hid_t   H5Gopen2( hid_t loc_id, char * name, hid_t gapl_id) except *
    herr_t  H5Gget_info( hid_t group_id, H5G_info_t *group_info) except *
    herr_t  H5Gget_info_by_name( hid_t loc_id, char *group_name, H5G_info_t *group_info, hid_t lapl_id) except *
    hid_t   H5Gget_create_plist(hid_t group_id) except *


# === H5I - Identifier and reflection interface ===============================

cdef extern from "hdf5.h":

  ctypedef enum H5I_type_t:
    H5I_BADID        = -1,  # invalid Group                   
    H5I_FILE        = 1,    # group ID for File objects           
    H5I_GROUP,              # group ID for Group objects           
    H5I_DATATYPE,           # group ID for Datatype objects           
    H5I_DATASPACE,          # group ID for Dataspace objects       
    H5I_DATASET,            # group ID for Dataset objects           
    H5I_ATTR,               # group ID for Attribute objects       
    H5I_REFERENCE,          # group ID for Reference objects       
    H5I_VFL,                # group ID for virtual file layer       
    H5I_GENPROP_CLS,        # group ID for generic property list classes
    H5I_GENPROP_LST,        # group ID for generic property lists      
    H5I_NGROUPS             # number of valid groups, MUST BE LAST!       

  H5I_type_t H5Iget_type(hid_t obj_id) except *
  ssize_t    H5Iget_name( hid_t obj_id, char *name, size_t size) except *
  hid_t      H5Iget_file_id(hid_t obj_id) except *
  int        H5Idec_ref(hid_t obj_id) except *
  int        H5Iget_ref(hid_t obj_id) except *
  int        H5Iinc_ref(hid_t obj_id) except *


# === H5L - Links interface (1.8.X only) ======================================

IF H5PY_18API:

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
                      void *op_data) except 2

    # Links API

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

    herr_t H5Lunpack_elink_val(void *ext_linkval, size_t link_size,
        unsigned *flags, char **filename, char **obj_path) except *

    herr_t H5Lcreate_external(char *file_name, char *obj_name,
        hid_t link_loc_id, char *link_name, hid_t lcpl_id, hid_t lapl_id) except *


# === H5O - General object operations (1.8.X only) ============================

IF H5PY_18API:

  cdef extern from "hdf5.h":

    ctypedef uint32_t H5O_msg_crt_idx_t

    ctypedef enum H5O_type_t:
      H5O_TYPE_UNKNOWN = -1,      # Unknown object type        
      H5O_TYPE_GROUP,             # Object is a group        
      H5O_TYPE_DATASET,           # Object is a dataset        
      H5O_TYPE_NAMED_DATATYPE,    # Object is a named data type    
      H5O_TYPE_NTYPES             # Number of different object types (must be last!) 

    unsigned int H5O_COPY_SHALLOW_HIERARCHY_FLAG    # (0x0001u) Copy only immediate members
    unsigned int H5O_COPY_EXPAND_SOFT_LINK_FLAG     # (0x0002u) Expand soft links into new objects
    unsigned int H5O_COPY_EXPAND_EXT_LINK_FLAG      # (0x0004u) Expand external links into new objects
    unsigned int H5O_COPY_EXPAND_REFERENCE_FLAG     # (0x0008u) Copy objects that are pointed by references
    unsigned int H5O_COPY_WITHOUT_ATTR_FLAG         # (0x0010u) Copy object without copying attributes
    unsigned int H5O_COPY_PRESERVE_NULL_FLAG        # (0x0020u) Copy NULL messages (empty space)
    unsigned int H5O_COPY_ALL                       # (0x003Fu) All object copying flags (for internal checking)

    # --- Components for the H5O_info_t struct ----------------------------------

    ctypedef struct space:
      hsize_t total           #  Total space for storing object header in file 
      hsize_t meta            #  Space within header for object header metadata information 
      hsize_t mesg            #  Space within header for actual message information 
      hsize_t free            #  Free space within object header 

    ctypedef struct mesg:
      unsigned long present   #  Flags to indicate presence of message type in header 
      unsigned long shared    #  Flags to indicate message type is shared in header 

    ctypedef struct hdr:
      unsigned version        #  Version number of header format in file 
      unsigned nmesgs         #  Number of object header messages 
      unsigned nchunks        #  Number of object header chunks 
      unsigned flags          #  Object header status flags 
      space space
      mesg mesg

    ctypedef struct H5_ih_info_t:
      hsize_t     index_size,  # btree and/or list
      hsize_t     heap_size

    cdef struct meta_size:
      H5_ih_info_t   obj,    #        v1/v2 B-tree & local/fractal heap for groups, B-tree for chunked datasets
      H5_ih_info_t   attr    #        v2 B-tree & heap for attributes

    ctypedef struct H5O_info_t:
      unsigned long   fileno      #  File number that object is located in 
      haddr_t         addr        #  Object address in file    
      H5O_type_t      type        #  Basic object type (group, dataset, etc.) 
      unsigned        rc          #  Reference count of object    
      time_t          atime       #  Access time            
      time_t          mtime       #  Modification time        
      time_t          ctime       #  Change time            
      time_t          btime       #  Birth time            
      hsize_t         num_attrs   #  # of attributes attached to object 
      hdr             hdr
      meta_size       meta_size

    # --- H5O API -------------------------------------------------------------

    hid_t H5Oopen(hid_t loc_id, char *name, hid_t lapl_id) except *
    hid_t H5Oopen_by_addr(hid_t loc_id, haddr_t addr) except *
    hid_t H5Oopen_by_idx(hid_t loc_id, char *group_name,
            H5_index_t idx_type, H5_iter_order_t order, hsize_t n, hid_t lapl_id) except *

    herr_t H5Oget_info(hid_t loc_id, H5O_info_t *oinfo) except *
    herr_t H5Oget_info_by_name(hid_t loc_id, char *name, H5O_info_t *oinfo,
              hid_t lapl_id) except *
    herr_t H5Oget_info_by_idx(hid_t loc_id, char *group_name,
            H5_index_t idx_type, H5_iter_order_t order, hsize_t n, H5O_info_t *oinfo,
            hid_t lapl_id) except *

    herr_t H5Olink(hid_t obj_id, hid_t new_loc_id, char *new_name,
            hid_t lcpl_id, hid_t lapl_id) except *
    herr_t H5Ocopy(hid_t src_loc_id, char *src_name, hid_t dst_loc_id,
            char *dst_name, hid_t ocpypl_id, hid_t lcpl_id) except *

    herr_t H5Oincr_refcount(hid_t object_id) except *
    herr_t H5Odecr_refcount(hid_t object_id) except *

    herr_t H5Oset_comment(hid_t obj_id, char *comment) except *
    herr_t H5Oset_comment_by_name(hid_t loc_id, char *name,
            char *comment, hid_t lapl_id) except *
    ssize_t H5Oget_comment(hid_t obj_id, char *comment, size_t bufsize) except *
    ssize_t H5Oget_comment_by_name(hid_t loc_id, char *name,
              char *comment, size_t bufsize, hid_t lapl_id) except *

    ctypedef herr_t (*H5O_iterate_t)(hid_t obj, char *name, H5O_info_t *info,
                      void *op_data) except 2

    herr_t H5Ovisit(hid_t obj_id, H5_index_t idx_type, H5_iter_order_t order,
            H5O_iterate_t op, void *op_data) except *
    herr_t H5Ovisit_by_name(hid_t loc_id, char *obj_name,
            H5_index_t idx_type, H5_iter_order_t order, H5O_iterate_t op,
            void *op_data, hid_t lapl_id) except *

    herr_t H5Oclose(hid_t object_id) except *

# === H5P - Property list API =================================================

cdef extern from "hdf5.h":

  int H5P_DEFAULT

  ctypedef int H5Z_filter_t

  # HDF5 layouts
  ctypedef enum H5D_layout_t:
    H5D_LAYOUT_ERROR    = -1,
    H5D_COMPACT         = 0,    # raw data is very small
    H5D_CONTIGUOUS      = 1,    # the default
    H5D_CHUNKED         = 2,    # slow and fancy
    H5D_NLAYOUTS        = 3     # this one must be last!

  ctypedef enum H5D_alloc_time_t:
    H5D_ALLOC_TIME_ERROR    =-1,
    H5D_ALLOC_TIME_DEFAULT  =0,
    H5D_ALLOC_TIME_EARLY    =1,
    H5D_ALLOC_TIME_LATE        =2,
    H5D_ALLOC_TIME_INCR        =3

  ctypedef enum H5D_space_status_t:
    H5D_SPACE_STATUS_ERROR            =-1,
    H5D_SPACE_STATUS_NOT_ALLOCATED    =0,
    H5D_SPACE_STATUS_PART_ALLOCATED    =1,
    H5D_SPACE_STATUS_ALLOCATED        =2

  ctypedef enum H5D_fill_time_t:
    H5D_FILL_TIME_ERROR    =-1,
    H5D_FILL_TIME_ALLOC =0,
    H5D_FILL_TIME_NEVER    =1,
    H5D_FILL_TIME_IFSET    =2

  ctypedef enum H5D_fill_value_t:
    H5D_FILL_VALUE_ERROR        =-1,
    H5D_FILL_VALUE_UNDEFINED    =0,
    H5D_FILL_VALUE_DEFAULT      =1,
    H5D_FILL_VALUE_USER_DEFINED =2

  cdef enum H5Z_EDC_t:
    H5Z_ERROR_EDC       = -1,
    H5Z_DISABLE_EDC     = 0,
    H5Z_ENABLE_EDC      = 1,
    H5Z_NO_EDC          = 2 

  cdef enum H5F_close_degree_t:
    H5F_CLOSE_WEAK  = 0,
    H5F_CLOSE_SEMI  = 1,
    H5F_CLOSE_STRONG = 2,
    H5F_CLOSE_DEFAULT = 3

  ctypedef enum H5FD_mem_t:
    H5FD_MEM_NOLIST    = -1,
    H5FD_MEM_DEFAULT    = 0,
    H5FD_MEM_SUPER      = 1,
    H5FD_MEM_BTREE      = 2,
    H5FD_MEM_DRAW       = 3,
    H5FD_MEM_GHEAP      = 4,
    H5FD_MEM_LHEAP      = 5,
    H5FD_MEM_OHDR       = 6,
    H5FD_MEM_NTYPES

  # Property list classes
  hid_t H5P_NO_CLASS
  hid_t H5P_FILE_CREATE 
  hid_t H5P_FILE_ACCESS 
  hid_t H5P_DATASET_CREATE 
  hid_t H5P_DATASET_XFER 
  IF H5PY_18API:
    hid_t H5P_OBJECT_CREATE
    hid_t H5P_OBJECT_COPY
    hid_t H5P_LINK_CREATE
    hid_t H5P_LINK_ACCESS
    hid_t H5P_GROUP_CREATE

  # General operations
  hid_t  H5Pcreate(hid_t plist_id) except *
  hid_t  H5Pcopy(hid_t plist_id) except *
  int    H5Pget_class(hid_t plist_id) except *
  herr_t H5Pclose(hid_t plist_id) except *
  htri_t H5Pequal( hid_t id1, hid_t id2 ) except *
  herr_t H5Pclose_class(hid_t id) except *

  # File creation properties
  herr_t H5Pget_version(hid_t plist, unsigned int *super_, unsigned int* freelist, 
                        unsigned int *stab, unsigned int *shhdr) except *
  herr_t H5Pset_userblock(hid_t plist, hsize_t size) except *
  herr_t H5Pget_userblock(hid_t plist, hsize_t * size) except *
  herr_t H5Pset_sizes(hid_t plist, size_t sizeof_addr, size_t sizeof_size) except *
  herr_t H5Pget_sizes(hid_t plist, size_t *sizeof_addr, size_t *sizeof_size) except *
  herr_t H5Pset_sym_k(hid_t plist, unsigned int ik, unsigned int lk) except *
  herr_t H5Pget_sym_k(hid_t plist, unsigned int *ik, unsigned int *lk) except *
  herr_t H5Pset_istore_k(hid_t plist, unsigned int ik) except *
  herr_t H5Pget_istore_k(hid_t plist, unsigned int *ik) except *

  # File access
  herr_t    H5Pset_fclose_degree(hid_t fapl_id, H5F_close_degree_t fc_degree) except *
  herr_t    H5Pget_fclose_degree(hid_t fapl_id, H5F_close_degree_t *fc_degree) except *
  herr_t    H5Pset_fapl_core( hid_t fapl_id, size_t increment, hbool_t backing_store) except *
  herr_t    H5Pget_fapl_core( hid_t fapl_id, size_t *increment, hbool_t *backing_store) except *
  herr_t    H5Pset_fapl_family ( hid_t fapl_id,  hsize_t memb_size, hid_t memb_fapl_id ) except *
  herr_t    H5Pget_fapl_family ( hid_t fapl_id, hsize_t *memb_size, hid_t *memb_fapl_id ) except *
  herr_t    H5Pset_family_offset ( hid_t fapl_id, hsize_t offset) except *
  herr_t    H5Pget_family_offset ( hid_t fapl_id, hsize_t *offset) except *
  herr_t    H5Pset_fapl_log(hid_t fapl_id, char *logfile, unsigned int flags, size_t buf_size) except *
  herr_t    H5Pset_fapl_multi(hid_t fapl_id, H5FD_mem_t *memb_map, hid_t *memb_fapl,
                char **memb_name, haddr_t *memb_addr, hbool_t relax) 
  herr_t H5Pset_cache(hid_t plist_id, int mdc_nelmts, int rdcc_nelmts,
                      size_t rdcc_nbytes, double rdcc_w0) except *
  herr_t H5Pget_cache(hid_t plist_id, int *mdc_nelmts, int *rdcc_nelmts,
                      size_t *rdcc_nbytes, double *rdcc_w0) except *
  herr_t H5Pset_fapl_sec2(hid_t fapl_id) except *
  herr_t H5Pset_fapl_stdio(hid_t fapl_id) except *
  hid_t  H5Pget_driver(hid_t fapl_id) except *

  # Dataset creation properties
  herr_t        H5Pset_layout(hid_t plist, int layout) except *
  H5D_layout_t  H5Pget_layout(hid_t plist) except *
  herr_t        H5Pset_chunk(hid_t plist, int ndims, hsize_t * dim) except *
  int           H5Pget_chunk(hid_t plist, int max_ndims, hsize_t * dims ) except *
  herr_t        H5Pset_deflate( hid_t plist, int level) except *
  herr_t        H5Pset_fill_value(hid_t plist_id, hid_t type_id, void *value ) except *
  herr_t        H5Pget_fill_value(hid_t plist_id, hid_t type_id, void *value ) except *
  herr_t        H5Pfill_value_defined(hid_t plist_id, H5D_fill_value_t *status ) except *
  herr_t        H5Pset_fill_time(hid_t plist_id, H5D_fill_time_t fill_time ) except *
  herr_t        H5Pget_fill_time(hid_t plist_id, H5D_fill_time_t *fill_time ) except *
  herr_t        H5Pset_alloc_time(hid_t plist_id, H5D_alloc_time_t alloc_time ) except *
  herr_t        H5Pget_alloc_time(hid_t plist_id, H5D_alloc_time_t *alloc_time ) except *
  herr_t        H5Pset_filter(hid_t plist, H5Z_filter_t filter, unsigned int flags,
                              size_t cd_nelmts, unsigned int cd_values[] ) except *
  htri_t        H5Pall_filters_avail(hid_t dcpl_id) except *
  int           H5Pget_nfilters(hid_t plist) except *
  H5Z_filter_t  H5Pget_filter(hid_t plist, unsigned int filter_number, 
                              unsigned int *flags, size_t *cd_nelmts, 
                              unsigned int *cd_values, size_t namelen, char name[] ) except *
  herr_t        H5Pget_filter_by_id( hid_t plist_id, H5Z_filter_t filter, 
                                     unsigned int *flags, size_t *cd_nelmts, 
                                     unsigned int cd_values[], size_t namelen, char name[]) except *
  herr_t        H5Pmodify_filter(hid_t plist, H5Z_filter_t filter, unsigned int flags,
                                 size_t cd_nelmts, unsigned int cd_values[] ) except *
  herr_t        H5Premove_filter(hid_t plist, H5Z_filter_t filter ) except *
  herr_t        H5Pset_fletcher32(hid_t plist) except *
  herr_t        H5Pset_shuffle(hid_t plist_id) except *
  herr_t        H5Pset_szip(hid_t plist, unsigned int options_mask, unsigned int pixels_per_block) except *


  # Transfer properties
  herr_t    H5Pset_edc_check(hid_t plist, H5Z_EDC_t check) except *
  H5Z_EDC_t H5Pget_edc_check(hid_t plist) except *

  # Other properties
  herr_t H5Pset_sieve_buf_size(hid_t fapl_id, size_t size) except *
  herr_t H5Pget_sieve_buf_size(hid_t fapl_id, size_t *size) except *
  herr_t H5Pset_fapl_log(hid_t fapl_id, char *logfile,
                         unsigned int flags, size_t buf_size) except *

  # New in 1.8
  IF H5PY_18API:

    herr_t H5Pset_nlinks(hid_t plist_id, size_t nlinks) except *
    herr_t H5Pget_nlinks(hid_t plist_id, size_t *nlinks) except *
    herr_t H5Pset_elink_prefix(hid_t plist_id, char *prefix) except *
    ssize_t H5Pget_elink_prefix(hid_t plist_id, char *prefix, size_t size) except *
    hid_t  H5Pget_elink_fapl(hid_t lapl_id) except *
    herr_t H5Pset_elink_fapl(hid_t lapl_id, hid_t fapl_id) except *

    herr_t H5Pset_create_intermediate_group(hid_t plist_id, unsigned crt_intmd) except *
    herr_t H5Pget_create_intermediate_group(hid_t plist_id, unsigned *crt_intmd) except *

    herr_t H5Pset_copy_object(hid_t plist_id, unsigned crt_intmd) except *
    herr_t H5Pget_copy_object(hid_t plist_id, unsigned *crt_intmd) except *

    herr_t H5Pset_char_encoding(hid_t plist_id, H5T_cset_t encoding) except *
    herr_t H5Pget_char_encoding(hid_t plist_id, H5T_cset_t *encoding) except *

    herr_t H5Pset_local_heap_size_hint(hid_t plist_id, size_t size_hint) except *
    herr_t H5Pget_local_heap_size_hint(hid_t plist_id, size_t *size_hint) except *
    herr_t H5Pset_link_phase_change(hid_t plist_id, unsigned max_compact, unsigned min_dense) except *
    herr_t H5Pget_link_phase_change(hid_t plist_id, unsigned *max_compact , unsigned *min_dense) except *
    herr_t H5Pset_est_link_info(hid_t plist_id, unsigned est_num_entries, unsigned est_name_len) except *
    herr_t H5Pget_est_link_info(hid_t plist_id, unsigned *est_num_entries , unsigned *est_name_len) except *
    herr_t H5Pset_link_creation_order(hid_t plist_id, unsigned crt_order_flags) except *
    herr_t H5Pget_link_creation_order(hid_t plist_id, unsigned *crt_order_flags) except *


# === H5R - Reference API =====================================================

cdef extern from "hdf5.h":

  size_t H5R_DSET_REG_REF_BUF_SIZE
  size_t H5R_OBJ_REF_BUF_SIZE

  ctypedef enum H5R_type_t:
    H5R_BADTYPE = (-1),
    H5R_OBJECT,
    H5R_DATASET_REGION,
    H5R_INTERNAL,
    H5R_MAXTYPE

  herr_t    H5Rcreate(void *ref, hid_t loc_id, char *name, H5R_type_t ref_type, 
                      hid_t space_id) except *
  hid_t     H5Rdereference(hid_t obj_id, H5R_type_t ref_type, void *ref) except *
  hid_t     H5Rget_region(hid_t dataset, H5R_type_t ref_type, void *ref) except *
  H5G_obj_t H5Rget_obj_type(hid_t id, H5R_type_t ref_type, void *ref) except *

  IF H5PY_18API:
    ssize_t H5Rget_name(hid_t loc_id, H5R_type_t ref_type, void *ref, char *name, size_t size) except *
    
# === H5S - Dataspaces ========================================================

cdef extern from "hdf5.h":

  int H5S_ALL, H5S_MAX_RANK
  hsize_t H5S_UNLIMITED

  # Codes for defining selections
  ctypedef enum H5S_seloper_t:
    H5S_SELECT_NOOP      = -1,
    H5S_SELECT_SET       = 0,
    H5S_SELECT_OR,
    H5S_SELECT_AND,
    H5S_SELECT_XOR,
    H5S_SELECT_NOTB,
    H5S_SELECT_NOTA,
    H5S_SELECT_APPEND,
    H5S_SELECT_PREPEND,
    H5S_SELECT_INVALID    # Must be the last one

  ctypedef enum H5S_class_t:
    H5S_NO_CLASS         = -1,  #/*error                                     
    H5S_SCALAR           = 0,   #/*scalar variable                           
    H5S_SIMPLE           = 1,   #/*simple data space                         
    # no longer defined in 1.8
    #H5S_COMPLEX          = 2    #/*complex data space                        

  ctypedef enum H5S_sel_type:
    H5S_SEL_ERROR    = -1,         #Error           
    H5S_SEL_NONE    = 0,        #Nothing selected        
    H5S_SEL_POINTS    = 1,        #Sequence of points selected   
    H5S_SEL_HYPERSLABS  = 2,    #"New-style" hyperslab selection defined   
    H5S_SEL_ALL        = 3,        #Entire extent selected   
    H5S_SEL_N        = 4            #/*THIS MUST BE LAST       


  # Basic operations
  hid_t     H5Screate(H5S_class_t type) except *
  hid_t     H5Scopy(hid_t space_id ) except *
  herr_t    H5Sclose(hid_t space_id) except *

  # Simple dataspace operations
  hid_t     H5Screate_simple(int rank, hsize_t dims[], hsize_t maxdims[]) except *
  htri_t    H5Sis_simple(hid_t space_id) except *
  herr_t    H5Soffset_simple(hid_t space_id, hssize_t *offset ) except *

  int       H5Sget_simple_extent_ndims(hid_t space_id) except *
  int       H5Sget_simple_extent_dims(hid_t space_id, hsize_t dims[], hsize_t maxdims[]) except *
  hssize_t  H5Sget_simple_extent_npoints(hid_t space_id) except *
  H5S_class_t H5Sget_simple_extent_type(hid_t space_id) except *

  # Extents
  herr_t    H5Sextent_copy(hid_t dest_space_id, hid_t source_space_id ) except *
  herr_t    H5Sset_extent_simple(hid_t space_id, int rank, 
                hsize_t *current_size, hsize_t *maximum_size ) except *
  herr_t    H5Sset_extent_none(hid_t space_id) except *

  # Dataspace selection
  H5S_sel_type H5Sget_select_type(hid_t space_id) except *
  hssize_t  H5Sget_select_npoints(hid_t space_id) except *
  herr_t    H5Sget_select_bounds(hid_t space_id, hsize_t *start, hsize_t *end) except *

  herr_t    H5Sselect_all(hid_t space_id) except *
  herr_t    H5Sselect_none(hid_t space_id) except *
  htri_t    H5Sselect_valid(hid_t space_id) except *

  hssize_t  H5Sget_select_elem_npoints(hid_t space_id) except *
  herr_t    H5Sget_select_elem_pointlist(hid_t space_id, hsize_t startpoint, 
                hsize_t numpoints, hsize_t *buf) except *
  herr_t    H5Sselect_elements(hid_t space_id, H5S_seloper_t op, 
                size_t num_elements, hsize_t **coord) except *

  hssize_t  H5Sget_select_hyper_nblocks(hid_t space_id ) except *
  herr_t    H5Sget_select_hyper_blocklist(hid_t space_id, 
                hsize_t startblock, hsize_t numblocks, hsize_t *buf ) except *
  herr_t H5Sselect_hyperslab(hid_t space_id, H5S_seloper_t op,
                             hsize_t start[], hsize_t _stride[],
                             hsize_t count[], hsize_t _block[]) except *

  IF H5PY_18API:
    herr_t  H5Sencode(hid_t obj_id, void *buf, size_t *nalloc)
    hid_t   H5Sdecode(void *buf)


# === H5T - Datatypes =========================================================

cdef extern from "hdf5.h":

  hid_t H5P_DEFAULT

  # --- Enumerated constants --------------------------------------------------

  # Byte orders
  ctypedef enum H5T_order_t:
    H5T_ORDER_ERROR      = -1,  # error
    H5T_ORDER_LE         = 0,   # little endian
    H5T_ORDER_BE         = 1,   # bit endian
    H5T_ORDER_VAX        = 2,   # VAX mixed endian
    H5T_ORDER_NONE       = 3    # no particular order (strings, bits,..)

  # HDF5 signed enums
  ctypedef enum H5T_sign_t:
    H5T_SGN_ERROR        = -1,  # error
    H5T_SGN_NONE         = 0,   # this is an unsigned type
    H5T_SGN_2            = 1,   # two's complement
    H5T_NSGN             = 2    # this must be last!

  ctypedef enum H5T_norm_t:
    H5T_NORM_ERROR       = -1,
    H5T_NORM_IMPLIED     = 0,
    H5T_NORM_MSBSET      = 1,
    H5T_NORM_NONE        = 2

  ctypedef enum H5T_cset_t:
    H5T_CSET_ERROR       = -1,
    H5T_CSET_ASCII       = 0

  ctypedef enum H5T_str_t:
    H5T_STR_ERROR        = -1,
    H5T_STR_NULLTERM     = 0,
    H5T_STR_NULLPAD      = 1,
    H5T_STR_SPACEPAD     = 2

  # Atomic datatype padding
  ctypedef enum H5T_pad_t:
    H5T_PAD_ZERO        = 0,
    H5T_PAD_ONE         = 1,
    H5T_PAD_BACKGROUND  = 2

  # HDF5 type classes
  cdef enum H5T_class_t:
    H5T_NO_CLASS         = -1,  # error
    H5T_INTEGER          = 0,   # integer types
    H5T_FLOAT            = 1,   # floating-point types
    H5T_TIME             = 2,   # date and time types
    H5T_STRING           = 3,   # character string types
    H5T_BITFIELD         = 4,   # bit field types
    H5T_OPAQUE           = 5,   # opaque types
    H5T_COMPOUND         = 6,   # compound types
    H5T_REFERENCE        = 7,   # reference types
    H5T_ENUM             = 8,   # enumeration types
    H5T_VLEN             = 9,   # variable-length types
    H5T_ARRAY            = 10,  # array types
    H5T_NCLASSES                # this must be last

  # Native search direction
  cdef enum H5T_direction_t:
    H5T_DIR_DEFAULT,
    H5T_DIR_ASCEND,
    H5T_DIR_DESCEND

  # For vlen strings
  cdef size_t H5T_VARIABLE

  # --- Predefined datatypes --------------------------------------------------

  cdef enum:
    H5T_NATIVE_B8
    H5T_NATIVE_CHAR
    H5T_NATIVE_SCHAR
    H5T_NATIVE_UCHAR
    H5T_NATIVE_SHORT
    H5T_NATIVE_USHORT
    H5T_NATIVE_INT
    H5T_NATIVE_UINT
    H5T_NATIVE_LONG
    H5T_NATIVE_ULONG
    H5T_NATIVE_LLONG
    H5T_NATIVE_ULLONG
    H5T_NATIVE_FLOAT
    H5T_NATIVE_DOUBLE
    H5T_NATIVE_LDOUBLE

  # "Standard" types
  cdef enum:
    H5T_STD_I8LE
    H5T_STD_I16LE
    H5T_STD_I32LE
    H5T_STD_I64LE
    H5T_STD_U8LE
    H5T_STD_U16LE
    H5T_STD_U32LE
    H5T_STD_U64LE
    H5T_STD_B8LE
    H5T_STD_B16LE
    H5T_STD_B32LE
    H5T_STD_B64LE
    H5T_IEEE_F32LE
    H5T_IEEE_F64LE
    H5T_STD_I8BE
    H5T_STD_I16BE
    H5T_STD_I32BE
    H5T_STD_I64BE
    H5T_STD_U8BE
    H5T_STD_U16BE
    H5T_STD_U32BE
    H5T_STD_U64BE
    H5T_STD_B8BE
    H5T_STD_B16BE
    H5T_STD_B32BE
    H5T_STD_B64BE
    H5T_IEEE_F32BE
    H5T_IEEE_F64BE

  cdef enum:
    H5T_NATIVE_INT8
    H5T_NATIVE_UINT8
    H5T_NATIVE_INT16
    H5T_NATIVE_UINT16
    H5T_NATIVE_INT32
    H5T_NATIVE_UINT32
    H5T_NATIVE_INT64
    H5T_NATIVE_UINT64

  # Unix time types
  cdef enum:
    H5T_UNIX_D32LE
    H5T_UNIX_D64LE
    H5T_UNIX_D32BE
    H5T_UNIX_D64BE

  # String types
  cdef enum:
    H5T_FORTRAN_S1
    H5T_C_S1

  # References
  cdef enum:
    H5T_STD_REF_OBJ
    H5T_STD_REF_DSETREG

  # --- Datatype operations ---------------------------------------------------

  # General operations
  hid_t         H5Tcreate(H5T_class_t type, size_t size) except *
  hid_t         H5Topen(hid_t loc, char* name) except *
  herr_t        H5Tcommit(hid_t loc_id, char* name, hid_t type) except *
  htri_t        H5Tcommitted(hid_t type) except *
  hid_t         H5Tcopy(hid_t type_id) except *
  htri_t        H5Tequal(hid_t type_id1, hid_t type_id2 ) except *
  herr_t        H5Tlock(hid_t type_id) except *
  H5T_class_t   H5Tget_class(hid_t type_id) except *
  size_t        H5Tget_size(hid_t type_id) except? 0
  hid_t         H5Tget_super(hid_t type) except *
  htri_t        H5Tdetect_class(hid_t type_id, H5T_class_t dtype_class) except *
  herr_t        H5Tclose(hid_t type_id) except *

  hid_t         H5Tget_native_type(hid_t type_id, H5T_direction_t direction) except *

  # Not for public API
  herr_t        H5Tconvert(hid_t src_id, hid_t dst_id, size_t nelmts, void *buf, void *background, hid_t plist_id) except *

  # Atomic datatypes
  herr_t        H5Tset_size(hid_t type_id, size_t size) except *

  H5T_order_t   H5Tget_order(hid_t type_id) except *
  herr_t        H5Tset_order(hid_t type_id, H5T_order_t order) except *

  hsize_t       H5Tget_precision(hid_t type_id) except? 0
  herr_t        H5Tset_precision(hid_t type_id, size_t prec) except *

  int           H5Tget_offset(hid_t type_id) except *
  herr_t        H5Tset_offset(hid_t type_id, size_t offset) except *

  herr_t        H5Tget_pad(hid_t type_id, H5T_pad_t * lsb, H5T_pad_t * msb ) except *
  herr_t        H5Tset_pad(hid_t type_id, H5T_pad_t lsb, H5T_pad_t msb ) except *

  H5T_sign_t    H5Tget_sign(hid_t type_id) except *
  herr_t        H5Tset_sign(hid_t type_id, H5T_sign_t sign) except *

  herr_t        H5Tget_fields(hid_t type_id, size_t *spos, size_t *epos, 
                                size_t *esize, size_t *mpos, size_t *msize ) except *
  herr_t        H5Tset_fields(hid_t type_id, size_t spos, size_t epos, 
                                size_t esize, size_t mpos, size_t msize ) except *

  size_t        H5Tget_ebias(hid_t type_id) except? 0
  herr_t        H5Tset_ebias(hid_t type_id, size_t ebias) except *
  H5T_norm_t    H5Tget_norm(hid_t type_id) except *
  herr_t        H5Tset_norm(hid_t type_id, H5T_norm_t norm) except *
  H5T_pad_t     H5Tget_inpad(hid_t type_id) except *
  herr_t        H5Tset_inpad(hid_t type_id, H5T_pad_t inpad) except *
  H5T_cset_t    H5Tget_cset(hid_t type_id) except *
  herr_t        H5Tset_cset(hid_t type_id, H5T_cset_t cset) except *
  H5T_str_t     H5Tget_strpad(hid_t type_id) except *
  herr_t        H5Tset_strpad(hid_t type_id, H5T_str_t strpad) except *

  # VLENs
  hid_t     H5Tvlen_create(hid_t base_type_id) except *
  htri_t    H5Tis_variable_str(hid_t dtype_id) except *

  # Compound data types
  int           H5Tget_nmembers(hid_t type_id) except *
  H5T_class_t   H5Tget_member_class(hid_t type_id, int member_no) except *
  char*         H5Tget_member_name(hid_t type_id, unsigned membno) except? NULL
  hid_t         H5Tget_member_type(hid_t type_id, unsigned membno) except *
  int           H5Tget_member_offset(hid_t type_id, int membno) except *
  int           H5Tget_member_index(hid_t type_id, char* name) except *
  herr_t        H5Tinsert(hid_t parent_id, char *name, size_t offset,
                   hid_t member_id) except *
  herr_t        H5Tpack(hid_t type_id) except *

  # Enumerated types
  hid_t     H5Tenum_create(hid_t base_id) except *
  herr_t    H5Tenum_insert(hid_t type, char *name, void *value) except *
  herr_t    H5Tenum_nameof( hid_t type, void *value, char *name, size_t size ) except *
  herr_t    H5Tenum_valueof( hid_t type, char *name, void *value ) except *
  herr_t    H5Tget_member_value(hid_t type,  unsigned int memb_no, void *value ) except *

  # Array data types
  hid_t H5Tarray_create(hid_t base_id, int ndims, hsize_t dims[], int perm[]) except *
  int   H5Tget_array_ndims(hid_t type_id) except *
  int   H5Tget_array_dims(hid_t type_id, hsize_t dims[], int perm[]) except *

  # Opaque data types
  herr_t    H5Tset_tag(hid_t type_id, char* tag) except *
  char*     H5Tget_tag(hid_t type_id) except? NULL

  # 1.8-specific functions
  IF H5PY_18API:
    hid_t H5Tdecode(unsigned char *buf) except *
    herr_t H5Tencode(hid_t obj_id, unsigned char *buf, size_t *nalloc) except *

    herr_t H5Tcommit2(hid_t loc_id, char *name, hid_t dtype_id, hid_t lcpl_id,
            hid_t tcpl_id, hid_t tapl_id) 

  # Type-conversion infrastructure

  ctypedef enum H5T_pers_t:
    H5T_PERS_DONTCARE	= -1,
    H5T_PERS_HARD	= 0,	    # /*hard conversion function		     */
    H5T_PERS_SOFT	= 1 	    # /*soft conversion function		     */

  ctypedef enum H5T_cmd_t:
    H5T_CONV_INIT	= 0,	#/*query and/or initialize private data	     */
    H5T_CONV_CONV	= 1, 	#/*convert data from source to dest datatype */
    H5T_CONV_FREE	= 2	    #/*function is being removed from path	     */

  ctypedef enum H5T_bkg_t:
    H5T_BKG_NO		= 0, 	#/*background buffer is not needed, send NULL */
    H5T_BKG_TEMP	= 1,	#/*bkg buffer used as temp storage only       */
    H5T_BKG_YES		= 2	    #/*init bkg buf with data before conversion   */

  ctypedef struct H5T_cdata_t:
    H5T_cmd_t		command     # /*what should the conversion function do?    */
    H5T_bkg_t		need_bkg   #/*is the background buffer needed?	     */
    hbool_t		recalc	        # /*recalculate private data		     */
    void		*priv	        # /*private data				     */

  ctypedef struct hvl_t:
      size_t len # /* Length of VL data (in base type units) */
      void *p    #/* Pointer to VL data */

  ctypedef herr_t (*H5T_conv_t)(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
      size_t nelmts, size_t buf_stride, size_t bkg_stride, void *buf,
      void *bkg, hid_t dset_xfer_plist) except -1

  H5T_conv_t H5Tfind(hid_t src_id, hid_t dst_id, H5T_cdata_t **pcdata) except *

  herr_t    H5Tregister(H5T_pers_t pers, char *name, hid_t src_id,
                        hid_t dst_id, H5T_conv_t func) except *
  herr_t    H5Tunregister(H5T_pers_t pers, char *name, hid_t src_id,
			            hid_t dst_id, H5T_conv_t func) except *



# === H5Z - Filters ===========================================================

cdef extern from "hdf5.h":

    ctypedef int H5Z_filter_t

    int H5Z_FILTER_ERROR
    int H5Z_FILTER_NONE
    int H5Z_FILTER_ALL
    int H5Z_FILTER_DEFLATE
    int H5Z_FILTER_SHUFFLE 
    int H5Z_FILTER_FLETCHER32
    int H5Z_FILTER_SZIP
    int H5Z_FILTER_RESERVED
    int H5Z_FILTER_MAX
    int H5Z_MAX_NFILTERS

    int H5Z_FLAG_DEFMASK
    int H5Z_FLAG_MANDATORY
    int H5Z_FLAG_OPTIONAL

    int H5Z_FLAG_INVMASK
    int H5Z_FLAG_REVERSE
    int H5Z_FLAG_SKIP_EDC

    int H5_SZIP_ALLOW_K13_OPTION_MASK   #1
    int H5_SZIP_CHIP_OPTION_MASK        #2
    int H5_SZIP_EC_OPTION_MASK          #4
    int H5_SZIP_NN_OPTION_MASK          #32
    int H5_SZIP_MAX_PIXELS_PER_BLOCK    #32

    int H5Z_FILTER_CONFIG_ENCODE_ENABLED #(0x0001)
    int H5Z_FILTER_CONFIG_DECODE_ENABLED #(0x0002)

    cdef enum H5Z_EDC_t:
        H5Z_ERROR_EDC       = -1,
        H5Z_DISABLE_EDC     = 0,
        H5Z_ENABLE_EDC      = 1,
        H5Z_NO_EDC          = 2 

    htri_t H5Zfilter_avail(H5Z_filter_t id_) except *
    herr_t H5Zget_filter_info(H5Z_filter_t filter_, unsigned int *filter_config_flags) except *


# === H5A - Attributes API ====================================================

cdef extern from "hdf5.h":

  hid_t     H5Acreate(hid_t loc_id, char *name, hid_t type_id, hid_t space_id, hid_t create_plist) except *
  hid_t     H5Aopen_idx(hid_t loc_id, unsigned int idx) except *
  hid_t     H5Aopen_name(hid_t loc_id, char *name) except *
  herr_t    H5Aclose(hid_t attr_id) except *
  herr_t    H5Adelete(hid_t loc_id, char *name) except *

  herr_t    H5Aread(hid_t attr_id, hid_t mem_type_id, void *buf) except *
  herr_t    H5Awrite(hid_t attr_id, hid_t mem_type_id, void *buf ) except *

  int       H5Aget_num_attrs(hid_t loc_id) except *
  ssize_t   H5Aget_name(hid_t attr_id, size_t buf_size, char *buf) except *
  hid_t     H5Aget_space(hid_t attr_id) except *
  hid_t     H5Aget_type(hid_t attr_id) except *

  ctypedef herr_t (*H5A_operator_t)(hid_t loc_id, char *attr_name, void* operator_data) except 2
  herr_t    H5Aiterate(hid_t loc_id, unsigned * idx, H5A_operator_t op, void* op_data) except *

  IF H5PY_18API:

    ctypedef struct H5A_info_t:
      hbool_t corder_valid          # Indicate if creation order is valid
      H5O_msg_crt_idx_t corder      # Creation order
      H5T_cset_t        cset        # Character set of attribute name
      hsize_t           data_size   # Size of raw data

    herr_t  H5Adelete_by_name(hid_t loc_id, char *obj_name, char *attr_name,
                hid_t lapl_id) except *
    herr_t  H5Adelete_by_idx(hid_t loc_id, char *obj_name, H5_index_t idx_type,
                H5_iter_order_t order, hsize_t n, hid_t lapl_id) except *

    hid_t H5Acreate_by_name(hid_t loc_id, char *obj_name, char *attr_name,
        hid_t type_id, hid_t space_id, hid_t acpl_id, hid_t aapl_id, hid_t lapl_id) except *
 
    herr_t H5Aopen(hid_t obj_id, char *attr_name, hid_t aapl_id) except *
    herr_t H5Aopen_by_name( hid_t loc_id, char *obj_name, char *attr_name,
        hid_t aapl_id, hid_t lapl_id) except *
    herr_t H5Aopen_by_idx(hid_t loc_id, char *obj_name, H5_index_t idx_type,
        H5_iter_order_t order, hsize_t n, hid_t aapl_id, hid_t lapl_id) except *
    htri_t H5Aexists_by_name( hid_t loc_id, char *obj_name, char *attr_name,
                                hid_t lapl_id) except *
    htri_t H5Aexists(hid_t obj_id, char *attr_name) except *

    herr_t H5Arename(hid_t loc_id, char *old_attr_name, char *new_attr_name) except *
    herr_t H5Arename_by_name(hid_t loc_id, char *obj_name, char *old_attr_name,
            char *new_attr_name, hid_t lapl_id) except *

    herr_t H5Aget_info( hid_t attr_id, H5A_info_t *ainfo) except *
    herr_t H5Aget_info_by_name(hid_t loc_id, char *obj_name, char *attr_name,
                                H5A_info_t *ainfo, hid_t lapl_id) except *
    herr_t H5Aget_info_by_idx(hid_t loc_id, char *obj_name, H5_index_t idx_type,
              H5_iter_order_t order, hsize_t n, H5A_info_t *ainfo, hid_t lapl_id) except *

    ctypedef herr_t (*H5A_operator2_t)(hid_t location_id, char *attr_name,
            H5A_info_t *ainfo, void *op_data) except 2
    herr_t H5Aiterate2(hid_t obj_id, H5_index_t idx_type, H5_iter_order_t order,
            hsize_t *n, H5A_operator2_t op, void *op_data) except *

    hsize_t H5Aget_storage_size(hid_t attr_id) except *




