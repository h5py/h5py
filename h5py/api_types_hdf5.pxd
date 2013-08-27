# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

from api_types_ext cimport *

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

  # New in 1.8.X
  ctypedef enum H5_iter_order_t:
    H5_ITER_UNKNOWN  =-1
    H5_ITER_INC
    H5_ITER_DEC
    H5_ITER_NATIVE
    H5_ITER_N

  ctypedef enum H5_index_t:
    H5_INDEX_UNKNOWN    =-1
    H5_INDEX_NAME
    H5_INDEX_CRT_ORDER
    H5_INDEX_N

# === H5D - Dataset API =======================================================

  ctypedef enum H5D_layout_t:
    H5D_LAYOUT_ERROR  =-1
    H5D_COMPACT       = 0
    H5D_CONTIGUOUS    = 1
    H5D_CHUNKED       = 2
    H5D_NLAYOUTS      = 3

  ctypedef enum H5D_alloc_time_t:
    H5D_ALLOC_TIME_ERROR    =-1
    H5D_ALLOC_TIME_DEFAULT  = 0
    H5D_ALLOC_TIME_EARLY    = 1
    H5D_ALLOC_TIME_LATE     = 2
    H5D_ALLOC_TIME_INCR     = 3

  ctypedef enum H5D_space_status_t:
    H5D_SPACE_STATUS_ERROR           =-1
    H5D_SPACE_STATUS_NOT_ALLOCATED   = 0
    H5D_SPACE_STATUS_PART_ALLOCATED  = 1
    H5D_SPACE_STATUS_ALLOCATED       = 2

  ctypedef enum H5D_fill_time_t:
    H5D_FILL_TIME_ERROR  =-1
    H5D_FILL_TIME_ALLOC  = 0
    H5D_FILL_TIME_NEVER  = 1
    H5D_FILL_TIME_IFSET  = 2

  ctypedef enum H5D_fill_value_t:
    H5D_FILL_VALUE_ERROR         =-1
    H5D_FILL_VALUE_UNDEFINED     = 0
    H5D_FILL_VALUE_DEFAULT       = 1
    H5D_FILL_VALUE_USER_DEFINED  = 2

  ctypedef  herr_t (*H5D_operator_t)(void *elem, hid_t type_id, unsigned ndim,
                    hsize_t *point, void *operator_data) except -1

# === H5F - File API ==========================================================

  # File constants
  cdef enum:
    H5F_ACC_RDONLY
    H5F_ACC_RDWR
    H5F_ACC_TRUNC
    H5F_ACC_EXCL
    H5F_ACC_DEBUG
    H5F_ACC_CREAT

  # The difference between a single file and a set of mounted files
  ctypedef enum H5F_scope_t:
    H5F_SCOPE_LOCAL   = 0
    H5F_SCOPE_GLOBAL  = 1
    H5F_SCOPE_DOWN      = 2     

  ctypedef enum H5F_close_degree_t:
    H5F_CLOSE_DEFAULT  = 0
    H5F_CLOSE_WEAK     = 1
    H5F_CLOSE_SEMI     = 2
    H5F_CLOSE_STRONG   = 3

  int H5F_OBJ_FILE
  int H5F_OBJ_DATASET
  int H5F_OBJ_GROUP
  int H5F_OBJ_DATATYPE
  int H5F_OBJ_ATTR
  int H5F_OBJ_ALL
  int H5F_OBJ_LOCAL

  ctypedef enum H5F_libver_t:
    H5F_LIBVER_EARLIEST
    H5F_LIBVER_LATEST

# === H5FD - Low-level file descriptor API ====================================

  ctypedef enum H5FD_mem_t:
    H5FD_MEM_NOLIST   =-1
    H5FD_MEM_DEFAULT  = 0
    H5FD_MEM_SUPER    = 1
    H5FD_MEM_BTREE    = 2
    H5FD_MEM_DRAW     = 3
    H5FD_MEM_GHEAP    = 4
    H5FD_MEM_LHEAP    = 5
    H5FD_MEM_OHDR     = 6
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

  ctypedef enum H5G_link_t:
    H5G_LINK_ERROR      = -1
    H5G_LINK_HARD       = 0
    H5G_LINK_SOFT       = 1


  ctypedef enum H5G_obj_t:
    H5G_UNKNOWN     =-1
    H5G_GROUP
    H5G_DATASET
    H5G_TYPE
    H5G_LINK
    H5G_UDLINK

  ctypedef struct H5G_stat_t:
    unsigned long   fileno[2]
    unsigned long   objno[2]
    unsigned        nlink
    H5G_obj_t       type
    time_t          mtime
    size_t          linklen
    #H5O_stat_t      ohdr            # Object header information. New in HDF5 1.6

  ctypedef herr_t (*H5G_iterate_t)(hid_t group, char *name, void* op_data) except 2

  ctypedef enum H5G_storage_type_t:
    H5G_STORAGE_TYPE_UNKNOWN       =-1
    H5G_STORAGE_TYPE_SYMBOL_TABLE
    H5G_STORAGE_TYPE_COMPACT
    H5G_STORAGE_TYPE_DENSE

  ctypedef struct H5G_info_t:
    H5G_storage_type_t   storage_type
    hsize_t              nlinks
    int64_t              max_corder
    #hbool_t              mounted

# === H5I - Identifier and reflection interface ===============================

  ctypedef enum H5I_type_t:
    H5I_UNINIT       =-2
    H5I_BADID        =-1
    H5I_FILE         = 1
    H5I_GROUP
    H5I_DATATYPE
    H5I_DATASPACE
    H5I_DATASET
    H5I_ATTR
    H5I_REFERENCE
    H5I_VFL
    H5I_GENPROP_CLS
    H5I_GENPROP_LST
    H5I_ERROR_CLASS
    H5I_ERROR_MSG
    H5I_ERROR_STACK
    H5I_NTYPES

# === H5L/H5O - Links interface (1.8.X only) ======================================

  # TODO: put both versions in h5t.pxd
  ctypedef enum H5T_cset_t:
    H5T_CSET_ERROR        =-1
    H5T_CSET_ASCII        = 0
    H5T_CSET_UTF8         = 1

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
    H5L_TYPE_ERROR     =-1
    H5L_TYPE_HARD      = 0
    H5L_TYPE_SOFT      = 1
    H5L_TYPE_EXTERNAL  = 64
    H5L_TYPE_MAX       = 255

  #  Information struct for link (for H5Lget_info/H5Lget_info_by_idx)
  cdef union _add_u:
    haddr_t address   #  Address hard link points to
    size_t val_size   #  Size of a soft link or UD link value

  ctypedef struct H5L_info_t:
    H5L_type_t   type
    hbool_t      corder_valid
    int64_t      corder
    H5T_cset_t   cset
    _add_u u

  #  Prototype for H5Literate/H5Literate_by_name() operator
  ctypedef herr_t (*H5L_iterate_t) (hid_t group, char *name, H5L_info_t *info,
                    void *op_data) except 2

  ctypedef uint32_t H5O_msg_crt_idx_t

  ctypedef enum H5O_type_t:
    H5O_TYPE_UNKNOWN         =-1
    H5O_TYPE_GROUP
    H5O_TYPE_DATASET
    H5O_TYPE_NAMED_DATATYPE
    H5O_TYPE_NTYPES

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

  ctypedef struct H5O_hdr_info_t:
    unsigned   version
    unsigned   nmesgs
    unsigned   nchunks
    unsigned   flags
    space      space
    mesg       mesg

  ctypedef struct H5_ih_info_t:
    hsize_t   index_size
    hsize_t   heap_size

  cdef struct meta_size:
    H5_ih_info_t   obj    #        v1/v2 B-tree & local/fractal heap for groups, B-tree for chunked datasets
    H5_ih_info_t   attr   #        v2 B-tree & heap for attributes

  ctypedef struct H5O_info_t:
    unsigned long    fileno
    haddr_t          addr
    H5O_type_t       type
    unsigned         rc
    time_t           atime
    time_t           mtime
    time_t           ctime
    time_t           btime
    hsize_t          num_attrs
    H5O_hdr_info_t   hdr
    meta_size       meta_size

  ctypedef herr_t (*H5O_iterate_t)(hid_t obj, char *name, H5O_info_t *info,
                    void *op_data) except 2

# === H5P - Property list API =================================================

  int H5P_DEFAULT

  ctypedef int H5Z_filter_t

  # HDF5 layouts
  ctypedef enum H5D_layout_t:
    H5D_LAYOUT_ERROR  =-1
    H5D_COMPACT       = 0
    H5D_CONTIGUOUS    = 1
    H5D_CHUNKED       = 2
    H5D_NLAYOUTS      = 3

  ctypedef enum H5D_alloc_time_t:
    H5D_ALLOC_TIME_ERROR    =-1
    H5D_ALLOC_TIME_DEFAULT  = 0
    H5D_ALLOC_TIME_EARLY    = 1
    H5D_ALLOC_TIME_LATE     = 2
    H5D_ALLOC_TIME_INCR     = 3

  ctypedef enum H5D_space_status_t:
    H5D_SPACE_STATUS_ERROR           =-1
    H5D_SPACE_STATUS_NOT_ALLOCATED   = 0
    H5D_SPACE_STATUS_PART_ALLOCATED  = 1
    H5D_SPACE_STATUS_ALLOCATED       = 2

  ctypedef enum H5D_fill_time_t:
    H5D_FILL_TIME_ERROR  =-1
    H5D_FILL_TIME_ALLOC  = 0
    H5D_FILL_TIME_NEVER  = 1
    H5D_FILL_TIME_IFSET  = 2

  ctypedef enum H5D_fill_value_t:
    H5D_FILL_VALUE_ERROR         =-1
    H5D_FILL_VALUE_UNDEFINED     = 0
    H5D_FILL_VALUE_DEFAULT       = 1
    H5D_FILL_VALUE_USER_DEFINED  = 2

  ctypedef enum H5Z_EDC_t:
    H5Z_ERROR_EDC    =-1
    H5Z_DISABLE_EDC  = 0
    H5Z_ENABLE_EDC   = 1
    H5Z_NO_EDC       = 2

  ctypedef enum H5F_close_degree_t:
    H5F_CLOSE_DEFAULT  = 0
    H5F_CLOSE_WEAK     = 1
    H5F_CLOSE_SEMI     = 2
    H5F_CLOSE_STRONG   = 3

  ctypedef enum H5FD_mem_t:
    H5FD_MEM_NOLIST   =-1
    H5FD_MEM_DEFAULT  = 0
    H5FD_MEM_SUPER    = 1
    H5FD_MEM_BTREE    = 2
    H5FD_MEM_DRAW     = 3
    H5FD_MEM_GHEAP    = 4
    H5FD_MEM_LHEAP    = 5
    H5FD_MEM_OHDR     = 6
    H5FD_MEM_NTYPES

  # Property list classes
  hid_t H5P_NO_CLASS
  hid_t H5P_FILE_CREATE
  hid_t H5P_FILE_ACCESS
  hid_t H5P_DATASET_CREATE
  hid_t H5P_DATASET_ACCESS
  hid_t H5P_DATASET_XFER

  hid_t H5P_OBJECT_CREATE
  hid_t H5P_OBJECT_COPY
  hid_t H5P_LINK_CREATE
  hid_t H5P_LINK_ACCESS
  hid_t H5P_GROUP_CREATE
  hid_t H5P_CRT_ORDER_TRACKED
  hid_t H5P_CRT_ORDER_INDEXED

# === H5R - Reference API =====================================================

  size_t H5R_DSET_REG_REF_BUF_SIZE
  size_t H5R_OBJ_REF_BUF_SIZE

  ctypedef enum H5R_type_t:
    H5R_BADTYPE         =-1
    H5R_OBJECT
    H5R_DATASET_REGION
    H5R_MAXTYPE

# === H5S - Dataspaces ========================================================

  int H5S_ALL, H5S_MAX_RANK
  hsize_t H5S_UNLIMITED

  # Codes for defining selections
  ctypedef enum H5S_seloper_t:
    H5S_SELECT_NOOP     =-1
    H5S_SELECT_SET      = 0
    H5S_SELECT_OR
    H5S_SELECT_AND
    H5S_SELECT_XOR
    H5S_SELECT_NOTB
    H5S_SELECT_NOTA
    H5S_SELECT_APPEND
    H5S_SELECT_PREPEND
    H5S_SELECT_INVALID

  ctypedef enum H5S_class_t:
    H5S_NO_CLASS  =-1
    H5S_SCALAR    = 0
    H5S_SIMPLE    = 1
    H5S_NULL      = 2
    # no longer defined in 1.8
    #H5S_COMPLEX          = 2    #/*complex data space

  ctypedef enum H5S_sel_type:
    H5S_SEL_ERROR       =-1
    H5S_SEL_NONE        = 0
    H5S_SEL_POINTS      = 1
    H5S_SEL_HYPERSLABS  = 2
    H5S_SEL_ALL         = 3
    H5S_SEL_N

# === H5T - Datatypes =========================================================

  # --- Enumerated constants --------------------------------------------------

  # Byte orders
  ctypedef enum H5T_order_t:
    H5T_ORDER_ERROR  =-1
    H5T_ORDER_LE     = 0
    H5T_ORDER_BE     = 1
    H5T_ORDER_VAX    = 2
    H5T_ORDER_MIXED  = 3
    H5T_ORDER_NONE   = 4

  # HDF5 signed enums
  ctypedef enum H5T_sign_t:
    H5T_SGN_ERROR  =-1
    H5T_SGN_NONE   = 0
    H5T_SGN_2      = 1
    H5T_NSGN       = 2

  ctypedef enum H5T_norm_t:
    H5T_NORM_ERROR    =-1
    H5T_NORM_IMPLIED  = 0
    H5T_NORM_MSBSET   = 1
    H5T_NORM_NONE     = 2

  ctypedef enum H5T_cset_t:
    H5T_CSET_ERROR        =-1
    H5T_CSET_ASCII        = 0

  ctypedef enum H5T_str_t:
    H5T_STR_ERROR        =-1
    H5T_STR_NULLTERM     = 0
    H5T_STR_NULLPAD      = 1
    H5T_STR_SPACEPAD     = 2

  # Atomic datatype padding
  ctypedef enum H5T_pad_t:
    H5T_PAD_ERROR       =-1
    H5T_PAD_ZERO        = 0
    H5T_PAD_ONE         = 1
    H5T_PAD_BACKGROUND  = 2
    H5T_NPAD            = 3

  # HDF5 type classes
  ctypedef enum H5T_class_t:
    H5T_NO_CLASS   =-1
    H5T_INTEGER    = 0
    H5T_FLOAT      = 1
    H5T_TIME       = 2
    H5T_STRING     = 3
    H5T_BITFIELD   = 4
    H5T_OPAQUE     = 5
    H5T_COMPOUND   = 6
    H5T_REFERENCE  = 7
    H5T_ENUM       = 8
    H5T_VLEN       = 9
    H5T_ARRAY      = 10
    H5T_NCLASSES

  # Native search direction
  ctypedef enum H5T_direction_t:
    H5T_DIR_DEFAULT  = 0
    H5T_DIR_ASCEND   = 1
    H5T_DIR_DESCEND  = 2

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

  # Type-conversion infrastructure

  ctypedef enum H5T_pers_t:
    H5T_PERS_DONTCARE  =-1
    H5T_PERS_HARD      = 0
    H5T_PERS_SOFT      = 1

  ctypedef enum H5T_cmd_t:
    H5T_CONV_INIT  = 0
    H5T_CONV_CONV  = 1
    H5T_CONV_FREE  = 2

  ctypedef enum H5T_bkg_t:
    H5T_BKG_NO    = 0
    H5T_BKG_TEMP  = 1
    H5T_BKG_YES   = 2

  ctypedef struct H5T_cdata_t:
    H5T_cmd_t   command
    H5T_bkg_t   need_bkg
    hbool_t     recalc
    void        *priv

  ctypedef struct hvl_t:
    size_t   len
    void     *p

  ctypedef herr_t (*H5T_conv_t)(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
      size_t nelmts, size_t buf_stride, size_t bkg_stride, void *buf,
      void *bkg, hid_t dset_xfer_plist) except -1

# === H5Z - Filters ===========================================================

  ctypedef int H5Z_filter_t

  int H5Z_FILTER_ERROR
  int H5Z_FILTER_NONE
  int H5Z_FILTER_ALL
  int H5Z_FILTER_DEFLATE
  int H5Z_FILTER_SHUFFLE
  int H5Z_FILTER_FLETCHER32
  int H5Z_FILTER_SZIP
  int H5Z_FILTER_NBIT
  int H5Z_FILTER_SCALEOFFSET
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
  
  int H5Z_SO_INT_MINBITS_DEFAULT

  int H5Z_FILTER_CONFIG_ENCODE_ENABLED #(0x0001)
  int H5Z_FILTER_CONFIG_DECODE_ENABLED #(0x0002)

  ctypedef enum H5Z_EDC_t:
    H5Z_ERROR_EDC    =-1
    H5Z_DISABLE_EDC  = 0
    H5Z_ENABLE_EDC   = 1
    H5Z_NO_EDC       = 2

  ctypedef enum H5Z_SO_scale_type_t:
    H5Z_SO_FLOAT_DSCALE  = 0
    H5Z_SO_FLOAT_ESCALE  = 1
    H5Z_SO_INT           = 2

# === H5A - Attributes API ====================================================

  ctypedef herr_t (*H5A_operator_t)(hid_t loc_id, char *attr_name, void* operator_data) except 2

  ctypedef struct H5A_info_t:
    hbool_t             corder_valid
    H5O_msg_crt_idx_t   corder
    H5T_cset_t          cset
    hsize_t             data_size

  ctypedef herr_t (*H5A_operator2_t)(hid_t location_id, char *attr_name,
          H5A_info_t *ainfo, void *op_data) except 2



#  === H5AC - Attribute Cache configuration API ================================


  unsigned int H5AC__CURR_CACHE_CONFIG_VERSION  # 	1
  # I don't really understand why this works, but
  # https://groups.google.com/forum/?fromgroups#!topic/cython-users/-fLG08E5lYM
  # suggests it and it _does_ work
  enum: H5AC__MAX_TRACE_FILE_NAME_LEN	#	1024

  unsigned int H5AC_METADATA_WRITE_STRATEGY__PROCESS_0_ONLY   # 0
  unsigned int H5AC_METADATA_WRITE_STRATEGY__DISTRIBUTED      # 1


  cdef extern from "H5Cpublic.h":
  # === H5C - Cache configuration API ================================
    cdef enum H5C_cache_incr_mode:
      H5C_incr__off,
      H5C_incr__threshold,


    cdef enum H5C_cache_flash_incr_mode:
      H5C_flash_incr__off,
      H5C_flash_incr__add_space


    cdef enum H5C_cache_decr_mode:
      H5C_decr__off,
      H5C_decr__threshold,
      H5C_decr__age_out,
      H5C_decr__age_out_with_threshold

    ctypedef struct H5AC_cache_config_t:
      #     /* general configuration fields: */
      int version
      hbool_t rpt_fcn_enabled
    #hbool_t    open_trace_file
    #hbool_t    close_trace_file
    #char       trace_file_name[]
      hbool_t evictions_enabled
      hbool_t set_initial_size
      size_t initial_size
      double min_clean_fraction
      size_t max_size
      size_t min_size
      long int epoch_length
      #    /* size increase control fields: */
      H5C_cache_incr_mode incr_mode
      double lower_hr_threshold
      double increment
      hbool_t apply_max_increment
      size_t max_increment
      H5C_cache_flash_incr_mode flash_incr_mode
      double flash_multiple
      double flash_threshold
      # /* size decrease control fields: */
      H5C_cache_decr_mode decr_mode
      double upper_hr_threshold
      double decrement
      hbool_t apply_max_decrement
      size_t max_decrement
      int epochs_before_eviction
      hbool_t apply_empty_reserve
      double empty_reserve
      # /* parallel configuration fields: */
      int dirty_bytes_threshold
      #  int metadata_write_strategy # present in 1.8.6 and higher




cdef extern from "hdf5_hl.h":
# === H5DS - Dimension Scales API =============================================

  ctypedef herr_t  (*H5DS_iterate_t)(hid_t dset, unsigned dim, hid_t scale, void *visitor_data) except 2

# === fixing const =============================================
#http://wiki.cython.org/FAQ#HowdoIuse.27const.27.3F
cdef extern from *:
  ctypedef void const_void "const void"
  ctypedef char const_char "const char"
  ctypedef int const_int "const int"

  ctypedef long long const_long_long "const long long"

  ctypedef unsigned char const_unsigned_char "const unsigned char"
  ctypedef unsigned short const_unsigned_short "const unsigned short"
  ctypedef unsigned int const_unsigned_int "const unsigned int"
  ctypedef unsigned long const_unsigned_long "const unsigned long"

  ctypedef hsize_t const_hsize_t "const hsize_t"
  ctypedef hssize_t const_hssize_t "const hssize_t"
  ctypedef hid_t const_hid_t "const hid_t"
  ctypedef haddr_t const_haddr_t "const haddr_t"
  ctypedef H5FD_mem_t const_H5FD_mem_t "const H5FD_mem_t"
