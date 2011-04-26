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

# This file contains code or comments from the HDF5 library.  See the file
# licenses/hdf5.txt for the full HDF5 software license.

from defs cimport *

cdef extern from "hdf5.h":

  ctypedef enum H5FD_mem_t:
    H5FD_MEM_NOLIST	= -1,
    H5FD_MEM_DEFAULT	= 0,
    H5FD_MEM_SUPER      = 1,
    H5FD_MEM_BTREE      = 2,
    H5FD_MEM_DRAW       = 3,
    H5FD_MEM_GHEAP      = 4,
    H5FD_MEM_LHEAP      = 5,
    H5FD_MEM_OHDR       = 6,
    H5FD_MEM_NTYPES

  # HDF5 uses a clever scheme wherein these are actually init() calls
  # Hopefully Pyrex won't have a problem with this.
  # Thankfully they are defined but -1 if unavailable
  hid_t H5FD_CORE
  hid_t H5FD_FAMILY
# hid_t H5FD_GASS  not in 1.8.X
  hid_t H5FD_LOG
  hid_t H5FD_MPIO
  hid_t H5FD_MULTI
  hid_t H5FD_SEC2
  hid_t H5FD_STDIO
  IF UNAME_SYSNAME == "Windows":
    hid_t H5FD_WINDOWS

  int H5FD_LOG_LOC_READ   # 0x0001
  int H5FD_LOG_LOC_WRITE  # 0x0002
  int H5FD_LOG_LOC_SEEK   # 0x0004
  int H5FD_LOG_LOC_IO     # (H5FD_LOG_LOC_READ|H5FD_LOG_LOC_WRITE|H5FD_LOG_LOC_SEEK)

  # /* Flags for tracking number of times each byte is read/written */
  int H5FD_LOG_FILE_READ  # 0x0008
  int H5FD_LOG_FILE_WRITE # 0x0010
  int H5FD_LOG_FILE_IO    # (H5FD_LOG_FILE_READ|H5FD_LOG_FILE_WRITE)

  # /* Flag for tracking "flavor" (type) of information stored at each byte */
  int H5FD_LOG_FLAVOR     # 0x0020

  # /* Flags for tracking total number of reads/writes/seeks */
  int H5FD_LOG_NUM_READ   # 0x0040
  int H5FD_LOG_NUM_WRITE  # 0x0080
  int H5FD_LOG_NUM_SEEK   # 0x0100
  int H5FD_LOG_NUM_IO     # (H5FD_LOG_NUM_READ|H5FD_LOG_NUM_WRITE|H5FD_LOG_NUM_SEEK)

  # /* Flags for tracking time spent in open/read/write/seek/close */
  int H5FD_LOG_TIME_OPEN  # 0x0200        # /* Not implemented yet */
  int H5FD_LOG_TIME_READ  # 0x0400        # /* Not implemented yet */
  int H5FD_LOG_TIME_WRITE # 0x0800        # /* Partially implemented (need to track total time) */
  int H5FD_LOG_TIME_SEEK  # 0x1000        # /* Partially implemented (need to track total time & track time for seeks during reading) */
  int H5FD_LOG_TIME_CLOSE # 0x2000        # /* Fully implemented */
  int H5FD_LOG_TIME_IO    # (H5FD_LOG_TIME_OPEN|H5FD_LOG_TIME_READ|H5FD_LOG_TIME_WRITE|H5FD_LOG_TIME_SEEK|H5FD_LOG_TIME_CLOSE)

  # /* Flag for tracking allocation of space in file */
  int H5FD_LOG_ALLOC      # 0x4000
  int H5FD_LOG_ALL        # (H5FD_LOG_ALLOC|H5FD_LOG_TIME_IO|H5FD_LOG_NUM_IO|H5FD_LOG_FLAVOR|H5FD_LOG_FILE_IO|H5FD_LOG_LOC_IO)

