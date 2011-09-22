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

"""
    File driver constants (H5FD*).
"""

# === Multi-file driver =======================================================

MEM_DEFAULT = H5FD_MEM_DEFAULT
MEM_SUPER = H5FD_MEM_SUPER
MEM_BTREE = H5FD_MEM_BTREE
MEM_DRAW = H5FD_MEM_DRAW
MEM_GHEAP = H5FD_MEM_GHEAP
MEM_LHEAP = H5FD_MEM_LHEAP
MEM_OHDR = H5FD_MEM_OHDR
MEM_NTYPES = H5FD_MEM_NTYPES

# === Driver types ============================================================

CORE = H5FD_CORE
FAMILY = H5FD_FAMILY
LOG = H5FD_LOG
MPIO = H5FD_MPIO
MULTI = H5FD_MULTI
SEC2 = H5FD_SEC2
STDIO = H5FD_STDIO
IF UNAME_SYSNAME == "Windows":
    WINDOWS = H5FD_WINDOWS
ELSE:
    WINDOWS = -1

# === Logging driver ==========================================================

LOG_LOC_READ  = H5FD_LOG_LOC_READ   # 0x0001
LOG_LOC_WRITE = H5FD_LOG_LOC_WRITE  # 0x0002
LOG_LOC_SEEK  = H5FD_LOG_LOC_SEEK   # 0x0004
LOG_LOC_IO    = H5FD_LOG_LOC_IO     # (H5FD_LOG_LOC_READ|H5FD_LOG_LOC_WRITE|H5FD_LOG_LOC_SEEK)

# Flags for tracking number of times each byte is read/written 
LOG_FILE_READ = H5FD_LOG_FILE_READ  # 0x0008
LOG_FILE_WRITE= H5FD_LOG_FILE_WRITE # 0x0010
LOG_FILE_IO   = H5FD_LOG_FILE_IO    # (H5FD_LOG_FILE_READ|H5FD_LOG_FILE_WRITE)

# Flag for tracking "flavor" (type) of information stored at each byte 
LOG_FLAVOR    = H5FD_LOG_FLAVOR     # 0x0020

# Flags for tracking total number of reads/writes/seeks 
LOG_NUM_READ  = H5FD_LOG_NUM_READ   # 0x0040
LOG_NUM_WRITE = H5FD_LOG_NUM_WRITE  # 0x0080
LOG_NUM_SEEK  = H5FD_LOG_NUM_SEEK   # 0x0100
LOG_NUM_IO    = H5FD_LOG_NUM_IO     # (H5FD_LOG_NUM_READ|H5FD_LOG_NUM_WRITE|H5FD_LOG_NUM_SEEK)

# Flags for tracking time spent in open/read/write/seek/close 
LOG_TIME_OPEN = H5FD_LOG_TIME_OPEN  # 0x0200        # Not implemented yet 
LOG_TIME_READ = H5FD_LOG_TIME_READ  # 0x0400        # Not implemented yet 
LOG_TIME_WRITE= H5FD_LOG_TIME_WRITE # 0x0800        # Partially implemented (need to track total time) 
LOG_TIME_SEEK = H5FD_LOG_TIME_SEEK  # 0x1000        # Partially implemented (need to track total time & track time for seeks during reading) 
LOG_TIME_CLOSE= H5FD_LOG_TIME_CLOSE # 0x2000        # Fully implemented 
LOG_TIME_IO   = H5FD_LOG_TIME_IO    # (H5FD_LOG_TIME_OPEN|H5FD_LOG_TIME_READ|H5FD_LOG_TIME_WRITE|H5FD_LOG_TIME_SEEK|H5FD_LOG_TIME_CLOSE)

# Flag for tracking allocation of space in file 
LOG_ALLOC     = H5FD_LOG_ALLOC      # 0x4000
LOG_ALL       = H5FD_LOG_ALL        # (H5FD_LOG_ALLOC|H5FD_LOG_TIME_IO|H5FD_LOG_NUM_IO|H5FD_LOG_FLAVOR|H5FD_LOG_FILE_IO|H5FD_LOG_LOC_IO)

