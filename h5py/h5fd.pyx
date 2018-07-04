# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

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

# === MPI driver ==============================================================

MPIO_INDEPENDENT = H5FD_MPIO_INDEPENDENT
MPIO_COLLECTIVE = H5FD_MPIO_COLLECTIVE

# === Driver types ============================================================

CORE = H5FD_CORE
FAMILY = H5FD_FAMILY
LOG = H5FD_LOG
MPIO = H5FD_MPIO
MPIPOSIX = -1
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


# H5FD_t of file-like object
ctypedef struct H5FD_fileobj_t:
  H5FD_t base
  PyObject* fileobj
  haddr_t eoa


from cpython cimport Py_INCREF, Py_DECREF
from libc.stdint cimport *
from libc.stdio cimport *


cdef H5FD_fileobj_t *H5FD_fileobj_open(const char *name, unsigned flags, hid_t fapl, haddr_t maxaddr):
    cdef uintptr_t p
    cdef H5FD_fileobj_t *f
    if sscanf(name, "%lx", &p):
        f = <H5FD_fileobj_t *>malloc(sizeof(H5FD_fileobj_t))
        f.fileobj = <PyObject*>p
        Py_INCREF(<object>f.fileobj)
        f.eoa = 0
        return f
    else:
        return NULL

cdef herr_t H5FD_fileobj_close(H5FD_fileobj_t *f) except -1:
    Py_DECREF(<object>f.fileobj)
    return 0

cdef haddr_t H5FD_fileobj_get_eoa(const H5FD_fileobj_t *f, H5FD_mem_t type):
    return f.eoa

cdef herr_t H5FD_fileobj_set_eoa(H5FD_fileobj_t *f, H5FD_mem_t type, haddr_t addr) except -1:
    f.eoa = addr
    return 0

cdef haddr_t H5FD_fileobj_get_eof(const H5FD_fileobj_t *f, H5FD_mem_t type) except -1:
    (<object>f.fileobj).seek(0, SEEK_END)
    return (<object>f.fileobj).tell()

cdef herr_t H5FD_fileobj_read(H5FD_fileobj_t *f, H5FD_mem_t type, hid_t dxpl, haddr_t addr, size_t size, void *buf) except -1:
    (<object>f.fileobj).seek(addr)
    cdef b = (<object>f.fileobj).read(size)
    if len(b) != size:
        return 1
    cdef unsigned char[:] mview = bytearray(b)
    memcpy(buf, &mview[0], size)
    return 0

cdef herr_t H5FD_fileobj_write(H5FD_fileobj_t *f, H5FD_mem_t type, hid_t dxpl, haddr_t addr, size_t size, void *buf) except -1:
    (<object>f.fileobj).seek(addr)
    cdef b = bytearray(<unsigned char[:size]>buf)
    (<object>f.fileobj).write(b)
    return 0

cdef herr_t H5FD_fileobj_truncate(H5FD_fileobj_t *f, hid_t dxpl, hbool_t closing) except -1:
    (<object>f.fileobj).truncate(f.eoa)
    return 0

cdef herr_t H5FD_fileobj_flush(H5FD_fileobj_t *f, hid_t dxpl, hbool_t closing) except -1:
    (<object>f.fileobj).flush()
    return 0


cdef H5FD_class_t info
memset(&info, 0, sizeof(info))

info.name = 'fileobj'
info.maxaddr = SIZE_MAX - 1
info.fc_degree = H5F_CLOSE_WEAK
info.open = <H5FD_t *(*)(const char *name, unsigned flags, hid_t fapl, haddr_t maxaddr)>H5FD_fileobj_open
info.close = <herr_t (*)(H5FD_t *)>H5FD_fileobj_close
info.get_eoa = <haddr_t (*)(const H5FD_t *, H5FD_mem_t)>H5FD_fileobj_get_eoa
info.set_eoa = <herr_t (*)(H5FD_t *, H5FD_mem_t, haddr_t)>H5FD_fileobj_set_eoa
info.get_eof = <haddr_t (*)(const H5FD_t *, H5FD_mem_t)>H5FD_fileobj_get_eof
info.read = <herr_t (*)(H5FD_t *, H5FD_mem_t, hid_t, haddr_t, size_t, void *)>H5FD_fileobj_read
info.write = <herr_t (*)(H5FD_t *, H5FD_mem_t, hid_t, haddr_t, size_t, const void *)>H5FD_fileobj_write
info.truncate = <herr_t (*)(H5FD_t *, hid_t, hbool_t)>H5FD_fileobj_truncate
info.flush = <herr_t (*)(H5FD_t *, hid_t, hbool_t)>H5FD_fileobj_flush

fileobj_driver = H5FDregister(&info)
