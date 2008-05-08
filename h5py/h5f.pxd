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

  # --- File operations -------------------------------------------------------
  hid_t  H5Fcreate(char *filename, unsigned int flags,
                   hid_t create_plist, hid_t access_plist)
  hid_t  H5Fopen(char *name, unsigned flags, hid_t access_id)
  herr_t H5Fclose (hid_t file_id)
  htri_t H5Fis_hdf5(char *name)
  herr_t H5Fflush(hid_t object_id, H5F_scope_t scope)


