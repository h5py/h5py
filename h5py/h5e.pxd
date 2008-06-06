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

# This file contains code or comments from the HDF5 library. The complete HDF5
# license is available in the file licenses/hdf5.txt in the distribution
# root directory.

from defs_c cimport size_t, time_t
from h5 cimport hid_t, hbool_t, herr_t, htri_t, hsize_t, hssize_t, hvl_t

cdef extern from "hdf5.h":

  ctypedef enum H5E_major_t:
    H5E_NONE_MAJOR       = 0,
    H5E_ARGS,
    H5E_RESOURCE,
    H5E_INTERNAL,
    H5E_FILE,
    H5E_IO,
    H5E_FUNC,
    H5E_ATOM,
    H5E_CACHE,
    H5E_BTREE,
    H5E_SYM,
    H5E_HEAP,
    H5E_OHDR,
    H5E_DATATYPE,
    H5E_DATASPACE,
    H5E_DATASET,
    H5E_STORAGE,
    H5E_PLIST,
    H5E_ATTR,
    H5E_PLINE,
    H5E_EFL,
    H5E_REFERENCE,
    H5E_VFL,
    H5E_TBBT,
    H5E_TST,
    H5E_RS,
    H5E_ERROR,
    H5E_SLIST

  ctypedef enum H5E_minor_t:
    H5E_NONE_MINOR       = 0

  cdef enum H5E_direction_t:
    H5E_WALK_UPWARD    = 0  #/*begin deep, end at API function    */
    H5E_WALK_DOWNWARD = 1   #/*begin at API function, end deep    */

  ctypedef struct H5E_error_t:
    H5E_major_t     maj_num        # /*major error number             */
    H5E_minor_t     min_num        # /*minor error number             */
    char    *func_name      # /*function in which error occurred   */
    char    *file_name      # /*file in which error occurred       */
    unsigned    line        # /*line in file where error occurs    */
    char    *desc           # /*optional supplied description      */

  # --- Error handling --------------------------------------------------------


  char      *H5Eget_major(H5E_major_t n)
  ctypedef herr_t (*H5E_auto_t)(void *client_data)
  herr_t    H5Eset_auto(H5E_auto_t func, void *client_data )

  ctypedef herr_t (*H5E_walk_t)(int n, H5E_error_t *err_desc, void* client_data)  
  herr_t    H5Ewalk(H5E_direction_t direction, H5E_walk_t func, void* client_data  )

