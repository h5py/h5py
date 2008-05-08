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

  cdef enum H5E_direction_t:
    H5E_WALK_UPWARD    = 0  #/*begin deep, end at API function    */
    H5E_WALK_DOWNWARD = 1   #/*begin at API function, end deep    */

  ctypedef struct H5E_error_t:
    int      maj_num        # /*major error number             */
    int      min_num        # /*minor error number             */
    char    *func_name      # /*function in which error occurred   */
    char    *file_name      # /*file in which error occurred       */
    unsigned    line        # /*line in file where error occurs    */
    char    *desc           # /*optional supplied description      */

  # --- Error handling --------------------------------------------------------
  herr_t    H5Eset_auto(void* opt1, void* opt2)
  ctypedef herr_t (*H5E_walk_t)(int n, H5E_error_t *err_desc, client_data)  
  herr_t    H5Ewalk(H5E_direction_t direction, H5E_walk_t func, client_data  )


