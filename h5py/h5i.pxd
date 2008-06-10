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

from defs_c cimport size_t, time_t, ssize_t
from h5 cimport hid_t, hbool_t, herr_t, htri_t, hsize_t, hssize_t, hvl_t

cdef extern from "hdf5.h":

  # reflection
  ctypedef enum H5I_type_t:
    H5I_BADID        = -1,  # /*invalid Group                    */
    H5I_FILE        = 1,    # /*group ID for File objects            */
    H5I_GROUP,              # /*group ID for Group objects            */
    H5I_DATATYPE,           # /*group ID for Datatype objects            */
    H5I_DATASPACE,          # /*group ID for Dataspace objects        */
    H5I_DATASET,            # /*group ID for Dataset objects            */
    H5I_ATTR,               # /*group ID for Attribute objects        */
    H5I_REFERENCE,          # /*group ID for Reference objects        */
    H5I_VFL,                # /*group ID for virtual file layer        */
    H5I_GENPROP_CLS,        # /*group ID for generic property list classes */
    H5I_GENPROP_LST,        # /*group ID for generic property lists       */
    H5I_NGROUPS             # /*number of valid groups, MUST BE LAST!        */

  # --- Reflection ------------------------------------------------------------
  H5I_type_t H5Iget_type(hid_t obj_id) except *
  ssize_t    H5Iget_name( hid_t obj_id, char *name, size_t size) except *
  hid_t      H5Iget_file_id(hid_t obj_id) except *
  int        H5Idec_ref(hid_t obj_id) except *
  int        H5Iget_ref(hid_t obj_id) except *
  int        H5Iinc_ref(hid_t obj_id) except *





