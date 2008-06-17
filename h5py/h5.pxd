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

from defs_c cimport size_t, ssize_t

# Common structs and types from HDF5
cdef extern from "hdf5.h":

  ctypedef int hid_t  # In H5Ipublic.h
  ctypedef int hbool_t
  ctypedef int herr_t
  ctypedef int htri_t
  # hsize_t should be unsigned, but Windows platform does not support
  # such an unsigned long long type.
  ctypedef long long hsize_t
  ctypedef signed long long hssize_t
  ctypedef signed long long haddr_t  # I suppose this must be signed as well...

  ctypedef struct hvl_t:
    size_t len                 # Length of VL data (in base type units)
    void *p                    # Pointer to VL data

  int HADDR_UNDEF

  herr_t H5open() except *
  herr_t H5close() except *

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

  # --- Version functions -----------------------------------------------------
  herr_t H5get_libversion(unsigned *majnum, unsigned *minnum,
                          unsigned *relnum ) except *

# === Custom identifier wrappers ==============================================

cdef class ObjectID:
    """ Base wrapper class for HDF5 object identifiers """
    cdef readonly hid_t id

cdef class LockableID(ObjectID):
    cdef readonly int _locked






















