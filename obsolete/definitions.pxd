#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: June 20, 2005
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Id: definitions.pyd 1018 2005-06-20 09:43:34Z faltet $
#
########################################################################

"""Here are some definitions for sharing between extensions.

"""

import sys

from defs_c cimport size_t, time_t
from defs_h5 cimport hid_t, hbool_t, herr_t, htri_t, hsize_t, hssize_t, hvl_t

# Structs and types from HDF5
cdef extern from "hdf5.h":


  int H5FD_LOG_LOC_WRITE, H5FD_LOG_ALL
  int H5I_INVALID_HID

  # Native types
  # NOT MOVED
  cdef enum:
    H5T_C_S1


  # The order to retrieve atomic native datatype
  # NOT MOVED
  cdef enum H5T_direction_t:
    H5T_DIR_DEFAULT     = 0,    #default direction is inscendent
    H5T_DIR_ASCEND      = 1,    #in inscendent order
    H5T_DIR_DESCEND     = 2     #in descendent order






  # === HDF5 API ==============================================================





  










