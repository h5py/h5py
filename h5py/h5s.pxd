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

# This file contains code or comments from the HDF5 library. The complete HDF5
# license is available in the file licenses/hdf5.txt in the distribution
# root directory.

from defs_c cimport size_t, time_t
from h5 cimport hid_t, hbool_t, herr_t, htri_t, hsize_t, hssize_t, hvl_t

cdef extern from "hdf5.h":

  int H5S_ALL, H5S_UNLIMITED, H5S_MAX_RANK

  # Codes for defining selections
  cdef enum H5S_seloper_t:
    H5S_SELECT_NOOP      = -1,
    H5S_SELECT_SET       = 0,
    H5S_SELECT_OR,
    H5S_SELECT_AND,
    H5S_SELECT_XOR,
    H5S_SELECT_NOTB,
    H5S_SELECT_NOTA,
    H5S_SELECT_APPEND,
    H5S_SELECT_PREPEND,
    H5S_SELECT_INVALID    # Must be the last one

  cdef enum H5S_class_t:
    H5S_NO_CLASS         = -1,  #/*error                                      */
    H5S_SCALAR           = 0,   #/*scalar variable                            */
    H5S_SIMPLE           = 1,   #/*simple data space                          */
    H5S_COMPLEX          = 2    #/*complex data space                         */

  cdef enum H5S_sel_type:
    H5S_SEL_ERROR	= -1, 	    #/* Error			*/
    H5S_SEL_NONE	= 0,        #/* Nothing selected 		*/
    H5S_SEL_POINTS	= 1,        #/* Sequence of points selected	*/
    H5S_SEL_HYPERSLABS  = 2,    #/* "New-style" hyperslab selection defined	*/
    H5S_SEL_ALL		= 3,        #/* Entire extent selected	*/
    H5S_SEL_N		= 4	        #/*THIS MUST BE LAST		*/


  # --- Dataspace operations --------------------------------------------------
  hid_t H5Screate(H5S_class_t type)
  hid_t H5Screate_simple(int rank, hsize_t dims[], hsize_t maxdims[])
  int H5Sget_simple_extent_ndims(hid_t space_id)
  int H5Sget_simple_extent_dims(hid_t space_id, hsize_t dims[],
                                hsize_t maxdims[])
  herr_t H5Sselect_hyperslab(hid_t space_id, H5S_seloper_t op,
                             hsize_t start[], hsize_t _stride[],
                             hsize_t count[], hsize_t _block[])
  herr_t H5Sclose(hid_t space_id)
  herr_t H5Sget_select_bounds(hid_t space_id, hsize_t *start, hsize_t *end)
  herr_t H5Sselect_none(hid_t space_id)
  H5S_class_t H5Sget_simple_extent_type(hid_t space_id)

