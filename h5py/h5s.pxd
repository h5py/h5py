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

include "std_defs.pxi"
from h5s cimport ObjectID

cdef class SpaceID(ObjectID):
    pass

cdef extern from "hdf5.h":

  int H5S_ALL, H5S_MAX_RANK
  hsize_t H5S_UNLIMITED

  # Codes for defining selections
  ctypedef enum H5S_seloper_t:
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

  ctypedef enum H5S_class_t:
    H5S_NO_CLASS         = -1,  #/*error                                      */
    H5S_SCALAR           = 0,   #/*scalar variable                            */
    H5S_SIMPLE           = 1,   #/*simple data space                          */
    # no longer defined in 1.8
    #H5S_COMPLEX          = 2    #/*complex data space                         */

  ctypedef enum H5S_sel_type:
    H5S_SEL_ERROR	= -1, 	    #/* Error			*/
    H5S_SEL_NONE	= 0,        #/* Nothing selected 		*/
    H5S_SEL_POINTS	= 1,        #/* Sequence of points selected	*/
    H5S_SEL_HYPERSLABS  = 2,    #/* "New-style" hyperslab selection defined	*/
    H5S_SEL_ALL		= 3,        #/* Entire extent selected	*/
    H5S_SEL_N		= 4	        #/*THIS MUST BE LAST		*/


  # --- Basic operations ------------------------------------------------------
  hid_t     H5Screate(H5S_class_t type) except *
  hid_t     H5Scopy(hid_t space_id  ) except *
  herr_t    H5Sclose(hid_t space_id) except *

  # --- Simple dataspace operations -------------------------------------------
  hid_t     H5Screate_simple(int rank, hsize_t dims[], hsize_t maxdims[]) except *
  htri_t    H5Sis_simple(hid_t space_id) except *
  herr_t    H5Soffset_simple(hid_t space_id, hssize_t *offset  ) except *

  int       H5Sget_simple_extent_ndims(hid_t space_id) except *
  int       H5Sget_simple_extent_dims(hid_t space_id, hsize_t dims[], hsize_t maxdims[]) except *
  hssize_t  H5Sget_simple_extent_npoints(hid_t space_id) except *
  H5S_class_t H5Sget_simple_extent_type(hid_t space_id) except *

  # --- Extents ---------------------------------------------------------------
  herr_t    H5Sextent_copy(hid_t dest_space_id, hid_t source_space_id  ) except *
  herr_t    H5Sset_extent_simple(hid_t space_id, int rank, 
                hsize_t *current_size, hsize_t *maximum_size  ) except *
  herr_t    H5Sset_extent_none(hid_t space_id) except *

  # --- Dataspace selection ---------------------------------------------------
  H5S_sel_type H5Sget_select_type(hid_t space_id) except *
  hssize_t  H5Sget_select_npoints(hid_t space_id) except *
  herr_t    H5Sget_select_bounds(hid_t space_id, hsize_t *start, hsize_t *end) except *

  herr_t    H5Sselect_all(hid_t space_id) except *
  herr_t    H5Sselect_none(hid_t space_id) except *
  htri_t    H5Sselect_valid(hid_t space_id) except *

  hssize_t  H5Sget_select_elem_npoints(hid_t space_id) except *
  herr_t    H5Sget_select_elem_pointlist(hid_t space_id, hsize_t startpoint, 
                hsize_t numpoints, hsize_t *buf) except *
  herr_t    H5Sselect_elements(hid_t space_id, H5S_seloper_t op, 
                size_t num_elements, hsize_t **coord) except *

  hssize_t  H5Sget_select_hyper_nblocks(hid_t space_id  ) except *
  herr_t    H5Sget_select_hyper_blocklist(hid_t space_id, 
                hsize_t startblock, hsize_t numblocks, hsize_t *buf  ) except *
  herr_t H5Sselect_hyperslab(hid_t space_id, H5S_seloper_t op,
                             hsize_t start[], hsize_t _stride[],
                             hsize_t count[], hsize_t _block[]) except *







