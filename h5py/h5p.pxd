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

from defs_c   cimport size_t, time_t
from h5  cimport hid_t, hbool_t, herr_t, htri_t, hsize_t, hssize_t, hvl_t
from h5d cimport H5D_layout_t, H5D_fill_value_t, H5D_fill_time_t, H5D_alloc_time_t
from h5z cimport H5Z_filter_t, H5Z_EDC_t
from h5f cimport H5F_close_degree_t

cdef extern from "hdf5.h":

  int H5P_DEFAULT

  # Property list classes
  int H5P_NO_CLASS
  int H5P_FILE_CREATE 
  int H5P_FILE_ACCESS 
  int H5P_DATASET_CREATE 
  int H5P_DATASET_XFER 

  # --- Property list operations ----------------------------------------------
  # General operations
  hid_t  H5Pcreate(hid_t plist_id)
  hid_t  H5Pcopy(hid_t plist_id)
  int    H5Pget_class(hid_t plist_id)
  herr_t H5Pclose(hid_t plist_id)
  htri_t H5Pequal( hid_t id1, hid_t id2  )

  # File creation properties
  herr_t H5Pget_version(hid_t plist, unsigned int *super_, unsigned int* freelist, 
                        unsigned int *stab, unsigned int *shhdr)
  herr_t H5Pset_userblock(hid_t plist, hsize_t size)
  herr_t H5Pget_userblock(hid_t plist, hsize_t * size)
  herr_t H5Pset_sizes(hid_t plist, size_t sizeof_addr, size_t sizeof_size)
  herr_t H5Pget_sizes(hid_t plist, size_t *sizeof_addr, size_t *sizeof_size)
  herr_t H5Pset_sym_k(hid_t plist, unsigned int ik, unsigned int lk)
  herr_t H5Pget_sym_k(hid_t plist, unsigned int *ik, unsigned int *lk)
  herr_t H5Pset_istore_k(hid_t plist, unsigned int ik)
  herr_t H5Pget_istore_k(hid_t plist, unsigned int *ik  )

  # File access
  herr_t    H5Pset_fclose_degree(hid_t fapl_id, H5F_close_degree_t fc_degree)
  herr_t    H5Pget_fclose_degree(hid_t fapl_id, H5F_close_degree_t *fc_degree)
  herr_t    H5Pset_fapl_core( hid_t fapl_id, size_t increment, hbool_t backing_store)
  herr_t    H5Pget_fapl_core( hid_t fapl_id, size_t *increment, hbool_t *backing_store)
  herr_t    H5Pset_fapl_family ( hid_t fapl_id,  hsize_t memb_size, hid_t memb_fapl_id  )
  herr_t    H5Pget_fapl_family ( hid_t fapl_id, hsize_t *memb_size, hid_t *memb_fapl_id  )
  herr_t    H5Pset_family_offset ( hid_t fapl_id, hsize_t offset)
  herr_t    H5Pget_family_offset ( hid_t fapl_id, hsize_t *offset)
  herr_t    H5Pset_fapl_log(hid_t fapl_id, char *logfile, unsigned int flags, size_t buf_size)

  # Dataset creation properties
  herr_t        H5Pset_layout(hid_t plist, H5D_layout_t layout )
  H5D_layout_t  H5Pget_layout(hid_t plist)
  herr_t        H5Pset_chunk(hid_t plist, int ndims, hsize_t * dim)
  int           H5Pget_chunk(hid_t plist, int max_ndims, hsize_t * dims  )
  herr_t        H5Pset_deflate( hid_t plist, int level)
  herr_t        H5Pset_fill_value(hid_t plist_id, hid_t type_id, void *value  )
  herr_t        H5Pget_fill_value(hid_t plist_id, hid_t type_id, void *value  )
  herr_t        H5Pfill_value_defined(hid_t plist_id, H5D_fill_value_t *status  )
  herr_t        H5Pset_fill_time(hid_t plist_id, H5D_fill_time_t fill_time  )
  herr_t        H5Pget_fill_time(hid_t plist_id, H5D_fill_time_t *fill_time  )
  herr_t        H5Pset_alloc_time(hid_t plist_id, H5D_alloc_time_t alloc_time  )
  herr_t        H5Pget_alloc_time(hid_t plist_id, H5D_alloc_time_t *alloc_time  )
  herr_t        H5Pset_filter(hid_t plist, H5Z_filter_t filter, unsigned int flags,
                              size_t cd_nelmts, unsigned int cd_values[]  )
  htri_t        H5Pall_filters_avail(hid_t dcpl_id)
  int           H5Pget_nfilters(hid_t plist)
  H5Z_filter_t  H5Pget_filter(hid_t plist, unsigned int filter_number, 
                              unsigned int *flags, size_t *cd_nelmts, 
                              unsigned int *cd_values, size_t namelen, char name[]  )
  herr_t        H5Pget_filter_by_id( hid_t plist_id, H5Z_filter_t filter, 
                                     unsigned int *flags, size_t *cd_nelmts, 
                                     unsigned int cd_values[], size_t namelen, char name[]  )
  herr_t        H5Pmodify_filter(hid_t plist, H5Z_filter_t filter, unsigned int flags,
                                 size_t cd_nelmts, unsigned int cd_values[]  )
  herr_t        H5Premove_filter(hid_t plist, H5Z_filter_t filter  )
  herr_t        H5Pset_fletcher32(hid_t plist)
  herr_t        H5Pset_shuffle(hid_t plist_id)
  herr_t        H5Pset_szip(hid_t plist, unsigned int options_mask, unsigned int pixels_per_block)


  # Transfer properties
  herr_t    H5Pset_edc_check(hid_t plist, H5Z_EDC_t check)
  H5Z_EDC_t H5Pget_edc_check(hid_t plist)

  # Other properties
  herr_t H5Pset_cache(hid_t plist_id, int mdc_nelmts, int rdcc_nelmts,
                      size_t rdcc_nbytes, double rdcc_w0)
  herr_t H5Pset_sieve_buf_size(hid_t fapl_id, hsize_t size)
  herr_t H5Pset_fapl_log(hid_t fapl_id, char *logfile,
                         unsigned int flags, size_t buf_size)


