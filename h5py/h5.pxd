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

include "std_defs.pxi"

# === Custom C extensions =====================================================

cdef int _enable_exceptions() except -1
cdef int _disable_exceptions() except -1

cdef object standard_richcmp(object self, object other, int how)

cdef class PHIL:

    cdef object lock

    cpdef bint __enter__(self) except -1
    cpdef bint __exit__(self, a, b, c) except -1
    cpdef bint acquire(self, int blocking=*) except -1
    cpdef bint release(self) except -1

cpdef PHIL get_phil()

cdef class H5PYConfig:

    # Global 
    cdef object _complex_names      # ('r','i')
    cdef readonly object API_16
    cdef readonly object API_18
    cdef readonly object DEBUG
    cdef readonly object THREADS

cdef class ObjectID:
    """ Base wrapper class for HDF5 object identifiers """
    cdef object __weakref__
    cdef readonly hid_t id
    cdef readonly int _locked
    cdef object _hash           # Used by subclasses to cache a hash value,
                                # which may be expensive to compute.

# === HDF5 API ================================================================

cdef extern from "hdf5.h":

  ctypedef struct hvl_t:
    size_t len                 # Length of VL data (in base type units)
    void *p                    # Pointer to VL data

  int HADDR_UNDEF

  herr_t H5open() except *
  herr_t H5close() except *


  # --- Reflection ------------------------------------------------------------
  ctypedef enum H5I_type_t:
    H5I_BADID        = -1

  H5I_type_t H5Iget_type(hid_t obj_id) except *
  int        H5Idec_ref(hid_t obj_id) except *
  int        H5Iget_ref(hid_t obj_id) except *
  int        H5Iinc_ref(hid_t obj_id) except *

  # --- Version functions -----------------------------------------------------
  herr_t H5get_libversion(unsigned *majnum, unsigned *minnum,
                          unsigned *relnum ) except *

  # --- Error handling --------------------------------------------------------

  # Major error numbers
  ctypedef enum H5E_major_t:
    H5E_NONE_MAJOR       = 0,   # special zero, no error                     
    H5E_ARGS,                   # invalid arguments to routine               
    H5E_RESOURCE,               # resource unavailable                       
    H5E_INTERNAL,               #  Internal error (too specific to document)
    H5E_FILE,                   # file Accessability                         
    H5E_IO,                     # Low-level I/O                              
    H5E_FUNC,                   # function Entry/Exit                        
    H5E_ATOM,                   # object Atom                                
    H5E_CACHE,                  # object Cache                               
    H5E_BTREE,                  # B-Tree Node                                
    H5E_SYM,                    # symbol Table                               
    H5E_HEAP,                   # Heap                                       
    H5E_OHDR,                   # object Header                              
    H5E_DATATYPE,               # Datatype                                   
    H5E_DATASPACE,              # Dataspace                                  
    H5E_DATASET,                # Dataset                                    
    H5E_STORAGE,                # data storage                               
    H5E_PLIST,                  # Property lists                             
    H5E_ATTR,                   # Attribute                                  
    H5E_PLINE,                  # Data filters                               
    H5E_EFL,                    # External file list                         
    H5E_REFERENCE,              # References                                 
    H5E_VFL,                    # Virtual File Layer                 
#   H5E_TBBT,                   # Threaded, Balanced, Binary Trees (removed)
    H5E_TST,                    # Ternary Search Trees                       
    H5E_RS,                     # Reference Counted Strings                  
    H5E_ERROR,                  # Error API                                  
    H5E_SLIST                   # Skip Lists                                 

  ctypedef enum H5E_minor_t:
    pass

  cdef enum H5E_direction_t:
    H5E_WALK_UPWARD    = 0  # begin deep, end at API function    
    H5E_WALK_DOWNWARD = 1   # begin at API function, end deep    

  ctypedef struct H5E_error_t:
    H5E_major_t     maj_num        #  major error number             
    H5E_minor_t     min_num        #  minor error number             
    char    *func_name      #  function in which error occurred   
    char    *file_name      #  file in which error occurred       
    unsigned    line        #  line in file where error occurs    
    char    *desc           #  optional supplied description      

  char      *H5Eget_major(H5E_major_t n)
  char      *H5Eget_minor(H5E_minor_t n)
  herr_t    H5Eclear() except *
  ctypedef herr_t (*H5E_auto_t)(void *client_data)
  herr_t    H5Eset_auto(H5E_auto_t func, void *client_data)
  herr_t    H5Eget_auto(H5E_auto_t *func, void** client_data)
  ctypedef herr_t (*H5E_walk_t)(int n, H5E_error_t *err_desc, void* client_data)  
  herr_t    H5Ewalk(H5E_direction_t direction, H5E_walk_t func, void* client_data  )

  int       H5Fget_obj_count(hid_t file_id, unsigned int types) except *
  int       H5Fget_obj_ids(hid_t file_id, unsigned int types, int max_objs, hid_t *obj_id_list) except *
  int       H5F_OBJ_ALL






















