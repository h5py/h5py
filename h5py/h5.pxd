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

include "conditions.pxi"
from defs_c cimport size_t, ssize_t, malloc, free

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

  # --- Reflection ------------------------------------------------------------
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
    H5E_TBBT,                   # Threaded, Balanced, Binary Trees           
    H5E_TST,                    # Ternary Search Trees                       
    H5E_RS,                     # Reference Counted Strings                  
    H5E_ERROR,                  # Error API                                  
    H5E_SLIST                   # Skip Lists                                 

  # Minor error numbers
  ctypedef enum H5E_minor_t:
    H5E_NONE_MINOR       = 0

    #  Argument errors 
    H5E_UNINITIALIZED,          # information is unitialized
    H5E_UNSUPPORTED,            # feature is unsupported
    H5E_BADTYPE,                # incorrect type found
    H5E_BADRANGE,               # argument out of range
    H5E_BADVALUE,               # bad value for argument

    #  Resource errors 
    H5E_NOSPACE,                # no space available for allocation
    H5E_CANTCOPY,               # unable to copy object
    H5E_CANTFREE,               # unable to free object
    H5E_ALREADYEXISTS,          # Object already exists
    H5E_CANTLOCK,               # Unable to lock object
    H5E_CANTUNLOCK,             # Unable to unlock object
    H5E_CANTGC,                 # Unable to garbage collect
    H5E_CANTGETSIZE,            # Unable to compute size

    #  File accessability errors 
    H5E_FILEEXISTS,             # file already exists                        
    H5E_FILEOPEN,               # file already open                          
    H5E_CANTCREATE,             # Can't create file                          
    H5E_CANTOPENFILE,           # Can't open file                            
    H5E_CANTCLOSEFILE,          # Can't close file                 
    H5E_NOTHDF5,                # not an HDF5 format file                    
    H5E_BADFILE,                # bad file ID accessed                       
    H5E_TRUNCATED,              # file has been truncated                    
    H5E_MOUNT,                    # file mount error                 

    #  Generic low-level file I/O errors
    H5E_SEEKERROR,              # seek failed                                
    H5E_READERROR,              # read failed                                
    H5E_WRITEERROR,             # write failed                               
    H5E_CLOSEERROR,             # close failed                               
    H5E_OVERFLOW,                # address overflowed                 
    H5E_FCNTL,                  # file fcntl failed                          

    #  Function entry/exit interface errors 
    H5E_CANTINIT,               # Can't initialize object                    
    H5E_ALREADYINIT,            # object already initialized                 
    H5E_CANTRELEASE,            # Can't release object                       

    #  Object atom related errors 
    H5E_BADATOM,                # Can't find atom information                
    H5E_BADGROUP,               # Can't find group information               
    H5E_CANTREGISTER,           # Can't register new atom                    
    H5E_CANTINC,                # Can't increment reference count            
    H5E_CANTDEC,                # Can't decrement reference count            
    H5E_NOIDS,                  # Out of IDs for group                       

    #  Cache related errors 
    H5E_CANTFLUSH,              # Can't flush object from cache              
    H5E_CANTSERIALIZE,          # Unable to serialize data from cache        
    H5E_CANTLOAD,               # Can't load object into cache               
    H5E_PROTECT,                # protected object error                     
    H5E_NOTCACHED,              # object not currently cached                
    H5E_SYSTEM,                 # Internal error detected                    
    H5E_CANTINS,                # Unable to insert metadata into cache       
    H5E_CANTRENAME,             # Unable to rename metadata                  
    H5E_CANTPROTECT,            # Unable to protect metadata                 
    H5E_CANTUNPROTECT,          # Unable to unprotect metadata               

    #  B-tree related errors 
    H5E_NOTFOUND,               # object not found                           
    H5E_EXISTS,                 # object already exists                      
    H5E_CANTENCODE,             # Can't encode value                         
    H5E_CANTDECODE,             # Can't decode value                         
    H5E_CANTSPLIT,              # Can't split node                           
    H5E_CANTINSERT,             # Can't insert object                        
    H5E_CANTLIST,               # Can't list node                            

    #  Object header related errors 
    H5E_LINKCOUNT,              # bad object header link count               
    H5E_VERSION,                # wrong version number                       
    H5E_ALIGNMENT,              # alignment error                            
    H5E_BADMESG,                # unrecognized message                       
    H5E_CANTDELETE,             #  Can't delete message                      
    H5E_BADITER,                #  Iteration failed                          

    #  Group related errors 
    H5E_CANTOPENOBJ,            # Can't open object                          
    H5E_CANTCLOSEOBJ,           # Can't close object                         
    H5E_COMPLEN,                # name component is too long                 
    H5E_LINK,                   # link count failure                         
    H5E_SLINK,                    # symbolic link error                 
    H5E_PATH,                    # Problem with path to object             

    #  Datatype conversion errors 
    H5E_CANTCONVERT,            # Can't convert datatypes  TypeError?
    H5E_BADSIZE,                # Bad size for object                        

    #  Dataspace errors 
    H5E_CANTCLIP,               # Can't clip hyperslab region                
    H5E_CANTCOUNT,              # Can't count elements                       
    H5E_CANTSELECT,             # Can't select hyperslab                     
    H5E_CANTNEXT,               # Can't move to next iterator location       
    H5E_BADSELECT,              # Invalid selection                          
    H5E_CANTCOMPARE,            # Can't compare objects                      

    #  Property list errors 
    H5E_CANTGET,                # Can't get value                            
    H5E_CANTSET,                # Can't set value                            
    H5E_DUPCLASS,               # Duplicate class name in parent class       

    #  Parallel errors 
    H5E_MPI,                    # some MPI function failed             
    H5E_MPIERRSTR,                # MPI Error String                  

    #  Heap errors 
    H5E_CANTRESTORE,            # Can't restore condition                    

    #  TBBT errors 
    H5E_CANTMAKETREE,            # Can't create TBBT tree                     

    #  I/O pipeline errors 
    H5E_NOFILTER,               # requested filter is not available          
    H5E_CALLBACK,               # callback failed                            
    H5E_CANAPPLY,               # error from filter "can apply" callback     
    H5E_SETLOCAL,               # error from filter "set local" callback     
    H5E_NOENCODER,              #  Filter present, but encoding disabled     

    #  System level errors 
    H5E_SYSERRSTR               #  System error message                 

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

  # --- Error handling --------------------------------------------------------

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

# === Custom C extensions =====================================================

ctypedef H5E_auto_t err_c
cdef int _enable_exceptions() except -1
cdef int _disable_exceptions() except -1

cdef err_c pause_errors() except? NULL
cdef int resume_errors(err_c cookie) except -1

# === Custom identifier wrappers ==============================================

cdef class ObjectID:
    """ Base wrapper class for HDF5 object identifiers """
    cdef object __weakref__
    cdef readonly hid_t id
    cdef readonly int _locked





















