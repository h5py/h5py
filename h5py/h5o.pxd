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

include "std_defs.pxi"

from h5 cimport ObjectID
from h5g cimport GroupID

cdef extern from "hdf5.h":

  ctypedef enum H5O_type_t:
    H5O_TYPE_UNKNOWN = -1,      #	Unknown object type		
    H5O_TYPE_GROUP,	            #   Object is a group		
    H5O_TYPE_DATASET,		    #   Object is a dataset		
    H5O_TYPE_NAMED_DATATYPE,    #   Object is a named data type	
    H5O_TYPE_NTYPES             #   Number of different object types (must be last!) 

  # --- Components for the H5O_info_t struct ----------------------------------

  cdef struct space:
    hsize_t total           #  Total space for storing object header in file 
    hsize_t meta            #  Space within header for object header metadata information 
    hsize_t mesg            #  Space within header for actual message information 
    hsize_t free            #  Free space within object header 

  cdef struct mesg:
    unsigned long present   #  Flags to indicate presence of message type in header 
    unsigned long shared    #  Flags to indicate message type is shared in header 

  cdef struct hdr:
    unsigned version        #  Version number of header format in file 
    unsigned nmesgs         #  Number of object header messages 
    unsigned nchunks        #  Number of object header chunks 
    unsigned flags          #  Object header status flags 
    space space
    mesg mesg

  ctypedef struct H5_ih_info_t:
    hsize_t     index_size,  # /* btree and/or list */
    hsize_t     heap_size

  cdef struct meta_size:
    H5_ih_info_t   obj,    #        /* v1/v2 B-tree & local/fractal heap for groups, B-tree for chunked datasets */
    H5_ih_info_t   attr    #        /* v2 B-tree & heap for attributes */

  ctypedef struct H5O_info_t:
    unsigned long   fileno         #  File number that object is located in 
    haddr_t         addr           #  Object address in file    
    H5O_type_t      type           #  Basic object type (group, dataset, etc.) 
    unsigned         rc            #  Reference count of object    
    time_t        atime            #  Access time            
    time_t        mtime            #  Modification time        
    time_t        ctime            #  Change time            
    time_t        btime            #  Birth time            
    hsize_t         num_attrs      #  # of attributes attached to object 
    hdr           hdr
    meta_size     meta_size

  ctypedef enum H5_index_t:
    H5_INDEX_NAME,
    H5_INDEX_CRT_ORDER

  ctypedef enum H5_iter_order_t:
     H5_ITER_INC,      # Increasing order
     H5_ITER_DEC,     # Decreasing order
     H5_ITER_NATIVE  # Fastest available order

  ctypedef herr_t (*H5O_iterate_t)(hid_t obj, char *name, H5O_info_t *info,
                    void *op_data)

  herr_t H5Ovisit(hid_t obj_id, H5_index_t idx_type, H5_iter_order_t order,
                    H5O_iterate_t op, void *op_data) except *

  herr_t H5Oget_info(hid_t loc_id, H5O_info_t *oinfo) except *








