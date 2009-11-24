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

cdef class PropLCID(PropCreateID):
    
    """ Link creation property list """

    
    def set_create_intermediate_group(self, bint create):
        """(BOOL create)

        Set whether missing intermediate groups are automatically created.
        """
        H5Pset_create_intermediate_group(self.id, create)

    
    def get_create_intermediate_group(self):
        """() => BOOL 

        Determine if missing intermediate groups are automatically created.
        """
        cdef unsigned int create
        H5Pget_create_intermediate_group(self.id, &create)
        return <bint>create
