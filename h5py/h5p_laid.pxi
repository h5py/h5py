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

cdef class PropLAID(PropInstanceID):

    """ Link access property list """

    def __cinit__(self, *args):
        self._buf = NULL

    def __dealloc__(self):
        efree(self._buf)

    
    def set_nlinks(self, size_t nlinks):
        """(UINT nlinks)

        Set the maximum traversal depth for soft links
        """
        H5Pset_nlinks(self.id, nlinks)

    
    def get_nlinks(self):
        """() => UINT

        Get the maximum traversal depth for soft links
        """
        cdef size_t nlinks
        H5Pget_nlinks(self.id, &nlinks)
        return nlinks

    
    def set_elink_prefix(self, char* prefix):
        """(STRING prefix)

        Set the external link prefix.
        """
        cdef size_t size

        # HDF5 requires that we hang on to this buffer
        efree(self._buf)
        size = strlen(prefix)
        self._buf = <char*>emalloc(size+1)
        strcpy(self._buf, prefix)

        H5Pset_elink_prefix(self.id, self._buf)

    
    def get_elink_prefix(self):
        """() => STRING prefix

        Get the external link prefix
        """
        cdef char* buf = NULL
        cdef ssize_t size

        size = H5Pget_elink_prefix(self.id, NULL, 0)
        buf = <char*>emalloc(size+1)
        try:
            H5Pget_elink_prefix(self.id, buf, size+1)
            pstr = buf
        finally:
            efree(buf)

        return pstr

    def set_elink_fapl(self, PropID fapl not None):
        """ (PropFAID fapl)

        Set the file access property list used when opening external files.
        """
        H5Pset_elink_fapl(self.id, fapl.id)

    def get_elink_fapl(self):
        """ () => PropFAID fapl

        Get the file access property list used when opening external files.
        """
        return propwrap(H5Pget_elink_fapl(self.id))

