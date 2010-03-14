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

cdef class PropFAID(PropInstanceID):

    """
        File access property list
    """

    
    def set_fclose_degree(self, int close_degree):
        """(INT close_degree)

        Set the file-close degree, which determines library behavior when
        a file is closed when objects are still open.  Legal values:

        * h5f.CLOSE_WEAK
        * h5f.CLOSE_SEMI
        * h5f.CLOSE_STRONG
        * h5f.CLOSE_DEFAULT
        """
        H5Pset_fclose_degree(self.id, <H5F_close_degree_t>close_degree)

    
    def get_fclose_degree(self):
        """() => INT close_degree
        - h5fd.
        Get the file-close degree, which determines library behavior when
        a file is closed when objects are still open.  Legal values:

        * h5f.CLOSE_WEAK
        * h5f.CLOSE_SEMI
        * h5f.CLOSE_STRONG
        * h5f.CLOSE_DEFAULT
        """
        cdef H5F_close_degree_t deg
        H5Pget_fclose_degree(self.id, &deg)
        return deg

    
    def set_fapl_core(self, size_t block_size=64*1024, hbool_t backing_store=1):
        """(UINT increment=64k, BOOL backing_store=True)

        Use the h5fd.CORE (memory-resident) file driver.

        increment
            Chunk size for new memory requests (default 1 meg)

        backing_store
            If True (default), memory contents are associated with an
            on-disk file, which is updated when the file is closed.
            Set to False for a purely in-memory file.
        """
        H5Pset_fapl_core(self.id, block_size, backing_store)

    
    def get_fapl_core(self):
        """() => TUPLE core_settings

        Determine settings for the h5fd.CORE (memory-resident) file driver.
        Tuple elements are:

        0. UINT "increment": Chunk size for new memory requests
        1. BOOL "backing_store": If True, write the memory contents to 
           disk when the file is closed.
        """
        cdef size_t increment
        cdef hbool_t backing_store
        H5Pget_fapl_core(self.id, &increment, &backing_store)
        return (increment, <bint>(backing_store))

    
    def set_fapl_family(self, hsize_t memb_size=2147483647, PropID memb_fapl=None):
        """(UINT memb_size=2**31-1, PropFAID memb_fapl=None)

        Set up the family driver.

        memb_size
            Member file size

        memb_fapl
            File access property list for each member access
        """
        cdef hid_t plist_id
        plist_id = pdefault(memb_fapl)
        H5Pset_fapl_family(self.id, memb_size, plist_id)

    
    def get_fapl_family(self):
        """() => TUPLE info

        Determine family driver settings. Tuple values are:

        0. UINT memb_size
        1. PropFAID memb_fapl or None
        """
        cdef hid_t mfapl_id
        cdef hsize_t msize
        cdef PropFAID plist
        plist = None

        H5Pget_fapl_family(self.id, &msize, &mfapl_id)

        if mfapl_id > 0:
            plist = PropFAID(mfapl_id)

        return (msize, plist)

    
    def set_fapl_log(self, char* logfile, unsigned int flags, size_t buf_size):
        """(STRING logfile, UINT flags, UINT buf_size)

        Enable the use of the logging driver.  See the HDF5 documentation
        for details.  Flag constants are stored in module h5fd.
        """
        H5Pset_fapl_log(self.id, logfile, flags, buf_size)

    
    def set_fapl_sec2(self):
        """()

        Select the "section-2" driver (h5fd.SEC2).
        """
        H5Pset_fapl_sec2(self.id)

    
    def set_fapl_stdio(self):
        """()

        Select the "stdio" driver (h5fd.STDIO)
        """
        H5Pset_fapl_stdio(self.id)

    
    def get_driver(self):
        """() => INT driver code

        Return an integer identifier for the driver used by this list.
        Although HDF5 implements these as full-fledged objects, they are
        treated as integers by Python.  Built-in drivers identifiers are
        listed in module h5fd; they are:

        - h5fd.CORE
        - h5fd.FAMILY
        - h5fd.LOG
        - h5fd.MPIO
        - h5fd.MULTI
        - h5fd.SEC2
        - h5fd.STDIO
        """
        return H5Pget_driver(self.id)

    
    def set_cache(self, int mdc, int rdcc, size_t rdcc_nbytes, double rdcc_w0):
        """(INT mdc, INT rdcc, UINT rdcc_nbytes, DOUBLE rdcc_w0)

        Set the metadata (mdc) and raw data chunk (rdcc) cache properties.
        See the HDF5 docs for a full explanation.
        """
        H5Pset_cache(self.id, mdc, rdcc, rdcc_nbytes, rdcc_w0)

    
    def get_cache(self):
        """() => TUPLE cache info

        Get the metadata and raw data chunk cache settings.  See the HDF5
        docs for element definitions.  Return is a 4-tuple with entries:

        1. INT mdc:              Number of metadata objects
        2. INT rdcc:             Number of raw data chunks
        3. UINT rdcc_nbytes:     Size of raw data cache
        4. DOUBLE rdcc_w0:       Preemption policy for data cache.
        """
        cdef int mdc, rdcc
        cdef size_t rdcc_nbytes
        cdef double w0

        H5Pget_cache(self.id, &mdc, &rdcc, &rdcc_nbytes, &w0)
        return (mdc, rdcc, rdcc_nbytes, w0)

    
    def set_sieve_buf_size(self, size_t size):
        """ (UINT size)

        Set the maximum size of the data sieve buffer (in bytes).  This
        buffer can improve I/O performance for hyperslab I/O, by combining
        reads and writes into blocks of the given size.  The default is 64k.
        """
        H5Pset_sieve_buf_size(self.id, size)

    
    def get_sieve_buf_size(self):
        """ () => UINT size

        Get the current maximum size of the data sieve buffer (in bytes).
        """
        cdef size_t size
        H5Pget_sieve_buf_size(self.id, &size)
        return size





