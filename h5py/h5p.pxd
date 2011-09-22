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

from defs cimport *

from _objects cimport ObjectID

# --- Base classes ---

cdef class PropID(ObjectID):
    """ Base class for all property lists """
    pass

cdef class PropClassID(PropID):
    """ Represents an HDF5 property list class.  These can be either (locked)
        library-defined classes or user-created classes.
    """
    pass

cdef class PropInstanceID(PropID):
    """ Represents an instance of a property list class (i.e. an actual list
        which can be passed on to other API functions).
    """
    pass

cdef class PropCreateID(PropInstanceID):
    """ Base class for all object creation lists.

        Also includes string character set methods.
    """
    pass

cdef class PropCopyID(PropInstanceID):
    """ Property list for copying objects (as in h5o.copy) """

# --- Object creation ---

cdef class PropDCID(PropCreateID):
    """ Dataset creation property list """
    pass

cdef class PropFCID(PropCreateID):
    """ File creation property list """
    pass


# --- Object access ---

cdef class PropFAID(PropInstanceID):
    """ File access property list """
    pass

cdef class PropDXID(PropInstanceID):
    """ Dataset transfer property list """
    pass


# --- New in 1.8 ---

cdef class PropLCID(PropCreateID):
    """ Link creation property list """
    pass

cdef class PropLAID(PropInstanceID):
    """ Link access property list """
    cdef char* _buf

cdef class PropGCID(PropCreateID):
    """ Group creation property list """
    pass

cdef hid_t pdefault(PropID pid)
cdef object propwrap(hid_t id_in)



