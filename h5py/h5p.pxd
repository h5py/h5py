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

include "defs.pxd"

from h5 cimport ObjectID

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

cdef class PropDCID(PropInstanceID):
    """ Dataset creation property list """
    pass

cdef class PropDXID(PropInstanceID):
    """ Dataset transfer property list """
    pass

cdef class PropFCID(PropInstanceID):
    """ File creation property list """
    pass

cdef class PropFAID(PropInstanceID):
    """ File access property list """
    pass

cdef hid_t pdefault(PropID pid)
cdef object propwrap(hid_t id_in)



