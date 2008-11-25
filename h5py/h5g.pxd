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

include "config.pxi"
include "defs.pxd"

from h5 cimport ObjectID

cdef class GroupID(ObjectID):

    IF H5PY_18API:
        cdef readonly object links
    pass


