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

# Common, stateless code safe for inclusion in each .pyx file.  Also brings
# in config.pxi.

include "config.pxi"

# Set up synchronization decorator for threads
IF H5PY_THREADS:
    from _extras import h5sync
    IF H5PY_DEBUG:
        import logging
        sync = h5sync(logging.getLogger('h5py.functions'))
    ELSE:
        sync = h5sync()
ELSE:
    IF H5PY_DEBUG:
        import logging
        from _extras import h5sync_dummy
        sync = h5sync_dummy(logging.getLogger('h5py.functions'))
    ELSE:
        cdef inline object sync(object func):
            return func
