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

# Header file which goes as the first line in all .pyx files.  Currently its
# main function is to help simplify the .pxi-based import fiasco.

# hdr_pxd already brought in hdf5 and stdlib defines.  So we just need to
# include the configs, and define things like synchronization decorators.

include "config.pxi"

# Defines the following decorators:
#
# sync:     Acquire PHIL for this function, and log function entry  in
#           debug mode.
# nosync:   Don't acquire PHIL, but log function entry in debug mode.
#

IF H5PY_DEBUG:
    import logging
    from _extras import h5sync_dummy
    nosync = h5sync_dummy(logging.getLogger('h5py.functions'))

    IF H5PY_THREADS:
        from _extras import h5sync
        sync = h5sync(logging.getLogger('h5py.functions'))
    ELSE:
        sync = nosync

ELSE:
    cdef inline object nosync(object func):
        return func

    IF H5PY_THREADS:
        from _extras import h5sync
        sync = h5sync()
    ELSE:
        cdef inline object sync(object func):
            return func
