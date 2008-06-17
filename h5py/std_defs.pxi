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

# "Boilerplate" includes which are so common I don't want to repeat them
# in every file.  These include all basic HDF5 and C typedefs.
# This file is designed to be included in *.pxd files; only definitions
# are allowed.

from h5 cimport hid_t, hbool_t, herr_t, htri_t, hsize_t, \
                hssize_t, haddr_t, hvl_t

from defs_c cimport size_t, time_t, ssize_t
