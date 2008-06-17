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

# h5py module __init__

__doc__ = \
"""
    This is the h5py package, a Python interface to the NCSA HDF5 
    scientific data format.

    Version %s

    See the docstring for the "version" module for a longer introduction.
"""

import utils, h5, h5f, h5g, h5s, h5t, h5d, h5a, h5p, h5z, h5i
import version

__doc__ = __doc__ % version.version
