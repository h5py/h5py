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

    HDF5 %s (using %s API)
"""

import utils, h5, h5a, h5d, h5f, h5g, h5i, h5p, h5r, h5s, h5t, h5z, highlevel

from highlevel import File, Group, Dataset, Datatype, AttributeManager

__doc__ = __doc__ % (h5.version, h5.hdf5_version, h5.api_version)

__all__ = ['h5', 'h5f', 'h5g', 'h5s', 'h5t', 'h5d', 'h5a', 'h5p',
           'h5z', 'h5i', 'File', 'Group', 'Dataset',
           'Datatype', 'AttributeManager']
