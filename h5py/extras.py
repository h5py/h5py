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
from __future__ import with_statement

from h5py import config

# Decorator utility for threads
from functools import update_wrapper
def h5sync(func):
    
    def wrap(*args, **kwds):
        with config.lock:
            return func(*args, **kwds)

    update_wrapper(wrap, func)
    return wrap

