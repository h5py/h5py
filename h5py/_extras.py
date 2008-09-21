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

import logging
from h5py import config

# Decorator utility for low-level thread safety
from functools import update_wrapper

def uw_apply(wrap, func):
    # Cython methods don't have a "module" attribute for some reason
    if hasattr(func, '__module__'):
        update_wrapper(wrap, func)
    else:
        update_wrapper(wrap, func, assigned=('__name__','__doc__'))

def h5sync(logger=None):

    if logger is None:
        def sync_simple(func):
            
            def wrap(*args, **kwds):
                with config.lock:
                    return func(*args, **kwds)

            uw_apply(wrap, func)
            return wrap

        return sync_simple

    else:

        def sync_debug(func):

            def wrap(*args, **kwds):
                logger.debug("$ Threadsafe function entry: %s" % func.__name__)
                with config.lock:
                    logger.debug("> Acquired lock on %s" % func.__name__)
                    retval = func(*args, **kwds)
                logger.debug("< Released lock on %s" % func.__name__)
                return retval

            uw_apply(wrap, func)
            return wrap

        return sync_debug

def h5sync_dummy(logger):

    def log_only(func):

        def wrap(*args, **kwds):
            logger.debug("$ Function entry: %s" % func.__name__)
            return func(*args, **kwds)

        uw_apply(wrap, func)
        return wrap

    return log_only





