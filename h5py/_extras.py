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
from functools import update_wrapper

from h5py.h5 import get_phil
phil = get_phil()

prof = {}
from time import time

def uw_apply(wrap, func):
    # Cython methods don't have a "module" attribute for some reason
    if hasattr(func, '__module__'):
        update_wrapper(wrap, func, assigned=('__module__', '__name__', '__doc__'))
    else:
        update_wrapper(wrap, func, assigned=('__name__','__doc__'))

def funcname(func):

    if hasattr(func, '__module__') and func.__module__ is not None:
        fullname = "%s.%s" % (func.__module__, func.__name__)
    elif hasattr(func, '__self__'):
        fullname = "%s.%s" % (func.__self__.__class__.__name__, func.__name__)
    else:
        fullname = func.__name__

    return fullname


def h5sync(logger=None):

    if logger is None:
        def sync_simple(func):
            
            def wrap(*args, **kwds):
                with phil:
                    return func(*args, **kwds)
            uw_apply(wrap, func)
            return wrap

        return sync_simple

    else:

        def sync_debug(func):

            fname = funcname(func)

            def wrap(*args, **kwds):
                with phil:
                    logger.debug( ("[ Call %s\n%s\n%s" % (fname, args, kwds)).replace("\n", "\n  ") )
                    stime = time()
                    try:
                        retval = func(*args, **kwds)
                    except Exception, e:
                        logger.debug('! Exception in %s: %s("%s")' % (fname, e.__class__.__name__, e))
                        raise
                    otime = time()
                    logger.debug( ("] Exit %s\n%s" % (fname,retval)).replace("\n", "\n  ") )
                    prof.setdefault(repr(func), set()).add(otime-stime)
                    return retval

            uw_apply(wrap, func)
            return wrap

        return sync_debug

def h5sync_dummy(logger):

    def log_only(func):

        def wrap(*args, **kwds):
            logger.debug("[ Function entry: %s" % func.__name__)
            try:
                retval = func(*args, **kwds)
            except:
                logger.debug("! Exception in %s" % func.__name__)
                raise
            logger.debug("] Function exit: %s" % func.__name__)
            return retval
        uw_apply(wrap, func)
        return wrap

    return log_only





