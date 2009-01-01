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

from h5py.h5 import get_phil, get_config

phil = get_phil()
config = get_config()
logger = logging.getLogger('h5py.functions')

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

if config.DEBUG:

    def nosync(func):

        fname = funcname(func)

        def wrap(*args, **kwds):
            logger.debug( ("[ Call %s\n%s\n%s" % (fname, args, kwds)).replace("\n", "\n  ") )
            try:
                retval = func(*args, **kwds)
            except BaseException, e:
                logger.debug('! Exception in %s: %s("%s")' % (fname, e.__class__.__name__, e))
                raise
            logger.debug( ("] Exit %s\n%s" % (fname,retval)).replace("\n", "\n  ") )
            return retval
        uw_apply(wrap, func)
        return wrap

    def sync(func):

        func = nosync(func)  #enables logging and updates __doc__, etc.

        def wrap(*args, **kwds):
            with phil:
                return func(*args, **kwds)
        return wrap

else:

    def sync(func):
        def wrap(*args, **kwds):
            with phil:
                return func(*args, **kwds)
        uw_apply(wrap, func)
        return wrap

    def nosync(func):
        def wrap(*args, **kwds):
            return func(*args, **kwds)
        uw_apply(wrap, func)
        return wrap





















