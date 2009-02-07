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

def runtests():
    try:
        import nose
    except ImportError:
        raise ImportError("python-nose is required to run unit tests")
    nose.run('h5py.tests')

def autotest():
    try:
        if not runtests():
            sys.exit(17)
    except:
        sys.exit(2)



