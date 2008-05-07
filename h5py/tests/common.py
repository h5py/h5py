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

import tempfile
import os
import shutil
from h5py import h5f, h5p

def getcopy(filename):
    """ Create a temporary working copy of "filename". Return is a 2-tuple
        containing (HDF5 file id, file name)
    """
    newname = tempfile.mktemp('.hdf5')
    shutil.copy(filename, newname)

    plist = h5p.create(h5p.CLASS_FILE_ACCESS)
    h5p.set_fclose_degree(plist, h5f.CLOSE_STRONG)
    fid = h5f.open(newname, h5f.ACC_RDWR)
    h5p.close(plist)

    return (fid, newname)

def deletecopy(fid, newname):
    h5f.close(fid)
    os.unlink(newname)

def errstr(arg1, arg2, msg=''):
    """ Used to mimic assertEqual-style auto-repr, where assertEqual doesn't
        work (i.e. for Numpy arrays where all() must be used)
    """
    return msg+'%s != %s' % (repr(arg1), repr(arg2))

