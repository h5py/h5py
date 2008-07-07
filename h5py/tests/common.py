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

class HCopy(object):

    """
        Use:

        from __future__ import with_statement

        with HCopy(filename) as fid:
            fid.frob()
            obj = h5g.open(fid, whatever)
            ...
    """
    def __init__(self, filename):
        self.filename = filename
        self.tmpname = None

    def __enter__(self):
        self.tmpname = tempfile.mktemp('.hdf5')
        shutil.copy(self.filename, self.tmpname)

        plist = h5p.create(h5p.FILE_ACCESS)
        plist.set_fclose_degree(h5f.CLOSE_STRONG)
        self.fid = h5f.open(self.tmpname, h5f.ACC_RDWR)
        return self.fid

    def __exit__(self, *args):
        self.fid.close()
        os.unlink(self.tmpname)

def errstr(arg1, arg2, msg=''):
    """ Used to mimic assertEqual-style auto-repr, where assertEqual doesn't
        work (i.e. for Numpy arrays where all() must be used)
    """
    return msg+'%s != %s' % (repr(arg1), repr(arg2))

