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

import unittest
import os
from numpy import all

import h5py
from h5py.highlevel import *

# --- Description of the PyTables test file smpl_compound_chunked -------------

HDFNAME2 = os.path.join(os.path.dirname(h5py.__file__), 'tests/data/smpl_compound_chunked.hdf5')
DTYPE = numpy.dtype([('a_name','>i4'),
                     ('c_name','|S6'),
                     ('d_name', numpy.dtype( ('>i2', (5,10)) )),
                     ('e_name', '>f4'),
                     ('f_name', numpy.dtype( ('>f8', (10,)) )),
                     ('g_name', '<u1')])
SHAPE = (6,)

basearray = numpy.ndarray(SHAPE, dtype=DTYPE)
for i in range(SHAPE[0]):
    basearray[i]["a_name"] = i,
    basearray[i]["c_name"] = "Hello!"
    basearray[i]["d_name"][:] = numpy.sum(numpy.indices((5,10)),0) + i # [:] REQUIRED for some stupid reason
    basearray[i]["e_name"] = 0.96*i
    basearray[i]["f_name"][:] = numpy.array((1024.9637*i,)*10)
    basearray[i]["g_name"] = 109

names = ("a_name","c_name","d_name","e_name","f_name","g_name")

class TestHighlevel(unittest.TestCase):


    def test_ds(self):
        myfile = File(HDFNAME2,'r')
        try:
            ds = myfile["CompoundChunked"]
            for i in range(6):
                self.assert_(all(ds[i]==basearray[i]), "%d"%i)
            for name in names:
                self.assert_(all(ds[name]==basearray[name]))
        finally:
            myfile.close()
            




