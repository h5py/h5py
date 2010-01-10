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
import numpy

from h5py import *

HDFNAME = 'attributes.hdf5'

TYPES = {h5p.FILE_CREATE: h5p.PropFCID,
         h5p.FILE_ACCESS: h5p.PropFAID,
         h5p.DATASET_CREATE: h5p.PropDCID,
         h5p.DATASET_XFER: h5p.PropDXID }

class TestH5P(unittest.TestCase):


    def test_create_get_class(self):
        for typecode, cls in TYPES.iteritems():
            instance = h5p.create(typecode)
            self.assertEqual(type(instance), cls)
            self.assert_(instance.get_class().equal(typecode))
        

class TestFCID(unittest.TestCase):

    def setUp(self):
        self.p = h5p.create(h5p.FILE_CREATE)

    def test_version(self):
        vers = self.p.get_version()
        self.assertEqual(len(vers), 4)

    def test_userblock(self):
        for size in (512,1024,2048):
            self.p.set_userblock(size)
            self.assertEqual(self.p.get_userblock(), size)
    
    def test_sizes(self):
        sizes = [(2,4), (8,16)]
        for a, s in sizes:
            self.p.set_sizes(a,s)
            self.assertEqual(self.p.get_sizes(), (a,s))

class TestFAID(unittest.TestCase):

    CLOSE_DEGREES = (h5f.CLOSE_WEAK,
                     h5f.CLOSE_SEMI,
                     h5f.CLOSE_STRONG,
                     h5f.CLOSE_DEFAULT)

    def setUp(self):
        self.p = h5p.create(h5p.FILE_ACCESS)

    def test_fclose_degree(self):
        for deg in self.CLOSE_DEGREES:
            self.p.set_fclose_degree(deg)
            self.assertEqual(self.p.get_fclose_degree(), deg)

    def test_fapl_core(self):
        settings = (2*1024*1024, 1)
        self.p.set_fapl_core(*settings)
        self.assertEqual(self.p.get_fapl_core(), settings)

    def test_sieve(self):
        self.p.get_sieve_buf_size()
        self.p.set_sieve_buf_size(128*1024)
        self.assertEqual(self.p.get_sieve_buf_size(), 128*1024)

class TestDCID(unittest.TestCase):

    LAYOUTS = (h5d.COMPACT,
                h5d.CONTIGUOUS,
                h5d.CHUNKED)

    CHUNKSIZES = ((1,), (4,4), (16,32,4))

    def setUp(self):
        self.p = h5p.create(h5p.DATASET_CREATE)

    def test_layout(self):
        for l in self.LAYOUTS:
            self.p.set_layout(l)
            self.assertEqual(self.p.get_layout(), l)

    def test_chunk(self):
        
        for c in self.CHUNKSIZES:
            self.p.set_chunk(c)
            self.assertEqual(self.p.get_chunk(), c)

    def test_fill_value(self):
        vals = [ numpy.array(1.0), numpy.array(2.0), numpy.array(4, dtype='=u8'),
                 numpy.array( (1,2,3.5+6j), dtype=[('a','<i4'),('b','=f8'),('c','<c16')] )]

        self.assertEqual(self.p.fill_value_defined(), h5d.FILL_VALUE_DEFAULT)

        for val in vals:
            self.p.set_fill_value(val)
            holder = numpy.ndarray(val.shape, val.dtype)
            self.p.get_fill_value(holder)
            self.assertEqual(holder, val)

        self.assertEqual(self.p.fill_value_defined(), h5d.FILL_VALUE_USER_DEFINED)


class TestDXID(unittest.TestCase):
    pass


