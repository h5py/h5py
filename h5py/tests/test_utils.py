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
import sys
import numpy

from common import TestCasePlus, api_18

from h5py import *
from h5py import utils

class TestUtils(TestCasePlus):

    def test_check_read(self):
        """ Check if it's possible to read from the NumPy array """

        carr = numpy.ones((10,10), order='C')
        farr = numpy.ones((10,10), order='F')
        oarr = numpy.ones((10,10), order='C')
        oarr.strides = (0,1)

        utils.check_numpy_read(carr)
        self.assertRaises(TypeError, utils.check_numpy_read, farr)
        self.assertRaises(TypeError, utils.check_numpy_read, oarr)

        s_space = h5s.create_simple((5,5))
        m_space = h5s.create_simple((10,10))
        l_space = h5s.create_simple((12,12))

        utils.check_numpy_read(carr, m_space.id)
        utils.check_numpy_read(carr, l_space.id)
        self.assertRaises(TypeError, utils.check_numpy_read, carr, s_space.id)

        # This should not matter for read
        carr.flags['WRITEABLE'] = False
        utils.check_numpy_read(carr)

    def test_check_write(self):
        """ Check if it's possible to write to the NumPy array """

        carr = numpy.ones((10,10), order='C')
        farr = numpy.ones((10,10), order='F')
        oarr = numpy.ones((10,10), order='C')
        oarr.strides = (0,1)

        utils.check_numpy_write(carr)
        self.assertRaises(TypeError, utils.check_numpy_write, farr)
        self.assertRaises(TypeError, utils.check_numpy_write, oarr)

        s_space = h5s.create_simple((5,5))
        m_space = h5s.create_simple((10,10))
        l_space = h5s.create_simple((12,12))

        utils.check_numpy_write(carr, s_space.id)
        utils.check_numpy_write(carr, m_space.id)
        self.assertRaises(TypeError, utils.check_numpy_write, carr, l_space.id)

        # This should matter now
        carr.flags['WRITEABLE'] = False
        self.assertRaises(TypeError, utils.check_numpy_write, carr)







