# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2021 Hyo-Kyung Lee
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Tests the h5py._hl.av module.

"""
import os
import numpy as np
import h5py
import pytest

from .common import ut, TestCase, insubprocess


class TestAsync(TestCase):

    def setUp(self):
        """ like TestCase.setUp but also store the file path """
        self.path = self.mktemp()
        self.f = h5py.File(self.path, 'w', sync=False)
        
    def tearDown(self):
        if self.f:
            self.f.close()

    def test_async(self):
        """ async_test_multifile.c """

        self.f.create_dataset("dset0",
                              data=np.zeros(10000, dtype=np.int32))
        self.f.close()

        with h5py.File(self.path, "r") as h5:
            h5["dset0"][0]
            

    
