# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Tests the h5py collective IO

"""
from __future__ import absolute_import

import numpy as np
import h5py

from ..common import ut, TestCase


# Check if we are in an MPI environment, need more than 1 process for these
# tests to be meaningful
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPI_ENV = (comm.size > 1)
    MPI_SIZE = comm.size
    MPI_RANK = comm.rank
except ImportError:
    MPI_ENV = False


@ut.skipUnless(h5py.get_config().mpi and MPI_ENV, 'MPI support required')
class TestCollectiveWrite(TestCase):

    def setUp(self):
        """Open a file in MPI mode"""
        self.path = self.mktemp() if MPI_RANK == 0 else None
        self.path = comm.bcast(self.path, root=0)
        self.f = h5py.File(self.path, 'w', driver='mpio', comm=comm)

    def test_collective_write(self):
        """Test a standard collective write."""

        dset = self.f.create_dataset("test_data", (MPI_SIZE, 20), dtype=np.int32)

        # Write dataset collectively, each process writes one row
        with dset.collective:
            dset[MPI_RANK:(MPI_RANK + 1)] = MPI_RANK
        self.f.close()

        # Test that the array is as expected
        with h5py.File(self.path, "r") as fh:
            self.assertEqual(fh["test_data"].shape, (MPI_SIZE, 20))
            arr = np.tile(np.arange(MPI_SIZE), (20, 1)).T
            self.assertTrue((fh["test_data"][:] == arr).all())

    def test_collective_write_empty_rank(self):
        """Test a collective write where some ranks may be empty.

        WARNING: if this test fails it may cause a lockup in the MPI code.
        """

        # Only the first NUM_WRITE ranks will actually write anything
        NUM_WRITE = MPI_SIZE // 2

        dset = self.f.create_dataset("test_data", (NUM_WRITE, 20), dtype=np.int32)

        # Write dataset collectively, each process writes one row
        start = min(MPI_RANK, NUM_WRITE)
        end = min(MPI_RANK + 1, NUM_WRITE)
        with dset.collective:
            dset[start:end] = MPI_RANK
        self.f.close()

        # Test that the array is as expected
        with h5py.File(self.path, "r") as fh:
            self.assertEqual(fh["test_data"].shape, (NUM_WRITE, 20))
            arr = np.tile(np.arange(NUM_WRITE), (20, 1)).T
            self.assertTrue((fh["test_data"][:] == arr).all())


@ut.skipUnless(h5py.get_config().mpi and MPI_ENV, 'MPI support required')
class TestCollectiveRead(TestCase):

    def setUp(self):
        """Open a file in MPI mode"""
        self.path = self.mktemp() if MPI_RANK == 0 else None
        self.path = comm.bcast(self.path, root=0)

        if MPI_RANK == 0:
            with h5py.File(self.path, 'w') as fh:
                dset = fh.create_dataset("test_data", (20, MPI_SIZE), dtype=np.int32)
                dset[:] = np.arange(MPI_SIZE)[np.newaxis, :]

        self.f = h5py.File(self.path, 'r', driver='mpio', comm=comm)

    def test_collective_read(self):
        """Test a standard collective read."""

        dset = self.f["test_data"]

        self.assertEqual(dset.shape, (20, MPI_SIZE))

        # Read dataset collectively, each process reads one column
        with dset.collective:
            d = dset[:, MPI_RANK:(MPI_RANK + 1)]

        self.assertTrue((d == MPI_RANK).all())

    def test_collective_read_empty_rank(self):
        """Test a collective read where some ranks may read nothing.

        WARNING: if this test fails it may cause a lockup in the MPI code.
        """

        start = 0 if MPI_RANK == 0 else MPI_SIZE
        end = MPI_SIZE

        dset = self.f["test_data"]
        self.assertEqual(dset.shape, (20, MPI_SIZE))

        # Read dataset collectively, only the first rank should actually read
        # anything
        with dset.collective:
            d = dset[:, start:end]

        if MPI_RANK == 0:
            self.assertTrue((d == np.arange(MPI_SIZE)[np.newaxis, :]).all())
        else:
            self.assertEqual(d.shape, (20, 0))
