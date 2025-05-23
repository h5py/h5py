# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.
import os
import random
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import h5py
from h5py import _objects as o
from .common import TestCase


class TestObjects(TestCase):

    def test_invalid(self):
        # Check for segfault on close
        oid = o.ObjectID(0)
        del oid
        oid = o.ObjectID(1)
        del oid

    def test_equality(self):
        # Identifier-based equality
        oid1 = o.ObjectID(42)
        oid2 = o.ObjectID(42)
        oid3 = o.ObjectID(43)

        self.assertEqual(oid1, oid2)
        self.assertNotEqual(oid1, oid3)

    def test_hash(self):
        # Default objects are not hashable
        oid = o.ObjectID(42)
        with self.assertRaises(TypeError):
            hash(oid)

    def test_phil_fork_with_threads(self):
        # Test that handling of the phil Lock after fork is correct
        # even when multiple threads are present.

        with tempfile.TemporaryDirectory() as tmpdir:
            fns = []
            for i in range(10):
                fn = f'{tmpdir}/test{i}.h5'
                with h5py.File(fn, 'w') as f:
                    f.create_dataset('values', data=np.random.rand(1000, 1000))
                fns.append(fn)

            def f(fn):
                with h5py.File(fn, 'r') as f:
                    for _ in range(100):
                        _ = f['values'][:]

            # create 10 threads, each reading from an HDF5 file
            threads = []
            for fn in fns:
                thread = threading.Thread(target=f, args=(fn,))
                thread.start()
                threads.append(thread)

            # While the threads are running (and potentially holding the phil Lock)
            # create 10 processes, each also reading from an HDF5 file
            worker2pid = {}
            for worker_id, fn in enumerate(fns):
                pid = os.fork()
                if pid == 0:
                    # child process
                    f(fn)
                    os._exit(0)
                else:
                    # parent process
                    worker2pid[worker_id] = pid

            # Wait for all child processes to finish
            for worker_id, pid in worker2pid.items():
                os.waitpid(pid, 0)

            # Wait for all threads to finish
            for thread in threads:
                thread.join()
