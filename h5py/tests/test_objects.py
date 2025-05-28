# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.
import os
import tempfile
import threading
from unittest import SkipTest

import numpy as np
import time

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

        # On Windows forking (and the register_at_fork handler)
        # are not available, skip this test.
        if not hasattr(os, "fork"):
            raise SkipTest("os.fork not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            fns = []
            for i in range(10):
                fn = os.path.join(tmpdir, f'test{i}.h5')
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

    def test_phil_fork_with_threads_2(self):
        # Test that handling of the phil Lock after fork is correct.
        # We simulate a deadlock in the forked process by explicitly
        # waiting for the phil Lock to be acquired in a different thread
        # before forking.

        # On Windows forking (and the register_at_fork handler)
        # are not available, skip this test.
        if not hasattr(os, "fork"):
            raise SkipTest("os.fork not available")

        thread_acquired_phil_event = threading.Event()

        def f():
            o.phil.acquire()
            try:
                thread_acquired_phil_event.set()
                time.sleep(1)
            finally:
                o.phil.release()

        thread = threading.Thread(target=f)
        thread.start()
        try:
            # on the main thread wait for the thread to have acquired the phil lock
            thread_acquired_phil_event.wait()

            # now fork main thread while the other thread holds the lock
            pid = os.fork()
            if pid == 0:
                with o.phil:
                    pass
                os._exit(0)
            else:
                os.waitpid(pid, 0)
        finally:
            thread.join()
