# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.
import os
import signal
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

    def test_fork_with_threads(self):
        # Test that we do not deadlock after forking when the process
        # is using multiple threads to simultaneously perform h5py operations

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
            start = time.time()
            timeout = 60.0
            while time.time() < start + timeout:
                for worker_id in list(worker2pid):
                    pid = worker2pid[worker_id]
                    waited_pid, status = os.waitpid(pid, os.WNOHANG)
                    if waited_pid == pid:
                        assert os.WIFEXITED(status)
                        assert os.WEXITSTATUS(status) == 0
                        del worker2pid[worker_id]
                # If all child processes exited we can stop looping, otherwise sleep and try again
                if not worker2pid:
                    break
                time.sleep(0.1)

            # Make sure all child processes finished successfully
            if len(worker2pid) > 0:
                # Some child processes did not finish because they could not acquire the phil lock,
                # make sure we clean up after us.
                for worker_id, pid in worker2pid.items():
                    # Kill the zombie child processes
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)

                assert False, "Some child processes did not finish and had to be killed"

            # Wait for all threads to finish
            for thread in threads:
                thread.join()

    def test_phil_fork_with_threads(self):
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
            # wait for the thread running "f" to have acquired the phil lock
            thread_acquired_phil_event.wait()

            # now fork the current (main) thread while the other thread holds the lock
            pid = os.fork()
            if pid == 0:
                # child process
                # If we handle the phil lock correctly, this should not deadlock,
                # and we should be able to acquire the lock here.
                if o.phil.acquire(blocking=False):
                    o.phil.release()
                    os._exit(0)
                else:
                    os._exit(1)
            else:
                # parent process
                # wait for the child process to finish
                _, status = os.waitpid(pid, 0)
                assert os.WIFEXITED(status)
                assert os.WEXITSTATUS(status) == 0
        finally:
            thread.join()
