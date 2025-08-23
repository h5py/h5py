# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.
import os
import threading
from unittest import SkipTest

import time

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
