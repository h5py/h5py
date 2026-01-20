# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.
import os

import pytest

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

    @pytest.mark.thread_unsafe(reason="fork() from a thread may deadlock")
    @pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork() not available")
    def test_phil_fork(self):
        # Test that handling of the phil Lock after fork is correct.
        # Note that threading and fork() are mutually exclusive, so
        # the use case of a locked phil Lock during a fork() is unsupported.
        pid = os.fork()
        if pid == 0:
            # child process
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
