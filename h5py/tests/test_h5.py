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
import time
import unittest as ut

import numpy as np
import pytest

import h5py
from h5py import h5

from .common import TestCase

def fixnames():
    cfg = h5.get_config()
    cfg.complex_names = ('r','i')

class TestH5(TestCase):

    def test_config(self):
        cfg = h5.get_config()
        self.assertIsInstance(cfg, h5.H5PYConfig)
        cfg2 = h5.get_config()
        self.assertIs(cfg, cfg2)

    def test_cnames_get(self):
        cfg = h5.get_config()
        self.assertEqual(cfg.complex_names, ('r','i'))

    def test_cnames_set(self):
        self.addCleanup(fixnames)
        cfg = h5.get_config()
        cfg.complex_names = ('q','x')
        self.assertEqual(cfg.complex_names, ('q','x'))

    def test_cnames_set_exc(self):
        self.addCleanup(fixnames)
        cfg = h5.get_config()
        with self.assertRaises(TypeError):
            cfg.complex_names = ('q','i','v')
        self.assertEqual(cfg.complex_names, ('r','i'))

    def test_repr(self):
        cfg = h5.get_config()
        repr(cfg)


class TestInternalThreads(TestCase):

    """
        Feature: sizing the thread pool HDF5 uses internally
    """

    def test_negative_rejected(self):
        """ A negative thread count is rejected """
        with self.assertRaises(ValueError):
            h5py.set_internal_threads(-1)

    @ut.skipIf(h5.get_config().internal_threads,
               "HDF5 built with internal I/O thread support")
    def test_unavailable_names_requirement(self):
        """ Without support, the error says what HDF5 is required """
        with self.assertRaises(RuntimeError) as cm:
            h5py.set_internal_threads(1)
        self.assertIn("2.3.0", str(cm.exception))

    @pytest.mark.thread_unsafe(reason="Resizes a process-global thread pool")
    @ut.skipUnless(h5.get_config().internal_threads,
                   "HDF5 not built with internal I/O thread support")
    def test_set_and_disable(self):
        """ The pool accepts a size and can be turned off again """
        try:
            h5py.set_internal_threads(2)
        finally:
            # Back to the default so threading does not leak into other tests.
            h5py.set_internal_threads(0)

    @pytest.mark.thread_unsafe(reason="Resizes a process-global thread pool")
    @ut.skipUnless(h5.get_config().internal_threads,
                   "HDF5 not built with internal I/O thread support")
    def test_threaded_read(self):
        """ Reads through the pool return the data that was written """
        data = np.arange(64 * 1024, dtype='f8')
        # Filtered, so the workers run the filter pipeline too. LZF ships
        # with h5py.
        self.f.create_dataset('x', data=data, chunks=(1024,),
                              compression='lzf')
        # Reopen so the chunks come from disk, not the chunk cache.
        fname = self.f.filename
        self.f.close()
        self.f = h5py.File(fname, 'r')
        dset = self.f['x']
        try:
            h5py.set_internal_threads(4)
            np.testing.assert_array_equal(dset[...], data)
            np.testing.assert_array_equal(dset[100:50000:7],
                                          data[100:50000:7])
        finally:
            h5py.set_internal_threads(0)

    @pytest.mark.thread_unsafe(reason="fork() from a thread may deadlock")
    @pytest.mark.filterwarnings(
        # https://github.com/python/cpython/pull/100229
        r"ignore:.*use of fork\(\) may lead to deadlocks:DeprecationWarning"
    )
    @pytest.mark.skipif(not hasattr(os, "fork"), reason="fork() not available")
    @ut.skipUnless(h5.get_config().internal_threads,
                   "HDF5 not built with internal I/O thread support")
    def test_fork_with_pool(self):
        """ A process forked with the pool enabled can still read

        A child inherits HDF5's pool but not its threads, so h5py shuts the
        pool down across the fork. Without that, the child's first threaded
        read waits forever.
        """
        data = np.arange(64 * 1024, dtype='f8')
        self.f.create_dataset('x', data=data, chunks=(1024,),
                              compression='lzf')
        fname = self.f.filename
        self.f.close()

        def read_ok():
            with h5py.File(fname, 'r') as f:
                return np.array_equal(f['x'][...], data)

        try:
            h5py.set_internal_threads(4)
            pid = os.fork()
            if pid == 0:
                # Child process. Never return into the test runner.
                ok = False
                try:
                    ok = read_ok()
                finally:
                    os._exit(0 if ok else 1)

            # Parent process
            self.assertTrue(read_ok())

            deadline = time.monotonic() + 30
            while True:
                done, status = os.waitpid(pid, os.WNOHANG)
                if done:
                    break
                if time.monotonic() > deadline:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                    self.fail("Forked child hung reading with the pool")
                time.sleep(0.05)
            self.assertTrue(os.WIFEXITED(status))
            self.assertEqual(os.WEXITSTATUS(status), 0)
        finally:
            h5py.set_internal_threads(0)
