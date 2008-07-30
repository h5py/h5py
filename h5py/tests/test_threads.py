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
from __future__ import with_statement

import unittest
import threading
import dummy_threading
import tempfile
from threading import Thread
import os
import numpy
import time

from h5py import *
from h5py.h5 import H5Error
from h5py.extras import h5sync
import h5py

LOCKTYPE = threading.RLock
SHAPE = (10,10)
BIGSHAPE = (5300,5000)      # About 200MB worth of doubles

class WriterThread(Thread):

    def __init__(self, name, value, dset, reclist):
        Thread.__init__(self)
        self.arr = numpy.ones(SHAPE)*value  # Value to be written
        self.dset = dset                    # Dataset to write to
        self.reclist = reclist              # We'll append our name to this when done
        self.name = name                
        self.next_thread = None             # Thread we'll try to get to break the lock
        self.sleeptime = 0                  # How long do we give that thread to try

    def run(self):
        # Try to fill the dataset with our values

        with self.dset._lock:
            if self.next_thread is not None:
                self.next_thread.start()    # Try to make the next thread steal the dataset
                time.sleep(self.sleeptime)  # Make sure it has a chance to misbehave
            self.dset[...] = self.arr
            self.reclist.append(self.name)  # Add our name to the record, before releasing the lock.

class TimedWriter(Thread):

    def __init__(self, dset, arr):
        Thread.__init__(self)
        self.dset = dset
        self.arr = arr
        self.timestart = 0
        self.timestop = 0

    def run(self):
        self.timestart = time.time()
        self.dset.id.write(h5s.ALL, h5s.ALL, self.arr)
        self.timestop = time.time()

class NullWriter(Thread):

    def __init__(self):
        Thread.__init__(self)
        self.time = 0

    def run(self):
        self.time = time.time()

class TestThreads(unittest.TestCase):      

    def setUp(self):

        self.fname = tempfile.mktemp('.hdf5')
        self.f = File(self.fname, 'w')
        self.old_lock = h5py.config.lock
        h5py.config.lock = LOCKTYPE()

    def tearDown(self):
        self.f.close()
        os.unlink(self.fname)
        h5py.config.lock = self.old_lock

    def test_hl_pos(self):

        reclist = []

        dset = self.f.create_dataset('ds',(10,10), dtype='=f8')
        thread_a = WriterThread('A', 1.0, dset, reclist)
        thread_b = WriterThread('B', 2.0, dset, reclist)
        thread_c = WriterThread('C', 3.0, dset, reclist)
        thread_d = WriterThread('D', 4.0, dset, reclist)

        thread_a.next_thread = thread_b
        thread_b.next_thread = thread_c
        thread_c.next_thread = thread_d

        thread_a.sleeptime = 3  # Must be larger than b, so that b attempts to steal the lock.
        thread_b.sleeptime = 2  # Must be larger than c, for the same reason
        thread_c.sleeptime = 1
        
        thread_a.start()

        # Ensure they all finish
        # TODO: Handle a possible deadlock more gracefully.
        thread_a.join()
        thread_b.join()
        thread_c.join()
        thread_d.join()

        self.assertEqual(reclist, ['A','B','C','D'])
        self.assert_(numpy.all(dset.value == numpy.ones(SHAPE)*4.0))

    def test_hl_neg(self):

        oldlock = h5py.config.lock
        try:
            # Force the threads to operate in reverse order, by defeating locks
            h5py.config.lock = dummy_threading.RLock()
            
            reclist = []

            dset = self.f.create_dataset('ds',(10,10), dtype='=f8')
            thread_a = WriterThread('A', 1.0, dset, reclist)
            thread_b = WriterThread('B', 2.0, dset, reclist)
            thread_c = WriterThread('C', 3.0, dset, reclist)
            thread_d = WriterThread('D', 4.0, dset, reclist)

            thread_a.next_thread = thread_b
            thread_b.next_thread = thread_c
            thread_c.next_thread = thread_d

            thread_a.sleeptime = 3  # Must be larger than b, so that b attempts to steal the lock.
            thread_b.sleeptime = 2  # Must be larger than c, for the same reason
            thread_c.sleeptime = 1
            
            thread_a.start()

            # Ensure they all finish
            # TODO: Handle a possible deadlock more gracefully.
            thread_a.join()
            thread_b.join()
            thread_c.join()
            thread_d.join()

            self.assertEqual(reclist, ['D','C','B','A'])
            self.assert_(numpy.all(dset.value == numpy.ones(SHAPE)*1.0))
        finally:
            h5py.config.lock = oldlock

    def test_nonblock(self):
        # Ensure low-level I/O blocking behavior

        dset = self.f.create_dataset('ds', BIGSHAPE, '=f8')
        arr = numpy.ones(BIGSHAPE, '=f8')
        DELAY = 0.1

        thread_a = TimedWriter(dset, arr)
        thread_b = NullWriter()

        thread_a.start()
        time.sleep(DELAY)
        thread_b.start()

        thread_a.join()
        thread_b.join()

        write_time = thread_a.timestop - thread_a.timestart
        if write_time < DELAY:
            raise Exception("Write was too fast to test blocking (%f sec; need %f)" % (write_time, DELAY))

        if h5py.config.compile_opts['IO_NONBLOCK']:
            self.assert_(thread_b.time < thread_a.timestop)
        else:
            self.assert_(thread_b.time > thread_a.timestop)

    def test_lock_behavior(self):
        # Check to make sure the user-provided lock class behaves correctly
        # when called from C code

        dset = self.f.create_dataset('ds', SHAPE, '=f8')   
        arr = numpy.ones(SHAPE, '=f8')

        writethread = TimedWriter(dset, arr)

        with h5py.config.lock:
            writethread.start()
            time.sleep(2)       # give it more than enough time to finish, if it ignores the lock
            exit_lock_time = time.time()
            time.sleep(2)
        writethread.join()

        if h5py.config.compile_opts['IO_NONBLOCK']:
            # With non-blocking I/O, the library will double-check that the
            # global lock isn't held, to prevent more than one thread from
            # calling into the HDF5 API.
            self.assert_(writethread.timestop > exit_lock_time)
        else:
            # In blocking mode, the GIL ensures that only one thread at a time
            # can access the HDF5 library.  Therefore the library ignores the
            # state of the soft lock.  If this were a real program, the author
            # of "writethread" should acquire the lock first.
            self.assert_(writethread.timestop < exit_lock_time)


    def test_decorator(self):

        time1 = 0
        time2 = 0

        class SleeperThread(Thread):

            def __init__(self, sleeptime):
                Thread.__init__(self)
                self.sleeptime = sleeptime
                self.time = 0

            @h5sync
            def run(self):
                time.sleep(self.sleeptime)
                self.time = time.time()

        thread_a = SleeperThread(2)
        thread_b = SleeperThread(1)

        thread_a.start()
        thread_b.start()

        thread_a.join()
        thread_b.join()

        self.assert_(thread_a.time < thread_b.time)
        
        @h5sync
        def thisismyname(foo):
            pass
        
        self.assertEqual(thisismyname.__name__, "thisismyname")







