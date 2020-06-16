#!/usr/bin/env python3
import os
import time
import numpy
from tempfile import TemporaryDirectory
import logging
logger = logging.getLogger(__name__)
import h5py

#Needed for mutithreading:
from queue import Queue
from threading import Thread, Event
import multiprocessing


class Reader(Thread):
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, queue_in, queue_out, quit_event):
        Thread.__init__(self)
        self._queue_in = queue_in
        self._queue_out = queue_out
        self._quit_event = quit_event
        self.daemon = True
        self.start()

    def run(self):
        while not self._quit_event.is_set():
            task = self._queue_in.get()
            if task:
                fn, ds, position = task
            else:
                logger.debug("Swallow a bitter pill: %s", task)
                break
            try:
                r = fn(ds, position)
                self._queue_out.put((position, r))
            except Exception as e:
                raise(e)
            finally:
                self._queue_in.task_done()


class SlicingBenchmark:
    """
    Benchmark for reading slices in the most pathlogical way in a chunked dataset
    Allows the test
    """
    def __init__(self, ndim=3, size=1024, chunk=64, dtype="float32", precision=16, compression_kwargs=None):
        """
        Defines some parameters for the benchmark, can be tuned later on.

        :param ndim: work in 3D datasets
        :param size: Volume size 1024**3 elements
        :param chunk: size of one chunk, with itemsize = 32bits this makes block size of 1MB by default
        :param dtype: the type of data to be stored
        :param precision: to gain a bit in compression, number of trailing bits to be zeroed.
        :param compression_kwargs: a dict with all options for configuring the compression
        """
        self.ndim = ndim
        self.size = size
        self.dtype = numpy.dtype(dtype)
        self.chunk = chunk
        self.precision = precision
        self.tmpdir = None
        self.filename = None
        self.h5path = "data"
        self.total_size = self.size ** self.ndim * self.dtype.itemsize
        self.needed_memory = self.size ** (self.ndim-1) * self.dtype.itemsize * self.chunk
        if compression_kwargs is None:
            self.compression = {}
        else:
            self.compression = dict(compression_kwargs)

    def setup(self):
        self.tmpdir = TemporaryDirectory()
        self.filename = os.path.join(self.tmpdir.name, "benchmark_slicing.h5")
        logger.info("Saving data in %s", self.filename)
        logger.info("Total size: %i^%i volume size: %.3fGB, Needed memory: %.3fGB",
                    self.size, self.ndim, self.total_size/1e9, self.needed_memory/1e9)

        shape = [self.size]  * self.ndim
        chunks = (self.chunk,) * self.ndim
        if self.precision and self.dtype.char in "df":
            if self.dtype.itemsize == 4:
                mask = numpy.uint32(((1<<32) - (1<<(self.precision))))
            elif self.dtype.itemsize == 8:
                mask = numpy.uint64(((1<<64) - (1<<(self.precision))))
            else:
                logger.warning("Precision reduction: only float32 and float64 are supported")
        else:
            self.precision = 0
        t0 = time.time()
        with h5py.File(self.filename, 'w') as h:
            ds = h.create_dataset(self.h5path,
                                  shape,
                                  chunks=chunks,
                                  **self.compression)
            for i in range(0, self.size, self.chunk):
                x, y, z = numpy.ogrid[i:i+self.chunk, :self.size, :self.size]
                data = (numpy.sin(x/3)*numpy.sin(y/5)*numpy.sin(z/7)).astype(self.dtype)
                if self.precision:
                    idata = data.view(mask.dtype)
                    idata &= mask # mask out the last XX bits
                ds[i:i+self.chunk] = data
            t1 = time.time()
        dt = t1 - t0
        filesize = os.stat(self.filename).st_size
        logger.info("Compression: %.3f time %.3fs uncompressed data saving speed %.3f MB/s effective write speed  %.3f MB/s ",
                    self.total_size/filesize, dt,  self.total_size/dt/1e6, filesize/dt/1e6)

    def teardown(self):
        self.tmpdir.cleanup()
        self.filename = None

    @staticmethod
    def read_slice(dataset, position):
        """This reads all hyperplans crossing at the given position:
        enforces many reads of different chunks,
        Probably one of the most pathlogical use-case"""
        assert dataset.ndim == len(position)
        l = len(position)
        res = []
        noneslice = slice(None)
        for i, w in enumerate(position):
            where = [noneslice]*i + [w] + [noneslice]*(l - 1 - i)
            res.append(dataset[tuple(where)])
        return res

    def time_sequential_reads(self, nb_read=64):
        "Perform the reading of many orthogonal hyperplanes"
        where = [[(i*(self.chunk+1+j))%self.size for j in range(self.ndim)] for i in range(nb_read)]
        with h5py.File(self.filename, "r") as h:
            ds = h[self.h5path]
            t0 = time.time()
            for i in where:
                self.read_slice(ds, i)
            t1 = time.time()
        dt = t1 - t0
        logger.info("Time for reading %sx%s slices: %.3fs fps: %.3f "%(self.ndim, nb_read, dt, self.ndim*nb_read/dt) +
                    "Uncompressed data read speed %.3f MB/s"%(self.ndim*nb_read*self.needed_memory/dt/1e6))
        return dt

    def time_threaded_reads(self, nb_read=64, nthreads=multiprocessing.cpu_count()):
        "Perform the reading of many orthogonal hyperplanes, threaded version"
        where = [[(i*(self.chunk+1+j))%self.size for j in range(self.ndim)] for i in range(nb_read)]
        tasks = Queue()
        results = Queue()
        quitevent = Event()
        pool = [Reader(tasks, results, quitevent) for i in range(nthreads)]
        res = []
        with h5py.File(self.filename, "r") as h:
            ds = h[self.h5path]
            t0 = time.time()
            for i in where:
                tasks.put((self.read_slice, ds, i))
            for i in where:
                a = results.get()
                res.append(a[0])
                results.task_done()
            tasks.join()
            results.join()
            t1 = time.time()
        # destroy the threads in the pool
        quitevent.set()
        for i in range(nthreads):
            tasks.put(None)

        dt = t1 - t0
        logger.info("Time for %s-threaded reading %sx%s slices: %.3fs fps: %.3f "%(nthreads, self.ndim, nb_read, dt, self.ndim*nb_read/dt) +
                    "Uncompressed data read speed %.3f MB/s"%(self.ndim*nb_read*self.needed_memory/dt/1e6))
        return dt


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    benckmark = SlicingBenchmark()
    benckmark.setup()
    benckmark.time_sequential_reads()
    benckmark.time_threaded_reads()
    benckmark.teardown()
