"""
    Demonstrate the use of h5py in SWMR mode to write to a dataset (appending)
    from one process while monitoring the growing dataset from another process.

    Usage:
            swmr_multiprocess.py [FILENAME [DATASETNAME]]

              FILENAME:    name of file to monitor. Default: swmrmp.h5
              DATASETNAME: name of dataset to monitor in DATAFILE. Default: data

    This script will start up two processes: a writer and a reader. The writer
    will open/create the file (FILENAME) in SWMR mode, create a dataset and start
    appending data to it. After each append the dataset is flushed and an event
    sent to the reader process. Meanwhile the reader process will wait for events
    from the writer and when triggered it will refresh the dataset and read the
    current shape of it.
"""

import sys
import h5py
import numpy as np
import logging
from multiprocessing import Process, Event

class SwmrReader(Process):
    def __init__(self, event, fname, dsetname, timeout = 2.0):
        super(SwmrReader, self).__init__()
        self._event = event
        self._fname = fname
        self._dsetname = dsetname
        self._timeout = timeout

    def run(self):
        self.log = logging.getLogger('reader')
        self.log.info("Waiting for initial event")
        assert self._event.wait( self._timeout )
        self._event.clear()

        self.log.info("Opening file %s", self._fname)
        f = h5py.File(self._fname, 'r', libver='latest', swmr=True)
        assert f.swmr_mode
        dset = f[self._dsetname]
        try:
            # monitor and read loop
            while self._event.wait( self._timeout ):
                self._event.clear()
                self.log.debug("Refreshing dataset")
                dset.refresh()

                shape = dset.shape
                self.log.info("Read dset shape: %s"%str(shape))
        finally:
            f.close()

class SwmrWriter(Process):
    def __init__(self, event, fname, dsetname):
        super(SwmrWriter, self).__init__()
        self._event = event
        self._fname = fname
        self._dsetname = dsetname

    def run(self):
        self.log = logging.getLogger('writer')
        self.log.info("Creating file %s", self._fname)
        f = h5py.File(self._fname, 'w', libver='latest')
        try:
            arr = np.array([1,2,3,4])
            dset = f.create_dataset(self._dsetname, chunks=(2,), maxshape=(None,), data=arr)
            assert not f.swmr_mode

            self.log.info("SWMR mode")
            f.swmr_mode = True
            assert f.swmr_mode
            self.log.debug("Sending initial event")
            self._event.set()

            # Write loop
            for i in range(5):
                new_shape = ((i+1) * len(arr), )
                self.log.info("Resizing dset shape: %s"%str(new_shape))
                dset.resize( new_shape )
                self.log.debug("Writing data")
                dset[i*len(arr):] = arr
                #dset.write_direct( arr, np.s_[:], np.s_[i*len(arr):] )
                self.log.debug("Flushing data")
                dset.flush()
                self.log.info("Sending event")
                self._event.set()
        finally:
            f.close()


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)10s  %(asctime)s  %(name)10s  %(message)s',level=logging.INFO)
    fname = 'swmrmp.h5'
    dsetname = 'data'
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    if len(sys.argv) > 2:
        dsetname = sys.argv[2]

    event = Event()
    reader = SwmrReader(event, fname, dsetname)
    writer = SwmrWriter(event, fname, dsetname)

    logging.info("Starting reader")
    reader.start()
    logging.info("Starting reader")
    writer.start()

    logging.info("Waiting for writer to finish")
    writer.join()
    logging.info("Waiting for reader to finish")
    reader.join()
