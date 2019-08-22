
"""
    Demonstrate the use of h5py in SWMR mode to monitor the growth of a dataset
    on notification of file modifications.

    This demo uses pyinotify as a wrapper of Linux inotify.
    https://pypi.python.org/pypi/pyinotify

    Usage:
            swmr_inotify_example.py [FILENAME [DATASETNAME]]

              FILENAME:    name of file to monitor. Default: swmr.h5
              DATASETNAME: name of dataset to monitor in DATAFILE. Default: data

    This script will open the file in SWMR mode and monitor the shape of the
    dataset on every write event (from inotify). If another application is
    concurrently writing data to the file, the writer must have have switched
    the file into SWMR mode before this script can open the file.
"""
import asyncore
import pyinotify
import sys
import h5py
import logging

#assert h5py.version.hdf5_version_tuple >= (1,9,178), "SWMR requires HDF5 version >= 1.9.178"

class EventHandler(pyinotify.ProcessEvent):

    def monitor_dataset(self, filename, datasetname):
        logging.info("Opening file %s", filename)
        self.f = h5py.File(filename, 'r', libver='latest', swmr=True)
        logging.debug("Looking up dataset %s"%datasetname)
        self.dset = self.f[datasetname]

        self.get_dset_shape()

    def get_dset_shape(self):
        logging.debug("Refreshing dataset")
        self.dset.refresh()

        logging.debug("Getting shape")
        shape = self.dset.shape
        logging.info("Read data shape: %s"%str(shape))
        return shape

    def read_dataset(self, latest):
        logging.info("Reading out dataset [%d]"%latest)
        self.dset[latest:]

    def process_IN_MODIFY(self, event):
        logging.debug("File modified!")
        shape = self.get_dset_shape()
        self.read_dataset(shape[0])

    def process_IN_CLOSE_WRITE(self, event):
        logging.info("File writer closed file")
        self.get_dset_shape()
        logging.debug("Good bye!")
        sys.exit(0)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s  %(levelname)s\t%(message)s',level=logging.INFO)

    file_name = "swmr.h5"
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    dataset_name = "data"
    if len(sys.argv) > 2:
        dataset_name = sys.argv[2]


    wm = pyinotify.WatchManager()  # Watch Manager
    mask = pyinotify.IN_MODIFY | pyinotify.IN_CLOSE_WRITE
    evh = EventHandler()
    evh.monitor_dataset( file_name, dataset_name )

    notifier = pyinotify.AsyncNotifier(wm, evh)
    wdd = wm.add_watch(file_name, mask, rec=False)

    # Sit in this loop() until the file writer closes the file
    # or the user hits ctrl-c
    asyncore.loop()
