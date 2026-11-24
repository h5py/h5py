"""Monitor a growing SWMR dataset when Linux reports file modifications.

This example uses watchdog's inotify observer. Install watchdog with
``python -m pip install watchdog`` before running it.

Usage::

    python swmr_inotify_example.py [FILENAME [DATASETNAME]]

FILENAME defaults to ``swmr.h5`` and DATASETNAME to ``data``. The writer must
create the dataset and switch the file into SWMR mode before the reader opens
it. The reader stops when the writer closes the file, or on Ctrl-C.
"""
import logging
import sys
from threading import Event

import h5py
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class EventHandler(FileSystemEventHandler):
    def __init__(self, dataset, writer_closed):
        self.dataset = dataset
        self.writer_closed = writer_closed
        self.latest = 0

    def read_updates(self):
        self.dataset.refresh()
        shape = self.dataset.shape
        logging.info("Read data shape: %s", shape)
        if shape[0] > self.latest:
            logging.info("Reading out dataset [%d:%d]", self.latest, shape[0])
            self.dataset[self.latest:shape[0]]
        self.latest = shape[0]

    def on_modified(self, event):
        if not event.is_directory:
            self.read_updates()

    def on_closed(self, event):
        if not event.is_directory:
            logging.info("File writer closed file")
            self.read_updates()
            self.writer_closed.set()


def main():
    if sys.platform != 'linux':
        raise SystemExit('This example requires Linux inotify.')

    logging.basicConfig(format='%(asctime)s  %(levelname)s\t%(message)s', level=logging.INFO)
    file_name = sys.argv[1] if len(sys.argv) > 1 else 'swmr.h5'
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else 'data'

    with h5py.File(file_name, 'r', libver='latest', swmr=True) as f:
        writer_closed = Event()
        handler = EventHandler(f[dataset_name], writer_closed)
        handler.read_updates()
        observer = Observer()
        observer.schedule(handler, file_name, recursive=False)
        observer.start()
        logging.info("Watching %s", file_name)
        try:
            writer_closed.wait()
        except KeyboardInterrupt:
            logging.info("Stopped by user")
        finally:
            observer.stop()
            observer.join()


if __name__ == '__main__':
    main()
