.. _swmr:

Single Writer Multiple Reader (SWMR)
====================================

Starting with version 2.5.0, h5py includes support for the HDF5 SWMR features.

What is SWMR?
-------------

The SWMR features allow simple concurrent reading of a HDF5 file while it is
being written from another process. Prior to this feature addition it was not
possible to do this as the file data and meta-data would not be synchronised
and attempts to read a file which was open for writing would fail or result in
garbage data.

A file which is being written to in SWMR mode is guaranteed to always be in a
valid (non-corrupt) state for reading. This has the added benefit of leaving a
file in a valid state even if the writing application crashes before closing
the file properly.

This feature has been implemented to work with independent writer and reader
processes. No synchronisation is required between processes and it is up to the
user to implement either a file polling mechanism, inotify or any other IPC
mechanism to notify when data has been written.

The SWMR functionality requires use of the latest HDF5 file format: v110. In
practice this implies using at least HDF5 1.10 (this can be checked via
``h5py.version.info``) and setting the libver bounding to "latest" when opening or
creating the file.


.. Warning:: New v110 format files are *not* compatible with v18 format. So,
             files written in SWMR mode with libver='latest' cannot be opened
             with older versions of the HDF5 library (basically any version
             older than the SWMR feature).


The HDF Group has documented the SWMR features in details on the website:
`Single-Writer/Multiple-Reader (SWMR) Documentation <https://support.hdfgroup.org/HDF5/docNewFeatures/NewFeaturesSwmrDocs.html>`_.
This is highly recommended reading for anyone intending to use the SWMR feature
even through h5py. For production systems in particular pay attention to the
file system requirements regarding POSIX I/O semantics.



Using the SWMR feature from h5py
--------------------------------

The following basic steps are typically required by writer and reader processes:

- Writer process creates the target file and all groups, datasets and attributes.
- Writer process switches file into SWMR mode.
- Reader process can open the file with swmr=True.
- Writer writes and/or appends data to existing datasets (new groups and datasets *cannot* be created when in SWMR mode).
- Writer regularly flushes the target dataset to make it visible to reader processes.
- Reader refreshes target dataset before reading new meta-data and/or main data.
- Writer eventually completes and close the file as normal.
- Reader can finish and close file as normal whenever it is convenient.

The following snippet demonstrate a SWMR writer appending to a single dataset::

    f = h5py.File("swmr.h5", 'w', libver='latest')
    arr = np.array([1,2,3,4])
    dset = f.create_dataset("data", chunks=(2,), maxshape=(None,), data=arr)
    f.swmr_mode = True
    # Now it is safe for the reader to open the swmr.h5 file
    for i in range(5):
        new_shape = ((i+1) * len(arr), )
        dset.resize( new_shape )
        dset[i*len(arr):] = arr
        dset.flush()
        # Notify the reader process that new data has been written


The following snippet demonstrate how to monitor a dataset as a SWMR reader::

    f = h5py.File("swmr.h5", 'r', libver='latest', swmr=True)
    dset = f["data"]
    while True:
        dset.id.refresh()
        shape = dset.shape
        print( shape )


Examples
--------

In addition to the above example snippets, a few more complete examples can be
found in the examples folder. These examples are described in the following
sections.

Dataset monitor with inotify
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The inotify example demonstrates how to use SWMR in a reading application which
monitors live progress as a dataset is being written by another process. This
example uses the the linux inotify
(`pyinotify <https://pypi.python.org/pypi/pyinotify>`_ python bindings) to
receive a signal each time the target file has been updated.

.. literalinclude:: ../examples/swmr_inotify_example.py

Multiprocess concurrent write and read
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The SWMR multiprocess example starts two concurrent child processes:
a writer and a reader.
The writer process first creates the target file and dataset. Then it switches
the file into SWMR mode and the reader process is notified (with a
multiprocessing.Event) that it is safe to open the file for reading.

The writer process then continue to append chunks to the dataset. After each
write it notifies the reader that new data has been written. Whether the new
data is visible in the file at this point is subject to OS and file system
latencies.

The reader first waits for the initial "SWMR mode" notification from the
writer, upon which it goes into a loop where it waits for further notifications
from the writer. The reader may drop some notifications, but for each one
received it will refresh the dataset and read the dimensions. After a time-out
it will drop out of the loop and exit.

.. literalinclude:: ../examples/swmr_multiprocess.py

The example output below (from a virtual Ubuntu machine) illustrate some
latency between the writer and reader::

    python examples/swmr_multiprocess.py
      INFO  2015-02-26 18:05:03,195        root  Starting reader
      INFO  2015-02-26 18:05:03,196        root  Starting reader
      INFO  2015-02-26 18:05:03,197      reader  Waiting for initial event
      INFO  2015-02-26 18:05:03,197        root  Waiting for writer to finish
      INFO  2015-02-26 18:05:03,198      writer  Creating file swmrmp.h5
      INFO  2015-02-26 18:05:03,203      writer  SWMR mode
      INFO  2015-02-26 18:05:03,205      reader  Opening file swmrmp.h5
      INFO  2015-02-26 18:05:03,210      writer  Resizing dset shape: (4,)
      INFO  2015-02-26 18:05:03,212      writer  Sending event
      INFO  2015-02-26 18:05:03,213      reader  Read dset shape: (4,)
      INFO  2015-02-26 18:05:03,214      writer  Resizing dset shape: (8,)
      INFO  2015-02-26 18:05:03,214      writer  Sending event
      INFO  2015-02-26 18:05:03,215      writer  Resizing dset shape: (12,)
      INFO  2015-02-26 18:05:03,215      writer  Sending event
      INFO  2015-02-26 18:05:03,215      writer  Resizing dset shape: (16,)
      INFO  2015-02-26 18:05:03,215      reader  Read dset shape: (12,)
      INFO  2015-02-26 18:05:03,216      writer  Sending event
      INFO  2015-02-26 18:05:03,216      writer  Resizing dset shape: (20,)
      INFO  2015-02-26 18:05:03,216      reader  Read dset shape: (16,)
      INFO  2015-02-26 18:05:03,217      writer  Sending event
      INFO  2015-02-26 18:05:03,217      reader  Read dset shape: (20,)
      INFO  2015-02-26 18:05:03,218      reader  Read dset shape: (20,)
      INFO  2015-02-26 18:05:03,219        root  Waiting for reader to finish
