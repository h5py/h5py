.. _vds:

Virtual Dataset (VDS)
====================================

Starting with version 1.10.0, h5py includes support for the HDF5 VDS features. The VDS feature is available in the 1.10.0 version of the HDF5 library.


What is Virtual Dataset?
------------------------

The virtual dataset feature allows the mapping together of datasets in multiple
files, into a single, sliceable dataset via an interface layer. The mapping can
be made ahead of time, before the parent files are written, and is transparent to
the parent dataset characteristics (SWMR, chunking, compression etc...).
The datasets can be meshed in arbitrary combinations, and even the data type 
converted.

.. Warning:: Virtual dataset files cannot be opened by earlier versions of the hdf5 library.


The HDF Group has documented the VDS features in details on the website:
`Virtual Datasets (VDS) Documentation <https://support.hdfgroup.org/HDF5/docNewFeatures/NewFeaturesVirtualDatasetDocs.html>`_.


Using the VDS feature from h5py
--------------------------------

The following basic steps are required to create a Virtual Dataset using the h5py interface:

- Creation of a target data file.
- Knowledge of the future or existing parent dataset locations.
- A list of Virtual Maps to link the source and target datasets, including type conversion.

The following snippet demonstrates a Virtual Dataset being created to simply stack together files::

    f = h5py.File("VDS.h5", 'w', libver='latest')
    file_names_to_concatenate = ['1.h5', '2.h5', '3.h5', '4.h5', '5.h5']
    entry_key = 'data' # where the data is inside of the source files.
    sh = h5.File(file_names_to_concatenate[0],'r')[entry_key].shape # get the first ones shape.
    
    TGT = h5.VirtualTarget(outfile, outkey, shape=(len(file_names_to_concatenate, ) + sh)
    
    for i in range(num_projections):
        VSRC = h5.VirtualSource(file_names_to_concatenate[i]), entry_key, shape=sh)
        VM = h5.VirtualMap(VSRC[:,:,:], TGT[i:(i+1):1,:,:,:],dtype=np.float)
        VMlist.append(VM)
    
    d = f.create_virtual_dataset(VMlist=VMlist,fillvalue=0)
    f.close()
    

Examples
--------

In addition to the above example snippets, a few more complete examples can be
found in the examples folder. These examples are described in the following 
sections

Dataset monitor with inotify
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The inotify example demonstrate how to use SWMR in a reading application which
monitors live progress as a dataset is being written by another process. This
example uses the the linux inotify 
(`pyinotify <https://pypi.python.org/pypi/pyinotify>`_ python bindings) to 
receive a signal each time the target file has been updated.

.. literalinclude:: ../examples/swmr_inotify_example.py

Multiprocess concurrent write and read
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The SWMR multiprocess example starts starts two concurrent child processes: 
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

