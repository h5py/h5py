.. currentmodule:: h5py
.. _vds:

Virtual Datasets (VDS)
======================

Starting with version 2.9, h5py includes high-level support for HDF5
'virtual datasets'.  The VDS feature is available in version 1.10 of
the HDF5 library; h5py must be built with a new enough version of HDF5
to create or read virtual datasets.


What are virtual datasets?
--------------------------

Virtual datasets allow a number of real datasets to be mapped together into
a single, sliceable dataset via an interface layer. The mapping can
be made ahead of time, before the parent files are written, and is transparent to
the parent dataset characteristics (SWMR, chunking, compression etc...).
The datasets can be meshed in arbitrary combinations, and even the data type
converted.

Once a virtual dataset has been created, it can be read just like any other
HDF5 dataset.

.. Warning::

   Virtual dataset files cannot be opened with versions of the hdf5 library
   older than 1.10.

The HDF Group has documented the VDS features in detail on the website:
`Virtual Datasets (VDS) Documentation <https://support.hdfgroup.org/HDF5/docNewFeatures/NewFeaturesVirtualDatasetDocs.html>`_.

.. _creating_vds:

Creating virtual datasets in h5py
---------------------------------

To make a virtual dataset using h5py, you need to:

1. Create a :class:`VirtualLayout` object representing the dimensions and data type
   of the virtual dataset.
2. Create a number of :class:`VirtualSource` objects, representing the datasets
   the array will be built from. These objects can be created either
   from an h5py :class:`Dataset`, or from a filename, dataset name and shape.
   This can be done even before the source file exists.
3. Map slices from the sources into the layout.
4. Convert the :class:`VirtualLayout` object into a virtual dataset in an HDF5
   file.

The following snippet creates a virtual dataset to stack
together four 1D datasets from separate files into a 2D dataset::

    layout = h5py.VirtualLayout(shape=(4, 100), dtype='i4')

    for n in range(1, 5):
        filename = "{}.h5".format(n)
        vsource = h5py.VirtualSource(filename, 'data', shape=(100,))
        layout[n - 1] = vsource

    # Add virtual dataset to output file
    with h5py.File("VDS.h5", 'w', libver='latest') as f:
        f.create_virtual_dataset('data', layout, fillvalue=-5)

This is an extract from the ``vds_simple.py`` example in the examples folder.

.. note::

   Slices up to ``h5py.h5s.UNLIMITED`` can be used to create an unlimited selection
   along a single axis. Resizing the source data along this axis will cause the
   virtual dataset to grow. E.g.::

       layout[n - 1, :UNLIMITED] = vsource[:UNLIMITED]

   A normal slice with no defined end point (``[:]``) is fixed based on the
   shape when you define it.

   .. versionadded:: 3.0

Examples
--------

In addition to the above example snippet, a few more complete examples can be
found in the examples folder:

- `vds_simple.py <https://github.com/h5py/h5py/blob/master/examples/vds_simple.py>`_
  is a self-contained, runnable example which creates four
  source files, and then maps them into a virtual dataset as shown above.
- `dataset_concatenation.py <https://github.com/h5py/h5py/blob/master/examples/dataset_concatenation.py>`_
  illustrates virtually stacking datasets together along a new axis.
- A number of examples are based on the sample use cases presented in the
  `virtual datasets RFC <https://support.hdfgroup.org/HDF5/docNewFeatures/VDS/HDF5-VDS-requirements-use-cases-2014-12-10.pdf>`__:

  - `excalibur_detector_modules.py <https://github.com/h5py/h5py/blob/master/examples/excalibur_detector_modules.py>`_
  - `dual_pco_edge.py <https://github.com/h5py/h5py/blob/master/examples/dual_pco_edge.py>`_
  - `eiger_use_case.py <https://github.com/h5py/h5py/blob/master/examples/eiger_use_case.py>`_
  - `percival_use_case.py <https://github.com/h5py/h5py/blob/master/examples/percival_use_case.py>`_

Reference
---------

.. class:: VirtualLayout(shape, dtype, maxshape=None)

   Object for building a virtual dataset.

   Instantiate this class to define a virtual dataset, assign
   :class:`VirtualSource` objects to slices of it, and then pass it to
   :meth:`Group.create_virtual_dataset` to add the virtual dataset to a file.

   This class does not allow access to the data; the virtual dataset must
   be created in a file before it can be used.

   :param tuple shape: The full shape of the virtual dataset.
   :param dtype: Numpy dtype or string.
   :param tuple maxshape: The virtual dataset is resizable up to this shape.
     Use None for axes you want to be unlimited.

.. class:: VirtualSource(path_or_dataset, name=None, shape=None, dtype=None, \
                         maxshape=None)

   Source definition for virtual data sets.

   Instantiate this class to represent an entire source dataset, and then
   slice it to indicate which regions should be used in the virtual dataset.

   When `creating a virtual dataset <creating_vds_>`_, paths to sources present
   in the same file are changed to a ".", refering to the current file (see
   `H5Pset_virtual <https://portal.hdfgroup.org/display/HDF5/H5P_SET_VIRTUAL>`_).
   This will keep such sources valid in case the file is renamed.

   :param path_or_dataset:
       The path to a file, or a :class:`Dataset` object. If a dataset is given,
       no other parameters are allowed, as the relevant values are taken from
       the dataset instead.
   :param str name:
       The name of the source dataset within the file.
   :param tuple shape:
       The full shape of the source dataset.
   :param dtype:
       Numpy dtype or string.
   :param tuple maxshape:
       The source dataset is resizable up to this shape. Use None for
       axes you want to be unlimited.
