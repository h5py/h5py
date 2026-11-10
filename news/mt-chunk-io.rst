New features
------------

* Added ``h5py.set_internal_threads()``, which enables and sizes a thread pool
  that HDF5 uses to read the chunks of a chunked dataset concurrently. It
  requires HDF5 2.3.0 or later built with a threading package. HDF5 2.3.0
  still serializes this work internally, so no speedup is expected yet. See
  :ref:`internal_threads`.

Exposing HDF5 functions
-----------------------

* ``H5TSset_internal_threads``, ``H5Pset_io_threads`` and ``H5Pget_io_threads``
  (requires HDF5 2.3.0 or later).

Building h5py
-------------

* Set ``H5PY_INTERNAL_THREADS`` to ``1`` or ``0`` to override automatic detection
  and build h5py with or without support for HDF5's internal thread pool.
