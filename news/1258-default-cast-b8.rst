New features
------------

* H5T_BITFIELD types will now be cast to their ``numpy.uint`` equivalent by default
  (:issue:`1258`). This means that no knowledge of mixed type compound dataset
  schemas is required to read these types, and can simply be read as follows:

  .. code::

     arr = dset[:]

  Alternatively, 8-bit bitfields can still be cast to booleans explicitly:

  .. code::

     arr = dset.astype(numpy.bool_)[:]

Deprecations
------------

* <news item>

Exposing HDF5 functions
-----------------------

* <news item>

Bug fixes
---------

* <news item>

Building h5py
-------------

* <news item>

Development
-----------

* <news item>
