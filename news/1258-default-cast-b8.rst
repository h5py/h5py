New features
------------

* H5T_BITFIELD types will now be cast to their ``numpy.uint`` equivelant by default
  (:issue:`1258`). This means that no knowledge of mixed type compound dataset
  schemas is required to read these types, and can simply be read as follows:

  .. code::

     arr = dset[:]

  Alternatively, these types can still be cast to ``numpy.bool`` explicitly:

  .. code::

     with dset.astype(numpy.bool):
        arr = dset[:]

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
