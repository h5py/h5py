New features
------------

* Creating datasets and attributes from str or bytes objects works more
  consistently.

Deprecations
------------

* In previous versions, creating a dataset from a list of bytes objects would
  choose a fixed length string datatype to fit the biggest item. It will now
  use a variable length string datatype. To store fixed length strings, use a
  suitable dtype from :func:`h5py.string_dtype`.

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
