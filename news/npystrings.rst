New features
------------

* On NumPy 2.x, it is now possible to read and write native NumPy variable-width
  strings, a.k.a. NpyStrings (``dtype=np.dtypes.StringDType()`` or ``dtype='T'``
  for short), which are much faster than object strings. For the sake of
  interoperability with NumPy 1.x, users need to explicitly opt in.
  See :ref:`npystrings`.

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

* NpyStrings introduce no changes in the build process: you need NumPy 2.x to build
  (as before), and the same binary wheels are backwards compatible with NumPy 1.x.

Development
-----------

* <news item>
