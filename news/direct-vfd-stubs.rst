New features
------------

* Provide the ability to use the Direct Virtual File Driver (VFD) from
  HDF5 (Linux only).
  If the Direct VFD driver is present at the time of compilation, users can use the
  Direct VFD by passing the keyword argument ``driver="direct"`` to the
  ``h5py.File`` constructor.

  To use the Direct VFD HDF5 and h5py must have both been compiled with 
  the Direct VFD Driver enabled. Currently, h5py as released on pypi
  does not include the Direct VFD driver.
  Other packages such as the conda package on conda-forge might include it.
  Alternatively, you can build h5py from source against an HDF5 build 
  with the direct driver enabled.

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
