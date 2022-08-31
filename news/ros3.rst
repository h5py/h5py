New features
------------

* When using the ros3 driver, AWS authentication will be activated only if all
  three driver arguments are provided. Previously AWS authentication was active
  if any one of the arguments was set causing an error from the HDF5 library.
* HDF5 file names for ros3 driver can now also be s3:// resource locations. H5py
  will translate them into AWS path-style URLs for use by the driver.

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
