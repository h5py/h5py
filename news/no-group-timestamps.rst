New features
------------

* A new `track_times` parameter when creating a group or a file to control
  whether creation, modification, change and access timestamps are stored
  for group objects. This is False by default.

Deprecations & breaking changes
-------------------------------

* Timestamps are no longer stored by default for groups (including the root group)
  if the `track_order` parameter is set. Previously, setting this parameter also
  caused timestamps to be stored in the file.

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
