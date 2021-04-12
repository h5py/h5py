New features
------------

* <news item>

Deprecations
------------

* <news item>

Exposing HDF5 functions
-----------------------

* <news item>

Bug fixes
---------

* Protect :func:`h5py.h5f.get_obj_ids` against garbage collection invalidating
  HDF5 IDs while it is retrieving them (:issue:`1852`).
* Make file closing more robust, including when closing files while the
  interpreter is shutting down, by using lower-level code to close HDF5 IDs
  of objects inside the file (:issue:`1495`).

Building h5py
-------------

* <news item>

Development
-----------

* <news item>
