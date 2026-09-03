Bug fixes
---------

* Fixed an HDF5 type identifier leak in ``make_reduced_type`` (used when
  reading compound datasets with a ``fields`` selection): if inserting a
  member into the reduced compound type failed partway through, the newly
  created type was never closed.
