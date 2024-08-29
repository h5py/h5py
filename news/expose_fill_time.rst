Breaking Changes and Deprecations
------------

* Fill time for chunked storage was set to ``h5d.FILL_TIME_ALLOC``. Now this
  is handled by HDF5 library where the default is ``h5d.FILL_TIME_IFSET``
  (equivalent to ``fill_time='ifset'``). Please use ``fill_time='alloc'`` if
  the behaviour in previous releases is wanted.

Exposing HDF5 functions
-----------------------

* Expose fill time option in dataset creation property list via the
  ``fill_time`` parameter in ``create_dataset``.
