Bug fixes
---------

* Respect zero values for per-dataset chunk cache settings in
  ``Group.create_dataset()`` and ``Group.require_dataset()``. Setting
  ``rdcc_nbytes`` or ``rdcc_nslots`` to zero disables the cache, and setting
  ``rdcc_w0`` to zero selects strict LRU preemption.
  Supplying any of these options currently resets unspecified cache options
  to HDF5 defaults rather than inheriting the file's settings.
* Accept empty ``efile_prefix`` and ``virtual_prefix`` values in
  ``Group.create_dataset()`` and ``Group.require_dataset()`` without raising
  ``AttributeError``.
