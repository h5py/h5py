Bug fixes
---------

* Respect zero values for per-dataset chunk cache settings in
  ``Group.create_dataset()`` and ``Group.require_dataset()``. Setting
  ``rdcc_nbytes`` or ``rdcc_nslots`` to zero disables the cache, and setting
  ``rdcc_w0`` to zero selects strict LRU preemption.
