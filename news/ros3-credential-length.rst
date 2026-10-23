Bug fixes
---------

* ``PropFAID.set_fapl_ros3()`` now raises ``ValueError`` if ``aws_region`` is
  longer than 32 bytes or ``secret_id``/``secret_key`` are longer than 128
  bytes. Previously an overlong value was cut off without a terminating NUL,
  so HDF5 read it on into the following field (e.g. the region picked up the
  secret ID), and for ``secret_key``, the last field, ``get_fapl_ros3()`` read
  past the end of the driver info allocation into unrelated heap memory.
  Shorter values were also copied with a read past their end.
