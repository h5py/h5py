Bug fixes
---------

* ``PropFAID.set_fapl_ros3()`` now raises ``ValueError`` if ``aws_region`` is
  longer than 32 bytes or ``secret_id``/``secret_key`` are longer than 128
  bytes. Previously an overlong value was cut off without a terminating NUL,
  so HDF5 read it on into the following field (e.g. the region picked up the
  secret ID), and shorter values were copied with a read past their end.
