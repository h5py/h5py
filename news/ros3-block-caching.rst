New features
------------

* The Read-Only S3 ('ros3') driver can now be told how to read and cache blocks
  of a file, using three new :class:`h5py.File` keywords: ``block_size``,
  ``block_cache_size`` and ``lock_superblock``. Instead of sending a separate
  request to the object store for every read, the driver fetches the file in
  fixed-size blocks and holds recently used blocks in memory, so that reads
  landing in a block that has already been fetched are served without going back
  to the network. This tends to help most with files that were not written with
  cloud access in mind, where the metadata a reader needs is scattered across
  many small regions of the file. This feature requires HDF5 2.2.0 or later, and
  using it with an older library raises a ``ValueError``. By default the driver
  reads in blocks of 16 MiB and lets the cache grow to 128 MiB.

Exposing HDF5 functions
-----------------------

* ``H5Pget_fapl_ros3_block_caching`` & ``H5Pset_fapl_ros3_block_caching`` (where
  HDF5 is built with read-only S3 support, and is version 2.2.0 or later).
