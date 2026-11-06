New features
------------

* ``h5py.h5p.PropDAID.set_chunk_cache()`` now accepts ``None`` for any of its
  three parameters. A parameter given as ``None`` is left unset on the
  property list, which is how HDF5 is told that the dataset should take that
  value from the file it lives in. All three were previously required.

Bug fixes
---------

* Giving one per-dataset chunk cache option no longer pins the other two.
  Passing any of ``rdcc_nslots``, ``rdcc_nbytes`` or ``rdcc_w0`` to
  ``Group.create_dataset()`` or ``Group.require_dataset()`` used to fill the
  remaining options in from HDF5's library defaults, which bear no relation to
  the file being used. They are now left unset, so the dataset inherits them
  from its file, the behaviour ``H5Pset_chunk_cache()`` documents. This
  changes the cache a dataset ends up with whenever the file was opened with
  cache settings of its own: ``create_dataset(..., rdcc_nbytes=2048)`` on a
  file opened with ``rdcc_nslots=521`` now gives the dataset 521 slots rather
  than HDF5's default of 8191.
* A dataset access property list supplied through the ``dapl`` argument of
  ``Group.create_dataset()`` or ``Group.require_dataset()`` is no longer
  modified in place. One list can now be reused across several calls without
  collecting the options passed to each of them. Cache values already on such
  a list still apply to any option not named in the call, since nothing in the
  HDF5 API distinguishes a value that was set from one that was never touched.
