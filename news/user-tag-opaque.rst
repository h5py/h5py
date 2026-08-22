New features
------------

* Support for user supplied tags for opaque datatypes. :func:`h5py.opaque_dtype`
  now has a keyword ``tag`` argument to set a custom HDF5 opaque datatype
  tag. A new :func:`h5py.get_opaque_tag` helper returns the tag from a dtype,
  and :attr:`h5py.Datatype.opaque_tag` exposes it on committed opaque datatypes.
  Opaque datasets written by other tools (with a non-``NUMPY:`` tag) are now
  recognised by :func:`~h5py.check_opaque_dtype` on read, and their tag is
  available via :func:`~h5py.get_opaque_tag`. See :ref:`opaque_dtypes`.
