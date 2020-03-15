Building h5py
-------------

* use bare ``char*`` instead of ``array.array`` in h5d.read_direct_chunk since
  ``array.array`` is a private CPython C-API interface

* define ``NPY_NO_DEPRECATED_API`` to quiet a warning

* use ``dtype=object`` in tests with ragged arrays
