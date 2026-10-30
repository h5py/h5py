Bug fixes
---------

* ``h5t.TypeEnumID`` methods (``enum_insert``, ``enum_nameof``,
  ``enum_valueof`` and ``get_member_value``) read and write enum member values
  through an 8-byte ``long long`` buffer. HDF5 allows an enum base type wider
  than that (a >64-bit integer), including one loaded from a committed enum
  datatype in a file, which made the ``H5Tget_member_value`` /
  ``H5Tenum_valueof`` / ``H5Tconvert`` calls write past the buffer. Such base
  types are now rejected with ``ValueError`` before any access.
