Bug fixes
---------

* Files opened from a Python file-like object now handle ``readinto()`` and
  ``read()`` returning fewer bytes than requested, which raw streams are
  allowed to do. Previously the rest of the read was skipped without an
  error, so HDF5 used whatever was left in its buffer: datasets came back
  with wrong values, which could include data from a file read earlier in
  the same process. Reads past the end of the file now give zeros, as with
  the default driver.
