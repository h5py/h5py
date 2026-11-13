Bug fixes
---------

* Preserve exceptions from a Python file object's ``seek()`` or ``tell()``
  methods when HDF5 retries querying the end of the file (:issue:`2221`).
