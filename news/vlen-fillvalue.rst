Bug fixes
---------

* Setting and reading fill values for the following datatypes: array
  datatypes (``H5T_ARRAY``), variable-length sequences, and compound datatypes
  with a variable-length member. Previously, setting a fill value on a compound
  with a variable-length member wrote a file that no HDF5 tool could read back,
  reporting no error at the time. Reading a stored fill value of a
  variable-length sequence datatype aborted the process. Array datatypes were
  rejected outright, and reading a variable-length string fill value leaked its
  buffer on every access. Fill values are now converted with the same machinery
  used for dataset data.

Exposing HDF5 functions
-----------------------

* :meth:`h5py.h5p.PropDCID.set_fill_value` and
  :meth:`h5py.h5p.PropDCID.get_fill_value` take an optional ``dtype`` argument,
  for array datatypes that NumPy cannot express as ``ndarray.dtype``, naming the
  NumPy datatype the buffer describes instead. Both now also check the buffer is large
  enough.
