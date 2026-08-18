New features
------------

* Datasets with an array datatype (``H5T_ARRAY``), such as ``('f4', (3,))``,
  now support fill values. The value is broadcast to the shape of one element,
  so both ``fillvalue=7`` and ``fillvalue=[1, 2, 3]`` work, and
  :attr:`Dataset.fillvalue` reads such fill values back. The base type may be
  anything of a fixed size, including a compound type; array datatypes built on
  variable-length strings, variable-length sequences or references are not
  supported yet.

* :meth:`h5py.h5p.PropDCID.set_fill_value` and
  :meth:`h5py.h5p.PropDCID.get_fill_value` take an optional ``mtype`` argument
  giving the HDF5 datatype of the buffer, for datatypes NumPy cannot put on an
  array itself. Both now check that the buffer is large enough for the datatype.
