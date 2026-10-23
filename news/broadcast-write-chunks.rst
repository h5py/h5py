New features
------------

* Writing a scalar or a small array that is broadcast over a large selection,
  such as ``dset[...] = 0`` or ``dset[:, :] = row``, is much faster for big
  datasets. The value is now repeated into a buffer of whole chunks (up to
  16 MiB, or one chunk if that is bigger) and written in chunk-aligned blocks,
  rather than with one HDF5 write per row (:issue:`1067`).
