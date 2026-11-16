Bug fixes
---------

* Fixed a heap buffer overflow in :meth:`h5py.h5d.DatasetID.get_chunk_info`
  when the ``space`` argument has a lower rank than the dataset. The chunk
  offset buffer was sized from that space, but HDF5 always fills in one offset
  per dataset dimension, so it wrote past the end. The buffer is now sized from
  the dataset, and the returned ``chunk_offset`` always has the dataset's rank.
