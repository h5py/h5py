New features
------------

* With HDF5 2.4 or later, :meth:`h5py.h5d.DatasetID.read_direct_chunk` can read
  a byte range of a stored chunk: set it on a dataset transfer property list with
  :meth:`h5py.h5p.PropDXID.set_sub_chunk` and pass that as ``dxpl``. A codec
  whose stream is progressive, such as a JPEG 2000 codestream split by
  resolution, can then read only the bytes a reduced resolution needs.
* With HDF5 2.4 or later, :meth:`h5py.h5d.DatasetID.read_direct_chunk` and
  :meth:`h5py.h5d.DatasetID.get_chunk_info_by_coord` also work on a virtual
  dataset, when the offset is the image of the first element of a chunk of one
  source dataset under a mapping that is a plain translation. They return the
  source chunk.

Exposing HDF5 functions
-----------------------

* ``H5Dread_chunk2``
* ``H5Pset_sub_chunk``
* ``H5Pget_sub_chunk``
