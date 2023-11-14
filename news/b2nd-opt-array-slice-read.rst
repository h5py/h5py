New features
------------

* New optional support for Blosc2 NDim optimized slicing (:pr:`NNNN`). Common
  simple slicing operations on homogeneous multi-dimensional datasets
  compressed with Blosc2 are handled via direct chunk access and only the
  affected blocks (a subdivision of chunks) are processed, avoiding HDF5
  filter pipeline overhead and whole chunk decompression, and resulting in
  2x-3x speedups (for the moment).  Other slice/read/write operations resort
  to the filter pipeline.

Building h5py
-------------

* If you want to enable support for Blosc2 optimized slicing in your build
  using ``pip`` or similar, you may use the ``blosc2`` extra: e.g. ``pip
  install h5py[blosc2]`` (:pr:`NNNN`). This will also enable HDF5 filter
  pipeline-based writing and reading of Blosc2-compressed datasets (via
  hdf5plugin) whenever the optimized path is not available.
