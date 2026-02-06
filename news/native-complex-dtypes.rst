New features
------------

* Support for storing NumPy complex numbers in HDF5 files using C99 complex
  number datatypes. This feature is available starting with HDF5 library 2.0.0
  if the compiler and platform it was built on implement the relevant part of
  the C99 language standard.

  h5py will still convert NumPy complex numbers into an HDF5 compound datatype
  by default, preserving the behaviour from earlier versions. This is likely to
  change in a future major version. To create datasets or attributes using the
  new native complex datatypes, pass one as the ``dtype`` parameter::

    arr = np.arange(10, dtype='<c16')
    f.create_dataset('complex', data=arr, dtype=h5py.h5t.COMPLEX_IEEE_F64LE)
    f.attrs.create('complex', arr, dtype=h5py.h5t.COMPLEX_IEEE_F64LE)

  These are available in little-endian (LE) & big-endian (BE) versions, based
  on 16, 32 & 64 bit floating point numbers. The size of each complex dtype
  is double the named float type, so ``COMPLEX_IEEE_F64LE`` is 128 bits
  (16 bytes).

  Once a dataset or attribute exists with a complex datatype, reading and
  writing it with the corresponding NumPy complex dtype should work with no
  special steps.
