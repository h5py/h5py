New features
------------

* Support for storing NumPy complex numbers in HDF5 files using C99 complex
  number datatypes. This feature is available starting with HDF5 library 2.0.0
  if the compiler and platform it was built on implement the relevant part of
  the C99 language standard. Otherwise, the NumPy complex numbers will be stored
  using the already present method as an HDF5 compound datatype with two
  floating-point fields. Reading complex number data from both variants of the
  HDF5 datatypes will work as expected.

  The new HDF5 datatypes and their NumPy dtype equivalent:
  ``H5T_COMPLEX_IEEE_F32LE`` (``<c8``), ``H5T_COMPLEX_IEEE_F32LBE`` (``>c8``),
  ``H5T_COMPLEX_IEEE_F64LE`` (``<c16``), ``H5T_COMPLEX_IEEE_F64BE`` (``<c16``),
  ``H5T_NATIVE_FLOAT_COMPLEX`` (``=c8``), ``H5T_NATIVE_DOUBLE_COMPLEX``
  (``=c16``), and ``H5T_NATIVE_LDOUBLE_COMPLEX`` if ``numpy.complex256`` is
  available.
