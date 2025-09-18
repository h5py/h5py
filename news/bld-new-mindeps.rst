Building h5py
-------------

* The minimum versions build-time Python requirements were updated to
  ``Cython==3.0.0`` (up from ``0.29.1``), and ``numpy==1.25.0`` (down from
  ``2.0.0``). We still recommend building with numpy 2 or newer whenever
  possible, this is done to improve support for external package ecosystems,
  where the system uses the same version of packages for building and
  installation.
