New features
------------

* On Emscripten/WebAssembly (e.g. Pyodide), h5py now calls HDF5's
  ``H5dont_atexit()`` before its first HDF5 call, preventing crashes at
  interpreter exit when several wheels bundle their own statically linked
  copies of HDF5.  The new environment variable ``H5PY_DONT_ATEXIT``
  (``1``/``0``) forces this behaviour on or off on any platform, see
  :doc:`/config`.
