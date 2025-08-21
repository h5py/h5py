Multi-threading
===============
libhdf5, at least with the default compilation flags, is not thread-safe.

h5py supports multi-threading, in the sense that you can use h5py from multiple threads
without taking any special precautions. This is enabled by wrapping all critical
sections that invoke the libhdf5 C API with an internal *interpreter-wide reentrant
lock*, to be used either as a context manager or as a function decorator:

.. code-block:: python

    from h5py._objects import phil, with_phil

    with phil:
        # Calls to libhdf5 C API here

    @with_phil
    def my_function():
        # Calls to libhdf5 C API here

The global lock means that, while it is possible to pipeline h5py and non-h5py
work across multiple threads (e.g. read/write to a file and process the data),
multiple calls to the h5py API will not run in parallel - not even if they operate
on different datasets or different files.


free-threading
--------------
h5py can be compiled for free-threading Python interpreters (e.g. 3.14t), where the
GIL has been disabled. This does not disable the ``phil`` global lock, though, as
such lock protects against race conditions in libhdf5 and not in the Python interpreter.

Free-threading stability is, as of h5py 3.15.0, largely untested. For this reason,
free-threading wheels (e.g. ``cp314t``) are not published on PyPI and the functionality is
only intended for experimental use.
