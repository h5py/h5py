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

.. _internal_threads:

Internal multi-threading
------------------------
HDF5 2.3.0 and later can read the chunks of a chunked dataset concurrently on a
thread pool of its own. That parallelises work inside a single h5py call,
underneath ``phil``, so the limitation above does not apply to it. HDF5 builds
this pool whenever a threading package is available, except for a static library
on Windows.

The pool is process-global and starts out disabled:

.. code-block:: python

    import h5py
    h5py.set_internal_threads(4)

Pass ``0`` to turn internal threading off again. Changing the count destroys and
recreates the pool, so it is better to set it once than to tune it between
reads. h5py holds ``phil`` while changing it, so this is safe with respect to
other threads using h5py, but not if another library in the same process (e.g.
PyTables) is using the same HDF5 library at that time.

Only reads of chunked datasets use the thread pool, and only for chunks that are
not already in the chunk cache. For an unfiltered dataset, the chunks must also
fit in the chunk cache (8 MiB by default; see ``rdcc_nbytes`` in :class:`File`).

Internally threaded reads are disabled when the file uses an MPI-based driver
such as ``mpio``, and when the library was built with ``HDF5_ENABLE_THREADSAFE``.

The pool on HDF5's side does not handle forks at present - a forked child
inherits the pool but not its threads, and would hang on its first threaded
read. h5py works around this by shutting the pool down for the duration of
:func:`os.fork` (which :mod:`multiprocessing` uses) and restoring it in the
parent afterwards. The child starts with the pool off, and can enable its own.
Forks made from C code bypass :func:`os.fork`, so they are not covered, so
``h5py.set_internal_threads(0)`` should be called before those.

``h5py.get_config().internal_threads`` reports whether the feature is available.

.. note::
   As of HDF5 2.3.0 these threaded reads are still serialized internally, so
   there is no expected speedup at present.

.. _free_threading:

free-threading
--------------
h5py can be compiled for free-threading Python interpreters (e.g. 3.14t), where the
GIL has been disabled. This does not disable the ``phil`` global lock, though, as
such lock protects against race conditions in libhdf5 and not in the Python interpreter.

Free-threading stability is, as of h5py 3.15.0, largely untested. For this reason,
free-threading wheels (e.g. ``cp314t``) are not published on PyPI and the functionality is
only intended for experimental use.
