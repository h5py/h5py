*********
Threading
*********

Threading is an issue in h5py because HDF5 doesn't support thread-level
concurrency.  Some versions of HDF5 are not even thread-safe.  The package
tries to hide as much of these problems as possible using a combination of
the GIL and Python-side reentrant locks.

High-level
----------

The objects in h5py.highlevel (File, Dataset, etc) are always thread-safe.  You
don't need to do any explicit locking, regardless of how the library is
configured.

Low-level
---------

The low-level API (h5py.h5*) is also thread-safe, unless you use the
experimental non-blocking option to compile h5py.  Then, and only then, you
must acquire a global lock before calling into the low-level API.  This lock
is available on the global configuration object at "h5py.config.lock".  The
decorator "h5sync" in h5py.extras can wrap functions to do this automatically.


Non-Blocking Routines
---------------------

By default, all low-level HDF5 routines will lock the entire interpreter
until they complete, even in the case of lengthy I/O operations.  This is
unnecessarily restrictive, as it means even non-HDF5 threads cannot execute.

When the package is compiled with the option ``--io-nonblock``, a few C methods
involving I/O will release the global interpreter lock.  These methods always
acquire the global HDF5 lock before yielding control to other threads.  While
another thread seeking to acquire the HDF5 lock will block until the write
completes, other Python threads (GUIs, pure computation threads, etc) will
execute in a normal fashion.

However, this defeats the thread safety provided by the GIL.  If another thread
skips acquiring the HDF5 lock and blindly calls a low-level HDF5 routine while
such I/O is in progress, the results are undefined.  In the worst case,
irreversible data corruption and/or a crash of the interpreter is possible.
Therefore, it's very important to always acquire the global HDF5 lock before
calling into the h5py.h5* API when all the following are true:

    1. More than one thread is performing HDF5 operations
    2. Non-blocking I/O is enabled

This is not an issue for the h5py.highlevel components (Dataset, Group,
File objects, etc.) as they acquire the lock automatically.

The following operations will release the GIL during I/O:
    
    * DatasetID.read
    * DatasetID.write


Customizing the lock type
-------------------------

Applications that use h5py may have their own threading systems.  Since the
h5py locks are acquired and released alongside application code, you can
set the type of lock used internally by h5py.  The lock is stored as settable
property "h5py.config.lock" and should be a lock instance (not a constructor)
which provides the following methods:

    * __enter__
    * __exit__
    * acquire
    * release

The default lock type is the native Python threading.RLock, but h5py makes no
assumptions about the behavior or implementation of locks beyond reentrance and
the existence of the four required methods above.

It remains to be seen whether this is even necessary.  In future versions of
h5py, this attribute may disappear or become non-writable.







