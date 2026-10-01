.. currentmodule:: h5py
.. _exceptions:


Exceptions
==========

h5py does not define its own exception hierarchy.  Instead, it raises the
standard Python exceptions described below.  This makes it possible to catch
errors from h5py using the ordinary builtin exception classes, without
importing anything from h5py.

Errors reported by the underlying HDF5 C library are surfaced as one of these
Python exceptions; the original HDF5 error message is preserved in the
exception's message.

Built-in exceptions raised by the high-level API
------------------------------------------------

The following exceptions are raised by the high-level interface
(``h5py.File``, ``h5py.Group``, ``h5py.Dataset``, ``h5py.AttributeManager``,
and the ``h5py.h5*`` low-level modules).

``FileNotFoundError``
    Raised when opening a file in a mode that requires it to exist, but no
    such file is present.  For example::

        >>> h5py.File("does-not-exist.h5", "r")
        Traceback (most recent call last):
          ...
        FileNotFoundError: [Errno 2] Unable to open file (unable to open file: name = 'does-not-exist.h5', errno = 2, ...)

``KeyError``
    Raised when looking up a name that does not exist in a group or an
    attribute namespace.  For example::

        >>> f = h5py.File("test.h5", "w")
        >>> f["missing"]
        Traceback (most recent call last):
          ...
        KeyError: "Unable to synchronously open object (object 'missing' doesn't exist)"

``ValueError``
    Raised for a value that is well-typed but invalid in context.  This is the
    most common exception raised by the high-level API, and covers cases such
    as an invalid slice, an unknown file mode, or an incompatible shape.

``TypeError``
    Raised when an argument has the wrong type, or when an operation is not
    supported for the given object.  For example, using ``read_direct`` on an
    empty dataset, where the target array cannot match the dataset shape::

        >>> dset = f.create_dataset("empty", shape=(0,), dtype="f4")
        >>> dset.read_direct(np.zeros(1))
        Traceback (most recent call last):
          ...
        TypeError: Can't broadcast (0,) -> (1,)

``IndexError``
    Raised when an index is out of range for a dataset or a shape tuple.

``RuntimeError``
    Raised when the underlying HDF5 library reports a failure that does not
    map to one of the more specific exceptions above.

``NotImplementedError``
    Raised when a feature is not available in the version of the HDF5 library
    that h5py was built against.

``OSError``
    Raised for general I/O failures reported by the operating system or by
    HDF5.

``OverflowError``
    Raised when a value does not fit in the target datatype.

Low-level exceptions
--------------------

The low-level interface (``h5py.h5*``) raises the same builtin exceptions.
By default, h5py prints the HDF5 error stack to stderr whenever a low-level
call fails.  This can be controlled with the private ``h5py._errors`` module::

    import h5py._errors

    h5py._errors.silence_errors()    # stop printing HDF5 error stacks
    h5py._errors.unsilence_errors()  # resume printing them

These functions are not part of the public API and may change between
releases.

Catching errors
---------------

Because h5py uses builtin exceptions, a catch-all ``except Exception`` is
rarely necessary.  Prefer the specific class::

    import h5py

    try:
        with h5py.File("data.h5", "r") as f:
            dset = f["measurements"]
    except FileNotFoundError:
        ...  # the file is not there
    except KeyError:
        ...  # the file is there, but has no "measurements" dataset
