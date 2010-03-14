
.. _h5pyreference:

***********************
Reference Documentation
***********************

.. module:: h5py.highlevel

The high-level interface is the most convenient method to talk to HDF5.  There
are three main abstractions: files, groups, and datasets. Each is documented
separately below.

.. toctree::
    :maxdepth: 2

    file
    group
    dataset
    attr
    refs
    vl
    other

General information about h5py and HDF5
=======================================

Paths in HDF5
-------------

HDF5 files are organized like a filesystem.  :class:`Group` objects work like
directories; they can contain other groups, and :class:`Dataset` objects.  Like
a POSIX filesystem, objects are specified by ``/``-separated names, with the
root group ``/`` (represented by the :class:`File` class) at the base.

Wherever a name or path is called for, it may be relative or absolute.
Unfortunately, the construct ``..`` (parent group) is not allowed.

Exceptions
----------

As of version 1.2, h5py uses a "hybrid" exception system.  When an error is
detected inside HDF5, an exception is raised which inherits from both a
standard Python exception (TypeError, ValueError, etc), and an HDF5-specific
exception class (a subclass of H5Error).

It's recommended that you use the standard Python exceptions in you code;
for example, when indexing a Group object:

    >>> try:
    >>>     grp = mygroup[name]
    >>> except KeyError:
    >>>     print 'Group "%s" does not exist' % name

Library configuration
---------------------

A few library options are available to change the behavior of the library.
You can get a reference to the global library configuration object via the
function ``h5py.get_config()``.  This object supports the following attributes:

    **complex_names**
        Set to a 2-tuple of strings (real, imag) to control how complex numbers
        are saved.  The default is ('r','i').

    **bool_names**
        Booleans are saved as HDF5 enums.  Set this to a 2-tuple of strings
        (false, true) to control the names used in the enum.  The default
        is ("FALSE", "TRUE").

Threading
---------

H5py is now always thread-safe.  However, as HDF5 does not support thread-level
concurrency (and as it is not necessarily thread-safe), access to the library
is automatically serialized.  The GIL is released around read/write operations
so that non-HDF5 threads (GUIs, computation) can continue to execute.



