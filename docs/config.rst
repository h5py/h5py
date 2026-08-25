Configuring h5py
================

Library configuration
---------------------

A few library options are available to change the behavior of the library.
You can get a reference to the global library configuration object via the
function :func:`h5py.get_config`.  This object supports the following
attributes:

    **complex_names**
        Set to a 2-tuple of strings (real, imag) to control how complex numbers
        are saved.  The default is ('r','i').

    **bool_names**
        Booleans are saved as HDF5 enums.  Set this to a 2-tuple of strings
        (false, true) to control the names used in the enum.  The default
        is ("FALSE", "TRUE").

    **track_order**
        Whether to track dataset/group/attribute creation order.  If
        container creation order is tracked, its links and attributes
        are iterated in ascending creation order (consistent with
        ``dict`` in Python 3.7+); otherwise in ascending alphanumeric
        order.  Global configuration value can be overridden for
        particular container by specifying ``track_order`` argument to
        :class:`h5py.File`, :meth:`h5py.Group.create_group`,
        :meth:`h5py.Group.create_dataset`.  The default is ``False``.

Environment variables
---------------------

A few aspects of h5py's runtime behaviour can be controlled with environment
variables, which are read when :mod:`h5py` is first imported:

    **H5PY_DONT_ATEXIT**
        If set to ``1``, h5py calls the HDF5 function ``H5dont_atexit()``
        before any other HDF5 function, so that the HDF5 library does not
        install its own cleanup routines to run when the process exits.
        If set to ``0``, HDF5's normal exit cleanup is left enabled.

        If the variable is unset (or empty), HDF5's cleanup is left enabled
        on all platforms except Emscripten/WebAssembly (e.g. Pyodide), where
        h5py disables it by default: there, several packages may each bundle
        their own statically linked copy of HDF5, and running more than one
        copy's cleanup can crash the interpreter at exit.

Environment variables affecting how h5py is *built* are described in
:ref:`custom_install`.
