.. _build:

Downloading and Building h5py
=============================

Downloads for all platforms are available at http://h5py.googlecode.com.
Tar files are available for UNIX-like systems (Linux and Mac OS-X), and
a binary installer for Windows which includes HDF5 1.8.  You can also
install h5py via easy_install on UNIX, and via MacPorts.

Getting HDF5
------------

On Windows, HDF5 is provided as part of the integrated
installer for h5py.  

On Linux and OS-X, you must provide HDF5 yourself.  HDF5 versions **1.8.3**
and higher are supported.  **The best solution is
to install HDF5 via a package manager like apt, yum or fink.** Regardless of
how you decide to install HDF5, keep the following in mind:

* You'll need the development headers in addition to the library; sometimes
  package managers list this separately as ``libhdf5-dev`` or similar.

* HDF5 **must** be available as a shared library (``libhdf5.so.X``), or
  programs like h5py can't work.  In particular, the "static" distribution
  available from the HDF Group web site **will not work**.

* If you've manually installed one of the SZIP-aware builds from the HDF Group
  web site, be sure to also install the SZIP libraries.

The HDF Group downloads are located at http://www.hdfgroup.com/HDF5 .


.. _windows:

Installing on Windows
---------------------

Requires
^^^^^^^^

- Python 2.6, 2.7 or 3.2
- Any modern version of Numpy

Download the executable installer from `Google Code`__ and run it.  This
installs h5py and a private copy of HDF5 1.8.

__ http://h5py.googlecode.com


.. _linux:

Installing on Linux/Mac OS-X
----------------------------

This package is designed to be installed from source.  You will need
Python and a C compiler.  Setuptools and Cython are *not* required.

Requires
^^^^^^^^

- Python 2.6, 2.7 or 3.2
- C headers for Python (usually "python-dev" or similar)
- Numpy_ 1.0.3 or higher
- HDF5_ 1.8.3 or higher (with headers, "libhdf5-dev" or similar)

.. _Numpy: http://numpy.scipy.org/
.. _HDF5: http://www.hdfgroup.com/HDF5


Quick installation
^^^^^^^^^^^^^^^^^^

H5py can now be automatically installed, for example with setuptools'
easy_install command.  You don't need to download anything; just run the
command::

    $ [sudo] easy_install h5py

Alternatively, you can install in the traditional manner by downloading the
most recent tarball of h5py, uncompressing it, and running setup.py::

    $ python setup.py build
    $ [sudo] python setup.py install


Custom installation
^^^^^^^^^^^^^^^^^^^

Sometimes h5py may not be able to determine what version of HDF5 is installed.
Also, sometimes HDF5 may be installed in an unusual location.  When using
setup.py directly, you can specify the location of the HDF5 library:

    $ python setup.py build --hdf5=/path/to/hdf5
    $ [sudo] python setup.py install

The HDF5 directory you specify should contain sub-directories like "include",
"lib" and "bin".

Alternatively (for example, if installing with easy_install), you can use
environment variables::

    $ export HDF5_DIR=/path/to/hdf5
    $ easy_install h5py

Keep in mind that on some platforms, ``sudo`` will filter out your environment
variables.  If you need to be a superuser to run easy_install, you might
want to issue all three of these commands in a root shell.


Testing
-------

Running unit tests can help both you and the entire h5py community.  If you're
installing from a tarball we strongly recommend running the test suite
first::

    $ python setup.py test

Please report any failing tests to the mailing list (h5py at googlegroups.com),
or file a bug report at http://h5py.googlecode.com.














