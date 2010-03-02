.. _build:

******************
Installation guide
******************

Where to get h5py
=================

Downloads for all platforms are available at http://h5py.googlecode.com.
Tar files are available for UNIX-like systems (Linux and Mac OS-X), and
a binary installer for Windows which includes HDF5 1.8.  You can also
install h5py via easy_install on UNIX, and via MacPorts.

Getting HDF5
============

On Windows, HDF5 is provided as part of the integrated
installer for h5py.  

On Linux and OS-X, you must provide HDF5 yourself.  HDF5 versions **1.6.5**
through **1.8.3** are supported. **The best solution is
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
=====================

Requires
--------

- Python 2.5
- NumPy_ 1.0.3 or higher

Download the executable installer from `Google Code`__ and run it.  This
installs h5py and a private copy of HDF5 1.8.

__ http://h5py.googlecode.com


.. _linux:

Installing on Linux/Mac OS-X
============================

This package is designed to be installed from source.  You will need
Python and a C compiler, for setuptools to build the extensions.

Requires
--------
- Python 2.5 or 2.6, including headers ("python-dev")
- Numpy_ 1.0.3 or higher
- HDF5_ 1.6.5 or higher, including 1.8.X versions

.. _Numpy: http://numpy.scipy.org/
.. _HDF5: http://www.hdfgroup.com/HDF5

Please note that Cython (or Pyrex) is *not* required to build h5py.

.. note::
    Due to API changes, HDF5 1.8.0 and 1.8.1 will run in "HDF5 1.6 emulation
    mode".  If you want to use all the new features in HDF5 1.8, please
    install HDF5 1.8.2 or later.

Quick installation
------------------

H5py can now be automatically installed by setuptools' easy_install command.
You don't need to download anything; just run the command::

    $ [sudo] easy_install h5py

Alternatively, you can install in the traditional manner by downloading the
most recent tarball of h5py, uncompressing it, and running setup.py::

    $ python setup.py build
    $ [sudo] python setup.py install


Custom installation
-------------------

Sometimes h5py may not be able to determine what version of HDF5 is installed.
Also, sometimes HDF5 may be installed in an unusual location.  When using
setup.py directly, you can specify both your version of HDF5 and its location
through the ``configure`` command::

    $ python setup.py configure [--hdf5=/path/to/hdf5] [--api=<16 or 18>]
    $ python setup.py build
    $ [sudo] python setup.py install

The HDF5 directory you specify should contain sub-directories like "include",
"lib" and "bin".

Alternatively (for example, if installing with easy_install), you can use
environment variables::

    $ export HDF5_DIR=/path/to/hdf5
    $ export HDF5_API=<16 or 18>
    $ easy_install h5py

Keep in mind that on some platforms, ``sudo`` will filter out your environment
variables.  If you need to be a superuser to run easy_install, you might
want to issue all three of these commands in a root shell.

Settings issued with the ``configure`` command will always override those set
with environment variables.  Also, for technical reasons the configure command
must be run by itself, before any build commands.

The standard command::

    $ python setup.py clean

will clean up all temporary files, including the output of ``configure``.

Testing
=======

Running unit tests can help diagnose problems unique to your platform or
software configuration.  For the Unix version of h5py, running the command::

    $ python setup.py test

before installing will run the h5py test suite.  On both Unix and Windows
platforms, the tests may also be run after installation:

    >>> import h5py.tests
    >>> h5py.tests.runtests()

Please report any failing tests to "h5py at alfven dot org", or file an issue
report at http://h5py.googlecode.com.














