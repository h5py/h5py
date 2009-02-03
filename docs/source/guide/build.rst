.. _build:

******************
Installation guide
******************

Where to get h5py
=================

Downloads for all platforms are available at http://h5py.googlecode.com.
Tar files are available for UNIX-like systems (Linux and Mac OS-X), and
a binary installer for Windows which includes HDF5 1.8.  As of version 1.1,
h5py can also be installed via easy_install.

Getting HDF5
============

On Windows, HDF5 is provided as part of the integrated
installer for h5py.  On Linux and OS-X, you
must provide HDF5 yourself.  HDF5 versions **1.6.5** through **1.8.2** are
supported.

**The best solution for both Linux and OS-X is to install HDF5 via a
package manager.** If you decide to build HDF5 from source, be sure to
build it as a dynamic library.

`The HDF Group`__ provides several "dumb" (untar in **/**) binary distributions
for Linux, but traditionally only static libraries for Mac.  Mac OS-X users
should use something like Fink, or compile HDF5 from source.

__ http://www.hdfgroup.com/HDF5


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


Quick installation
------------------

H5py can now be automatically installed by setuptools' easy_install command::

    $ [sudo] easy_install h5py

Alternatively, you can install in the traditional manner by running setup.py::

    $ python setup.py build
    $ [sudo] python setup.py install


Custom installation
-------------------

Sometimes h5py may not be able to determine what version of HDF5 is installed.
Also, sometimes HDF5 may be installed in an unusual location.  You can
specify both your version of HDF5 and its location through the ``configure``
command::

    $ python setup.py configure [--hdf5=/path/to/hdf5] [--api=<16 or 18>]
    $ python setup.py build
    $ [sudo] python setup.py install

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

Problems
========

If you have trouble installing or using h5py, first read the FAQ at
http://h5py.googlecode.com for common issues.  You are also welcome to
open a new bug there, or email me directly at "h5py at alfven dot org".
Enjoy!














