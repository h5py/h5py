******************
Installation guide
******************

Where to get h5py
=================

See the download page at `the Google Code site`__.  If installing on Windows,
be sure to get a version that matches your Python version (2.5 or 2.6).

__ http://h5py.googlecode.com

Getting HDF5
============

On :ref:`Windows <windows>`, HDF5 is provided as part of the integrated
installer for h5py.  On UNIX platforms (:ref:`Linux and OS-X <linux>`), you
must provide HDF5 yourself.  The following HDF5 versions are supported:

* 1.6.5, 1.6.6, 1.6.7, 1.6.8, 1.8.0, 1.8.1, 1.8.2

**The best solution is to install HDF5 via a package manager.** If you must
install yourself from source, keep in mind that you *must* build as a dynamic
library.

`The HDF Group`__ provides several "dumb" (untar in "/") binary distributions
for Linux, but traditionally only static libraries for Mac.  Mac OS-X users
should use something like Fink, or compile HDF5 from source.

__ http://www.hdfgroup.com/HDF5


.. _windows:

Installing on Windows
=====================

Download the executable installer from `Google Code`__ and run it.  This
installs h5py and a private copy of HDF5 1.8.

__ http://h5py.googlecode.com

Requires
--------

- NumPy_ 1.0.3 or higher

.. _linux:

Installing on Linux/Mac OS-X
============================

This package is designed to be installed from source.  You will need
Python and a C compiler, for distutils to build the extensions.  Cython_ is
required only if you want to change compile-time options, like the
debugging level.


Requires
--------
- Python with headers for development
- Numpy_ 1.0.3 or higher
- HDF5_ 1.6.5 or higher, including 1.8.X versions
- Cython_ 0.9.8.1.1 or higher

- Unix-like environment (created/tested on 32-bit Intel linux)
- A working compiler for distutils

.. _Numpy: http://numpy.scipy.org/
.. _HDF5: http://www.hdfgroup.com/HDF5
.. _Cython: http://cython.org/

Procedure
---------
1.  Unpack the tarball and cd to the resulting directory
2.  Run ``python setup.py build`` to build the package
3.  Run ``python setup.py test`` to run unit tests in-place (optional)
4.  Run ``sudo python setup.py install`` to install into your main Python
    package directory.
5.  ``cd`` out of the installation directory before importing h5py, to prevent
    Python from trying to run from the source folder.

Additional options
------------------

::

 --hdf5=<path>   Path to your HDF5 installation (if not in one of the standard
                 places.  Must contain bin/ and lib/ directories.

 --cython        Force Cython to run

 --cython-only   Run Cython, and stop before compiling with GCC.
 
 --api=<16|18>   Force either 1.6 or 1.8 API compatibility level.  Use if h5py
                 does not correctly identify your installed HDF5 version.

 --diag=<int>    Compile in diagnostic (debug) mode.













