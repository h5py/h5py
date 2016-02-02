.. _install:

Installation
============

Pre-configured installation (recommended)
-----------------------------------------

It's strongly recommended that you use a Python distribution or package
manager to install h5py along with its compiled dependencies.  Here are some
which are popular in the Python community:

* `Anaconda <http://continuum.io/downloads>`_ or `Miniconda <http://conda.pydata.org/miniconda.html>`_ (Mac, Windows, Linux)
* `Enthought Canopy <https://www.enthought.com/products/canopy/>`_ (Mac, Windows, Linux)
* `PythonXY <https://code.google.com/p/pythonxy/>`_ (Windows)

::

    conda install h5py  # Anaconda/Miniconda
    enpkg h5py          # Canopy

Or, use your package manager:

* apt-get (Linux/Debian, including Ubuntu)
* yum (Linux/Red Hat, including Fedora and CentOS)
* Homebrew (OS X)


Source installation on Linux and OS X
-------------------------------------

You need, via apt-get, yum or Homebrew:

* Python 2.6, 2.7, 3.3, or 3.4 with development headers (``python-dev`` or similar)
* HDF5 1.8.4 or newer, shared library version with development headers (``libhdf5-dev`` or similar)
* NumPy 1.6.1 or later

::

    $ pip install h5py

or, from a tarball::

    $ python setup.py install


Source installation on Windows
------------------------------

Installing from source on Windows is effectively impossible because of the C
library dependencies involved.

If you don't want to use Anaconda, Canopy, or PythonXY, download
a `third-party wheel from Chris Gohlke's excellent collection <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_.


Custom installation
-------------------

You can specify build options for h5py with the ``configure`` option to
setup.py.  Options may be given together or separately::

    $ python setup.py configure --hdf5=/path/to/hdf5
    $ python setup.py configure --hdf5-version=X.Y.Z
    $ python setup.py configure --mpi
    
Note the ``--hdf5-version`` option is generally not needed, as h5py 
auto-detects the installed version of HDF5 (even for custom locations).

Once set, build options apply to all future builds in the source directory.
You can reset to the defaults with the ``--reset`` option::

    $ python setup.py configure --reset

You can also configure h5py using environment variables.  This is handy
when installing via ``pip``, as you don't have direct access to setup.py::

    $ HDF5_DIR=/path/to/hdf5 pip install h5py
    $ HDF5_VERSION=X.Y.Z pip install h5py
    
Here's a list of all the configure options currently supported:

======================= =========================== ===========================
Option                  Via setup.py                Via environment variable
======================= =========================== ===========================
Custom path to HDF5     ``--hdf5=/path/to/hdf5``    ``HDF5_DIR=/path/to/hdf5``
Force HDF5 version      ``--hdf5-version=X.Y.Z``    ``HDF5_VERSION=X.Y.Z``
Enable MPI mode         ``--mpi``                   (none)
======================= =========================== ===========================


Building against Parallel HDF5
------------------------------

If you just want to build with ``mpicc``, and don't care about using Parallel
HDF5 features in h5py itself::

    $ export CC=mpicc
    $ python setup.py install

If you want access to the full Parallel HDF5 feature set in h5py
(:ref:`parallel`), you will have to build in MPI mode.  Right now this must
be done with command-line options from the h5py tarball.

**You will need a shared-library build of Parallel HDF5 (i.e. built with
./configure --enable-shared --enable-parallel).**

To build in MPI mode, use the ``--mpi`` option to ``setup.py configure``::

    $ export CC=mpicc
    $ python setup.py configure --mpi
    $ python setup.py build

See also :ref:`parallel`.


