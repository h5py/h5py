.. _install:

Installation
============


For Python beginners
--------------------

It can be a pain to install NumPy, HDF5, h5py, Cython and other dependencies.
If you're just starting out, by far the easiest approach is to install h5py via
your package manager (``apt-get`` or similar), or by using one of the major
science-oriented Python distributions:

* `Anaconda <http://continuum.io/downloads>`_ (Linux, Mac, Windows)
* `PythonXY <https://code.google.com/p/pythonxy/>`_ (Windows)


Installing on Windows
---------------------

You will need:

  * Python 2.6, 2.7, 3.2, 3.3 or 3.4
  * NumPy 1.6.1 or newer

Download the installer from http://www.h5py.org and run it.  HDF5 is
included.


Installing on Linux and Mac OS X
--------------------------------

System dependencies
~~~~~~~~~~~~~~~~~~~

You will need:

* Python 2.6, 2.7, 3.2, 3.3, or 3.4 with development headers (``python-dev`` or similar)
* HDF5 1.8.4 or newer, shared library version with development headers (``libhdf5-dev`` or similar)

On Mac OS X, `homebrew <http://brew.sh>`_ is a reliable way of getting
Python, HDF5 and other dependencies set up.  It is also safe to use h5py
with the OS X system Python.

Install with pip
~~~~~~~~~~~~~~~~

Simply run::

    $ pip install h5py

All dependencies are installed automatically.

Via setup.py
~~~~~~~~~~~~

You will need:

* The h5py tarball from http://www.h5py.org.
* NumPy 1.6.1 or newer
* `Cython <http://cython.org>`_ 0.17 or newer

::

    $ tar xzf h5py-X.Y.Z.tar.gz
    $ cd h5py
    $ python setup.py install


Running the test suite
----------------------

With the tarball version of h5py::

    $ python setup.py build
    $ python setup.py test

After installing h5py::

    >>> import h5py
    >>> h5py.run_tests()


Custom installation
-------------------

You can specify build options for h5py with the ``configure`` option to
setup.py.  Options may be given together or separately::

    $ python setup.py configure --hdf5=/path/to/hdf5/{lib/include}
    $ python setup.py configure --hdf5-libdir=/path/to/libhdf5/lib
    $ python setup.py configure --hdf5-includedir=/path/to/hdf5/include
    $ python setup.py configure --hdf5-libname=customname-libhdf5
    $ python setup.py configure --hdf5-version=X.Y.Z
    $ python setup.py configure --mpi

Note the ``--hdf5-version`` option is generally not needed, as h5py
auto-detects the installed version of HDF5 (even for custom locations).

The custom library name may be needed when multiple versions of hdf5 are
installed under different names.

Furthermore note, that ``--hdf5-libdir`` and ``--hdf5-includedir`` take
precedence over ``--hdf5``

Once set, build options apply to all future builds in the source directory.
You can reset to the defaults with the ``--reset`` option::

    $ python setup.py configure --reset

You can also configure h5py using environment variables.  This is handy
when installing via ``pip``, as you don't have direct access to setup.py::

    $ HDF5_DIR=/path/to/hdf5 pip install h5py
    $ HDF5_LIB=/path/to/hdf5/lib pip install h5py
    $ HDF5_INCLUDE=/path/to/hdf5/include pip install h5py
    $ HDF5_LIBNAME=customname-libhdf5 pip install h5py
    $ HDF5_VERSION=X.Y.Z pip install h5py
    $ HDF5_MPI=1 pip install h5py

Here's a list of all the configure options currently supported:

===============================  ===========================================  ===================================
Option                           Via setup.py                                 Via environment variable
===============================  ===========================================  ===================================
Custom path to HDF5              ``--hdf5=/path/to/hdf5``                     ``HDF5_DIR=/path/to/hdf5``
Custom path to HDF5 lib dir      ``--hdf5-libdir=/path/to/hdf5/lib``          ``HDF5_LIB=/path/to/hdf5``
Custom path to HDF5 include dir  ``--hdf5-includedir=/path/to/hdf5/include``  ``HDF5_INCLUDE=/path/to/hdf5``
Custom name for HDF5             ``--hdf5-libname=customname-libhdf5``        ``HDF5_LIBNAME=customname-libhdf5``
Force HDF5 version               ``--hdf5-version=X.Y.Z``                     ``HDF5_VERSION=X.Y.Z``
Enable MPI mode                  ``--mpi``                                    ``HDF5_MPI=1``
===============================  ===========================================  ===================================


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


Help! It didn't work!
---------------------

You may wish to check the :ref:`faq` first for common installation problems.

Then, feel free to ask the discussion group
`at Google Groups <http://groups.google.com/group/h5py>`_. There's
only one discussion group for h5py, so you're likely to get help directly
from the maintainers.
