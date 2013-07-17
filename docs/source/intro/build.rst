.. _build:

Building h5py
=============

Downloads for all platforms are available at http://h5py.org.

Tar files are available for UNIX-like systems (Linux and Mac OS-X), and
a binary installer for Windows which includes HDF5 1.8.  

You can also
install h5py via ``pip`` or ``easy_install``.


Installing on Windows
---------------------

You will need:

  * Python 2.6 - 3.3

Download the MSI installer from http://www.h5py.org and run it.  HDF5 is
included.


Installing on Linux and Mac OS X
--------------------------------

You will need:

  * HDF5 1.8.4 or newer, shared library version with development headers (``libhdf5-dev`` or similar)
  * Python 2.6 - 3.3 with development headers (``python-dev`` or similar)
  * NumPy 1.5 or newer (1.6 recommended)

With ``pip`` or ``easy_install``::

    $ pip install h5py

Manually, from the h5py tarball::

    $ tar xzf h5py-X.Y.Z.tar.gz
    $ cd h5py
    $ python setup.py install

.. note::

    If you want access to features in HDF5 newer than 1.8.4, you will
    additionally need to install `Cython <http://cython.org>`_.


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

Build options can also be passed to ``setup.py`` directly.  Keep in mind these
options are "sticky".  They hang around between e.g. ``build`` and ``test``
invocations of h5py.

Specifying the path to HDF5::

    $ python setup.py install --hdf5=/path/to/hdf5
    $ python setup.py install --hdf5=default        # reset to compiler path

Manually specifying the HDF5 version (disables auto-detection)::

    $ python setup.py install --hdf5-version=1.8.11
    $ python setup.py install --hdf5-version=default   # re-enable autodetection

You can also configure h5py using environment variables.
The variable ``HDF5_DIR`` may contain the path to your
installation of HDF5.  The directory you provide should contain a subdirectory
called ``lib``::

    $ export HDF5_DIR=/path/to/hdf5
    $ pip install h5py

The variable ``HDF5_VERSION`` manually tells h5py to build against a specific
version of HDF5, and disables the version auto-detection in ``setup.py``::

    $ export HDF5_VERSION=1.8.11
    $ pip install h5py



Building against Parallel HDF5
------------------------------

If you just want to build with ``mpicc``, and don't care about using Parallel
HDF5 features in h5py itself::

    $ export CC=mpicc
    $ python setup.py install

If you want access to the full Parallel HDF5 feature set in h5py
(:ref:`parallel`), you will have to build in MPI mode.  Right now this must
be done with command-line options from the h5py tarball.  You will need:

  * Cython
  * A shared-library build of Parallel HDF5 (i.e. built with ``./configure --enable-shared --enable-parallel``).

To build in MPI mode (sticky option)::

    $ export CC=mpicc
    $ python setup.py build --mpi

Option ``--mpi=no`` will reset to the default (serial) build setting.

See also :ref:`parallel`.


Help! It didn't work!
---------------------

You may wish to check the :ref:`faq` first for common installation problems.

Then, feel free to ask the discussion group
`at Google Groups <http://groups.google.com/group/h5py‎>`_.  There's
only one discussion group for h5py, so you're likely to get help directly
from the maintainers.
