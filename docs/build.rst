.. _install:

Installation
============

.. _install_recommends:

It is highly recommended that you use a pre-built version of h5py, either from a
Python Distribution, an OS-specific package manager, or a pre-built wheel from
PyPI.

Be aware however that most pre-built versions lack MPI support, and that they
are built against a specific version of HDF5. If you require MPI support, or
newer HDF5 features, you will need to build from source.

After installing h5py, you should run the tests to be sure that everything was
installed correctly. This can be done in the python interpreter via::

    import h5py
    h5py.run_tests()

On Python 2.6, unittest2 must be installed to run the tests.

.. _prebuilt_install:

Pre-built installation (recommended)
-----------------------------------------

Pre-build h5py can be installed via many Python Distributions, OS-specific
package managers, or via h5py wheels.

Python Distributions
....................
If you do not already use a Python Distribution, we recommend either
`Anaconda <http://continuum.io/downloads>`_/`Miniconda <http://conda.pydata.org/miniconda.html>`_
or
`Enthought Canopy <https://www.enthought.com/products/canopy/>`_, both of which
support most versions of Microsoft Windows, OSX/MacOS, and a variety of Linux
Distributions. Installation of h5py can be done on the command line via::

    $ conda install h5py

for Anaconda/MiniConda, and via::

    $ enpkg h5py

for Canopy.

Wheels
......
If you have an existing Python installation (e.g. a python.org download,
or one that comes with your OS), then on Windows, MacOS/OSX, and
Linux on Intel computers, pre-built h5py wheels can be installed via pip from
PyPI::

    $ pip install h5py

Additionally, for Windows users, `Chris Gohlke provides third-party wheels
which use Intel's MKL <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_.

OS-Specific Package Managers
............................
On OSX/MacOS, h5py can be installed via `Homebrew <https://brew.sh/>`_,
`Macports <https://www.macports.org/>`_, or `Fink <http://finkproject.org/>`_.

The current state of h5py in various Linux Distributions can be seen at
https://pkgs.org/download/python-h5py, and can be installed via the package
manager.

As far as the h5py developers know, none of the Windows package managers (e.g.
`Chocolatey <https://chocolatey.org/>`_, `nuget <https://www.nuget.org/>`_)
have h5py included, however they may assist in installing h5py's requirements
when building from source.


.. _source_install:

Source installation
-------------------
To install h5py from source, you need three things installed:
* A supported Python version with development headers
* HDF5 1.8.4 or newer with development headers
* A C compiler
OS-specific instructions for installing HDF5, Python and a C compiler are in the next few
sections.

Additional Python-level requirements should be installed automatically (which
will require an internet connection).

The actual installation of h5py should be done via::

    $ pip install --no-binary=h5py h5py

or, from a tarball or git :ref:`checkout <git_checkout>` ::

    $ pip install -v .

or ::

    $ python setup.py install

If you are working on a development version and the underlying cython files change
it may be necessary to force a full rebuild.  The easiest way to achieve this is ::

    $ git clean -xfd

from the top of your clone and then rebuilding.

Source installation on OSX/MacOS
................................
HDF5 and Python are most likely in your package manager (e.g. `Homebrew <https://brew.sh/>`_,
`Macports <https://www.macports.org/>`_, or `Fink <http://finkproject.org/>`_).
Be sure to install the development headers, as sometimes they are not included
in the main package.

XCode comes with a C compiler (clang), and your package manager will likely have
other C compilers for you to install.

Source installation on Linux/Other Unix
.......................................
HDF5 and Python are most likely in your package manager. A C compiler almost
definitely is, usually there is some kind of metapackage to install the
default build tools, e.g. `build-essential`, which should be sufficient for our
needs. Make sure that that you have the development headers, as they are
usually not installed by default. They can usually be found in ``python-dev`` or
similar and ``libhdf5-dev`` or similar.

Source installation on Windows
..............................
Installing from source on Windows is a much more difficult prospect than
installing from source on other OSs, as not only are you likely to need to
compile HDF5 from source, everything must be built with the correct version of
Visual Studio. Additional patches are also needed to HDF5 to get HDF5 and Python
to work together.

We recommend examining the appveyor build scripts, and using those to build and
install HDF5 and h5py.

.. _custom_install:

Custom installation
-------------------
.. important:: Remember that pip installs wheels by default.
    To perform a custom installation with pip, you should use::

        $ pip install --no-binary=h5py h5py

    or build from a git checkout or downloaded tarball to avoid getting
    a pre-built version of h5py.

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

    $ HDF5_DIR=/path/to/hdf5 pip install --no-binary=h5py h5py
    $ HDF5_VERSION=X.Y.Z pip install --no-binary=h5py h5py
    $ CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/path/to/parallel-hdf5 pip install --no-binary=h5py h5py

Here's a list of all the configure options currently supported:

======================= =========================== ===========================
Option                  Via setup.py                Via environment variable
======================= =========================== ===========================
Custom path to HDF5     ``--hdf5=/path/to/hdf5``    ``HDF5_DIR=/path/to/hdf5``
Force HDF5 version      ``--hdf5-version=X.Y.Z``    ``HDF5_VERSION=X.Y.Z``
Enable MPI mode         ``--mpi``                   ``HDF5_MPI=ON``
======================= =========================== ===========================

.. _build_mpi:

Building against Parallel HDF5
------------------------------

If you just want to build with ``mpicc``, and don't care about using Parallel
HDF5 features in h5py itself::

    $ export CC=mpicc
    $ pip install --no-binary=h5py h5py

If you want access to the full Parallel HDF5 feature set in h5py
(:ref:`parallel`), you will further have to build in MPI mode.  This can either
be done with command-line options from the h5py tarball or by::

    $ export HDF5_MPI="ON"

**You will need a shared-library build of Parallel HDF5 (i.e. built with
./configure --enable-shared --enable-parallel).**

To build in MPI mode, use the ``--mpi`` option to ``setup.py configure`` or
export ``HDF5_MPI="ON"`` beforehand::

    $ export CC=mpicc
    $ export HDF5_MPI="ON"
    $ pip install --no-binary=h5py h5py

See also :ref:`parallel`.
