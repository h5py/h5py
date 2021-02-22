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
To install h5py from source, you need:

* A supported Python version with development headers
* HDF5 1.8.4 or newer with development headers
* A C compiler

On Unix platforms, you also need ``pkg-config`` unless you explicitly specify
a path for HDF5 as described in :ref:`custom_install`.

There are notes below on installing HDF5, Python and a C compiler on different
platforms.

Building h5py also requires several Python packages, but in most cases pip will
automatically install these in a build environment for you, so you don't need to
deal with them manually. See :ref:`dev_install` for a list.

The actual installation of h5py should be done via::

    $ pip install --no-binary=h5py h5py

or, from a tarball or git :ref:`checkout <git_checkout>`::

    $ pip install -v .

.. _dev_install:

Development installation
........................

When modifying h5py, you often want to reinstall it quickly to test your changes.
To benefit from caching and use NumPy & Cython from your existing Python
environment, run::

    $ H5PY_SETUP_REQUIRES=0 python3 setup.py build
    $ python3 -m pip install . --no-build-isolation

For convenience, these commands are also in a script ``dev-install.sh`` in the
h5py git repository.

This skips setting up a build environment, so you should
have already installed Cython, NumPy, pkgconfig (a Python interface to
``pkg-config``) and mpi4py (if you want MPI integration - see :ref:`build_mpi`).
See ``setup.py`` for minimum versions.

This will normally rebuild Cython files automatically when they change, but
sometimes it may be necessary to force a full rebuild. The easiest way to
achieve this is to discard everything but the code committed to git. In the root
of your git checkout, run::

    $ git clean -xfd

Then build h5py again as above.

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
default build tools, e.g. ``build-essential``, which should be sufficient for our
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

You can specify build options for h5py as environment variables when you build
it from source::

    $ HDF5_DIR=/path/to/hdf5 pip install --no-binary=h5py h5py
    $ HDF5_VERSION=X.Y.Z pip install --no-binary=h5py h5py
    $ CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/path/to/parallel-hdf5 pip install --no-binary=h5py h5py

The supported build options are:

- To specify where to find HDF5, use one of these options:

  - ``HDF5_LIBDIR`` and ``HDF5_INCLUDEDIR``: the directory containing the
    compiled HDF5 libraries and the directory containing the C header files,
    respectively.
  - ``HDF5_DIR``: a shortcut for common installations, a directory with ``lib``
    and ``include`` subdirectories containing compiled libraries and C headers.
  - ``HDF5_PKGCONFIG_NAME``: A name to query ``pkg-config`` for.
    If none of these options are specified, h5py will query ``pkg-config`` by
    default for ``hdf5``, or ``hdf5-openmpi`` if building with MPI support.

- ``HDF5_MPI=ON`` to build with MPI integration - see :ref:`build_mpi`.
- ``HDF5_VERSION`` to force a specified HDF5 version. In most cases, you don't
  need to set this; the version number will be detected from the HDF5 library.
- ``H5PY_SYSTEM_LZF=1`` to build the bundled LZF compression filter
  (see :ref:`dataset_compression`) against an external LZF library, rather than
  using the bundled LZF C code.

.. _build_mpi:

Building against Parallel HDF5
------------------------------

If you just want to build with ``mpicc``, and don't care about using Parallel
HDF5 features in h5py itself::

    $ export CC=mpicc
    $ pip install --no-binary=h5py h5py

If you want access to the full Parallel HDF5 feature set in h5py
(:ref:`parallel`), you will further have to build in MPI mode. This can be done
by setting the ``HDF5_MPI`` environment variable::

    $ export CC=mpicc
    $ export HDF5_MPI="ON"
    $ pip install --no-binary=h5py h5py

You will need a shared-library build of Parallel HDF5 as well, i.e. built with
``./configure --enable-shared --enable-parallel``.
