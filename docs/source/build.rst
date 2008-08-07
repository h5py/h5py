******************
Installation guide
******************

Installing on Linux/Mac OS-X
============================

This package is designed to be installed from source.  You will need
Python and a C compiler, for distutils to build the extensions.  Pyrex_ is
required only if you want to change compile-time options, like the
debugging level.

It's strongly recommended you use the versions of these packages provided
by your operating system's package manager/finder.  In particular, HDF5 can
be painful to install manually.

Requires
--------
- Unix-like environment (created/tested on 32-bit Intel linux)
- Python with headers for development
- Numpy_ 1.0.3 or higher
- HDF5_ 1.6.5 or higher, including 1.8.X versions
- A working compiler for distutils
- (Optionally) Pyrex_ 0.9.8.4 or higher

.. _Numpy: http://numpy.scipy.org/
.. _HDF5: http://www.hdfgroup.com/HDF5/
.. _Pyrex: http://www.cosc.canterbury.ac.nz/greg.ewing/python/Pyrex/

Procedure
---------
1.  Unpack the tarball and cd to the resulting directory
2.  Run ``python setup.py build`` to build the package
3.  Run ``python setup.py test`` to run unit tests in-place (optional)
4.  Run ``sudo python setup.py install`` to install into your main Python
    package directory.

Additional options
------------------

::

 --pyrex         Have Pyrex recompile changed pyx files.
 --pyrex-only    Have Pyrex recompile changed pyx files, and stop.
 --pyrex-force   Recompile all pyx files, regardless of timestamps.
 --no-pyrex      Don't run Pyrex, no matter what

 --hdf5=path     Use alternate HDF5 directory (containing bin, include, lib)

 --api=<n>       Specifies API version.  The 1.8.X API (--api=18) is a
                 work in progress.

 --debug=<n>     If nonzero, compile in debug mode.  The number is
                 interpreted as a logging-module level number.  Requires
                 Pyrex for recompilation.

 --io-nonblock   Enable experimental non-blocking I/O feature.  The GIL will
                 be released around lengthy HDF5 reads/writes.  See the
                 "Threading" manual entry for caveats.


Installing on Windows
=====================

**It's strongly recommended that you use the pre-built .exe installer.**  It
will install h5py, a private copy of HDF5 1.8.1 with ZLIB and SZIP compression
enabled, and the proper C runtime dependencies.  You must have the following
already installed:

- Python 2.5
- Numpy_ 1.0.3 or higher

If for some reason you want to build from source (for example, to change the
compile-time options), read the following instructions carefully.

Environment for source build
----------------------------

1. Install Python 2.5 and Numpy 1.0.3 or higher
2. Install MinGW to ``C:\MinGW``
3. Add ``C:\MinGW\bin`` to the Windows PATH
4. Download the ``pexports`` utility and place it in ``C:\MinGW\bin``
5. Go to ``C:\Python25\Lib\distutils`` (or equivalent path for your Python install)
   and create the file "distutils.cfg" with the following text::

    [build]
    compiler=mingw32

6. Add ``C:\Python25`` (or wherever python.exe lives) to your Windows PATH

Get HDF5 and create import file
-------------------------------

1. Download the pre-built version of HDF5 1.8.1, along with SZIP and ZLIB.
   You should use the versions built with Visual Studio 2005.  Make sure you
   get the version of SZIP with compression enabled.
2. Unpack the HDF5 archive to e.g. ``C:\hdf5``; this directory should now
   contain ``include`` and ``dll``, among other things.
3. Open a command prompt in ``C:\hdf5\dll`` and run
   ``pexports hdf5dll.dll > hdf5dll.def``
4. Create the directory ``C:\hdf5\dll2`` and move ``hdf5dll.def`` there

Compile h5py
------------

1. Download h5py and extract it to ``C:\h5py``.
2. In ``C:\h5py``, run ``python setup.py build --hdf5=C:\hdf5``
3. Copy the following files to ``C:\h5py\h5py``:

    * hdf5dll.dll (from ``C:\hdf5\dll``)
    * zlib1.dll (from the HDF group zlib archive)
    * szipdll.dll (from the HDF group szip archive)

4. Run unit tests via ``python setup.py test --hdf5=C:\hdf5``

.. note::

    If you get the message "DLL import failed: configuration incorrect" or
    some variant, you need to install the package
    "Microsoft Visual C++ 2005 SP1 Redistributable" from Microsoft's
    web site.  This is needed by the pre-built HDF5 library for some
    reason.

5. Install via ``python setup.py install --hdf5=C:\hdf5``.

After you're done, you can delete the ``C:\hdf5`` directory.  It isn't needed at
runtime.











