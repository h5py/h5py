README for the "h5py" Python/HDF5 interface
===========================================
Copyright (c) 2008 Andrew Collette

* http://h5py.alfven.org
* mail: "h5py" at the domain "alfven dot org"

Version 0.2.0

Introduction
============

The h5py package provides both a high- and low-level interface to the 
`NCSA HDF5 library`__ from Python.  The low-level interface is
intended to be a complete wrapping of the HDF5 1.6 API, while the high-
level component supports Python-style object-oriented access to HDF5 files, 
datasets and groups.

__ http://hdf.ncsa.uiuc.edu

The goal of this package is not to provide yet another scientific data
model. It is an attempt to create as straightforward a binding as possible
to the existing HDF5 API and abstractions, so that Python programs can
easily deal with HDF5 files and exchange data with other HDF5-aware
applications.

Documentation
-------------
Extensive documentation is available through docstrings, as well as in 
`HTML format on the web`__ and in the "docs/" directory in the 
distribution.  This document is an overview of some of the package's 
features and highlights.

__ http://h5py.alfven.org

Features
========

High-Level
----------

- Numpy-style access to HDF5 datasets, with automatic conversion between
  datatypes.  Slice into an HDF5 dataset and get a Numpy array back.
  Create and use datasets with chunking, compression, or other filters
  transparently.

- Command-line browsing of HDF5 files, including the capability to import
  HDF5 objects into an interactive Python session.

- Dictionary-style access to HDF5 groups and attributes, including 
  iteration.

- Automatic creation of HDF5 datasets, named types and hard links, by
  dictionary-style assignment. For example, Group["Name"] = <Numpy array>
  creates a dataset.

Low-Level
---------

- Low-level wrappings for most of the HDF5 1.6 C API, divided in an
  intuitive fashion across modules like h5a, h5d, h5s, etc.

- No more micro-managing of identifiers; a minimal object layer on top of
  HDF5 integer identifiers means you don't need to remember to close
  every single identifier you create.

- Most API functions are presented as methods on object identifiers, rather
  than functions.  In addition to being more Pythonic, this makes programs
  less verbose and objects easier to inspect from the command line.

- Automatic exception handling; using the HDF5 error callback mechanism,
  Python exceptions are raised by the library itself when errors occur.
  Many new exception classes are provided, based on the HDF5 major error
  codes.

- Minimal changes to the HDF5 API:

    - A near-1:1 mapping between HDF5 functions and h5py functions/methods
    - Constants have their original names; H5P_DEFAULT becomes h5p.DEFAULT
    - Python extensions are provided only when not doing so would be
      obviously wrong; you don't need to learn a totally new API
    - The majority of the HDF5 C-API documentation is still valid for the
      h5py interface

Installation
============

The h5py package is designed to be installed from source.  You will need
Python and a C compiler, for distutils to build the extensions.  Pyrex_ is
not required unless you want to change compile-time options, like the
debugging level.

It's strongly recommended you use the versions of these packages provided
by your operating system's package manager/finder.  In particular, HDF5 can
be painful to install manually.

Requires
--------
- Unix-like environment (created/tested on 32-bit Intel linux)
- Numpy_ 1.0.3 or higher
- HDF5_ 1.6.5 or higher (1.8 support experimental)
- A working compiler for distutils
- (Optionally) Pyrex_ 0.9.8.4 or higher

.. _Numpy: http://numpy.scipy.org/
.. _HDF5: http://hdf.ncsa.uiuc.edu/
.. _Pyrex: http://www.cosc.canterbury.ac.nz/greg.ewing/python/Pyrex/

Procedure
---------
1.  Unpack the tarball and cd to the resulting directory
2.  Run ``python setup.py build`` to build the package
3.  Run ``python setup.py test`` to run unit tests (optional)
4.  Run ``sudo python setup.py install`` to install into your main Python
    package directory.

Additional options
------------------
 --pyrex         Have Pyrex recompile changed pyx files.
 --pyrex-only    Have Pyrex recompile changed pyx files, and stop.
 --pyrex-force   Recompile all pyx files, regardless of timestamps.
 --no-pyrex      Don't run Pyrex, no matter what

 --api=<n>       Specifies API version.  Only --api=16 is currently allowed.
 --debug=<n>     If nonzero, compile in debug mode.  The number is
                 interpreted as a logging-module level number.  Requires
                 Pyrex for recompilation.

Bugs
----
I expect there are still a few. :) A FAQ page will soon be created at the
project hosting wiki (http://h5py.googlecode.com); check there.  You can
open a ticket there or email me at "h5py" at the domain "alfven dot org".

