README for the "h5py" Python/HDF5 interface
===========================================
Copyright (c) 2008 Andrew Collette

Version 1.1.0

* http://h5py.alfven.org        Main site, docs, quick-start guide
* http://h5py.googlecode.com    Downloads, FAQ and bug tracker

* mail: "h5py" at the domain "alfven dot org"

The h5py package provides both a high- and low-level interface to the 
HDF5 library from Python.  The low-level interface is
intended to be a complete wrapping of the HDF5 1.6 API, while the high-
level component supports Python-style object-oriented access to HDF5 files, 
datasets and groups.

The goal of this package is not to provide yet another scientific data
model. It is an attempt to create as straightforward a binding as possible
to the existing HDF5 API and abstractions, so that Python programs can
easily deal with HDF5 files and exchange data with other HDF5-aware
applications.

Installation
------------

See the file INSTALL.txt.

Documentation
-------------
Extensive documentation is available through docstrings, as well as in 
HTML format on the web and in the "docs/" directory in the 
distribution.  This document is an overview of some of the package's 
features and highlights.

Features
========

High-Level
----------

- Numpy-style access to HDF5 datasets, with automatic conversion between
  datatypes.  Slice into an HDF5 dataset and get a Numpy array back.
  Create and use datasets with chunking, compression, or other filters
  transparently.

- Dictionary-style access to HDF5 groups and attributes, including 
  iteration.

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


