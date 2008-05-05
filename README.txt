README for the "h5py" Python/HDF5 interface
===========================================
Copyright (c) 2008 Andrew Collette
http://h5py.alfven.org
mail: "h5py" at the domain "alfven dot org"

Version 0.1.0

DISCLAIMER
==========

This is the first release of h5py.  Many functions are untested, and it's 
quite possible that both the high- and low-level APIs will change in the 
future.  Also, it hasn't been tested on platforms other than 32-bit x86 
Linux.  For these reasons, you should treat it as an alpha.

Contents
========

* `Introduction`_
* `Features`_
* `High-level interface`_
* `Low-level interface`_

Introduction
============

The h5py package provides both a high- and low-level interface to the NCSA
HDF5 library (hdf.ncsa.uiuc.edu) from Python.  The low-level interface is
intended to be a complete wrapping of the HDF5 1.6 API, while the high-
level component supports Python-style object-oriented access to HDF5 files, 
datasets and groups.

Requires
--------
- Unix-like environment (created/tested on 32-bit Intel linux)
- Numpy 1.0.3 or higher
- HDF5 1.6.5 or higher (1.8 is untested)
- Pyrex 0.9.6.4 or higher

Installation
------------
See the file "INSTALL.txt"

Documentation
-------------
Extensive documentation is available through docstrings, as well as in 
HTML format on the web and in the "docs/" directory in this distribution.  
This document is an overview of some of the package's features and 
highlights.

Features
========

- Low-level wrappings for most of the HDF5 1.6 C api.  You can call H5* 
  functions directly from Python.  The wrapped APIs are:

    =====   ==============  =================
    HDF5        Purpose         Wrapping
    =====   ==============  =================
    H5A     Attributes      Module h5a
    H5F     Files           Module h5f
    H5D     Datasets        Module h5d
    H5G     Groups          Module h5g
    H5T     Datatypes       Module h5t
    H5S     Dataspaces      Module h5s
    H5I     Inspection      Module h5i
    H5Z     Filters         Module h5z
    H5P     Property lists  Module h5p
    H5E     Errors          Python exceptions
    =====   ==============  =================

  See the section "Low-level interface" below for a better overview.

- Calls that fail will raise exceptions; no more checking return values.
  Wrapper functions have been carefully designed to provide a Pythonic
  interface to the library.  Where multiple similar HDF5 functions exist
  (i.e. link and link2) they have been merged into one function, with
  additional Python keywords.

- Many new, C-level Python functions which smooth some of the rough edges. 
  For example, you can create a dataset with associated compression and
  chunking in one function call, get an iterator over the names in a group, 
  overwrite attributes without deleting them first, etc.

- Conversion functions between HDF5 datatypes and Numpy dtypes, including
  Numpy's complex numbers.  This lets you read/write data directly from an
  HDF5 dataset to a Numpy array, with the HDF5 library performing any 
  endianness or precision conversion for you automatically.

- High-level interface allows Numpy/Python-style  access to HDF5 files and 
  datasets, with automatic conversion between datatypes.  Slice into an 
  HDF5 dataset and get a Numpy array back; no extra work required.  You can 
  also create datasets which use chunking, compression, or other filters, 
  and use them like any other dataset object.

- High-level Group interface allows dictionary-style manipulation of HDF5
  groups and links, including automatic creation of datasets and attributes
  in response to assignment.

- No additional layers of abstraction beyond the HDF5 and Numpy conventions.
  I like PyTables a lot, but I don't speak database-ese. :) There are also 
  no new datatypes; just the built-in Numpy ones.


High-level interface
====================

The goal of this component is to present access to HDF5 data in a manner
consistent with the conventions of Python and Numpy.  For example, "Group" 
objects allow dictionary-style access to their members, both through the 
familiar "Object['name']" slicing syntax and by iteration.  "Dataset" 
objects support multidimensional slicing, Numpy dtype objects, and shape 
tuples.

This interface is extensively documented via module and class docstrings.
Consult the online HTML documentation (or Python's `help` command) for a 
more comprehensive guide.

Here's a (mockup) example of some of the highlights:

1. File objects support Python-like modes:

>>> from h5py.highlevel import File, Dataset, Group
>>> file = File('test_file.hdf5','r')
>>> file
File "test_file.hdf5", root members: "group1", "dataset"

2. Group objects support things like __len__ and iteration, along with
   dictionary-style access.

>>> grp = file["group1"]
>>> len(grp)
4
>>> list(grp)
['array1', 'array2', 'array3', 'array4']

3.  It's easy to add/remove members.  Datasets can even be automatically
    created from Python objects at assignment time:

>>> del grp['array2']
>>> list(grp)
['array1', 'array3', 'array4']

>>> grp['My float array'] = [1.0, 2.0, 3.5]
>>> list(grp)
['array1', 'array3', 'array4', 'My float array']
>>> grp['My float array']
Dataset: (3L,)  dtype('<f4')

4.  Datasets support the Numpy attributes shape and dtype.  Slicing a
    dataset object returns a Numpy array.

>>> dset = file["dataset"]
>>> dset
Dataset: (3L, 10L)  dtype(<f8')
>>> dset.shape        # Numpy-style shape tuples for dimensions
(3L, 10L)
>>> dset.dtype        # Genuine Numpy dtype objects represent the datatype
'<f4'

>>> dset[2,7]         # Full multidimensional slicing allowed
2.3
>>> dset[:,7]    
[-1.2, 0.5, 2.3]
>>> dset[0,0:10:2]    # Start/stop/strides work, in any combination
[-0.7, 9.1, 10.2, 2.6, 99.4]

>>> type(dset)
<class 'h5py.highlevel.Dataset'>
>>> type(dset[:,10])      # Slicing produces Numpy ndarrays
<type 'numpy.ndarray'>

5.  Full support for HDF5 scalar and array attributes:

>>> list(dest.attrs)
['Name', 'Id', 'IntArray']
>>> dset.attrs['Name']
"My Dataset"
>>> dset.attrs['Id']
42
>>> dset.attrs['IntArray']
array([0,1,2,3,4,5])
>>> dset.attrs['Name'] = "New name"
>>> dset.attrs['Name']
"New name"

Low-Level Interface
===================

The HDF5 library is divided into a number of groups (H5A, H5F, etc) which map
more-or-less directly into Python modules of the same name.  See the module
and function docstrings (or the online HTML help) for details.

Python extensions
-----------------
Most modules have several functions which are not part of the HDF5 spec.  
These are prefixed with "py" to underscore their unofficial nature.  They 
are designed to encapsulate common operations and provide a more Python/
Numpy-style interface, even at the low level of this interface.  For 
example, the functions h5t.py_h5t_to_dtype and h5t.py_dtype_to_h5t allow
automatic conversion between Numpy dtypes and HDF5 type objects.

Constants
---------
Constants are also available at the module level.  When a constant is part 
of an C enum, the name of the enum is prepended to the constant name.  For 
example, the dataspace-related enum H5S_class_t is wrapped like this:

    =============== ==============  =============
      H5S_class_t     h5s module        Value
    =============== ==============  =============
    H5S_NO_CLASS    CLASS_NO_CLASS   INT -1
    H5S_SCALAR      CLASS_SCALAR     INT 0
    H5S_SIMPLE      CLASS_SIMPLE     INT 1
    H5S_COMPLEX     CLASS_COMPLEX    INT 2
    <no equivalent> CLASS_MAPPER     DDict(...)
    =============== ==============  =============

The additional entry CLASS_MAPPER is a dictionary subclass "DDict" which 
maps  the integer values to string descriptions.  This simplifies debugging 
and  logging.  The DDict class overrides "dict" so that it will always 
return a value; if the given integer is not in the CLASS enum, the returned 
string is ``"*INVALID* (<given value>)"``.

Exceptions
----------
Each HDF5 API function is individually wrapped; the return value
is checked, and the appropriate exception is raised (from h5py.errors) if
something has gone wrong.  This way you can write more Pythonic, exception-
based code instead of checking return values. 

Additionally, the HDF5 error stack is automatically attached to the 
exception message, giving you a clear picture of what went wrong.  The 
library will never print anything to stderr; everything goes through Python.


