HDF5 for Python (h5py) 1.3.0 BETA
=================================

I'm pleased to announce that HDF5 for Python 1.3 is now available!  This
is a significant release introducing a number of new features, including
support for soft/external links as well as object and region references.

I encourage all interested HDF5/NumPy/Python users to give the beta a try
and to do your best to break it. :)  Download, documentation and contact
links are below.


What is h5py?
-------------

HDF5 for Python (h5py) is a general-purpose Python interface to the
Hierarchical Data Format library, version 5.  HDF5 is a mature scientific
software library originally developed at NCSA, designed for the fast,
flexible storage of enormous amounts of data.

From a Python programmer's perspective, HDF5 provides a robust way to
store data, organized by name in a tree-like fashion.  You can create
datasets (arrays on disk) hundreds of gigabytes in size, and perform
random-access I/O on desired sections.  Datasets are organized in a
filesystem-like hierarchy using containers called "groups", and 
accesed using the tradional POSIX /path/to/resource syntax.

In addition to providing interoperability with existing HDF5 datasets
and platforms, h5py is a convienient way to store and retrieve
arbitrary NumPy data and metadata.

HDF5 datasets and groups are presented as "array-like" and "dictionary-like"
objects in order to make best use of existing experience.  For example,
dataset I/O is done with NumPy-style slicing, and group access is via
indexing with string keys.  Standard Python exceptions (KeyError, etc) are
raised in response to underlying HDF5 errors.


New features in 1.3
-------------------

 - Full support for soft and external links

 - Full support for object and region references, in all contexts (datasets,
   attributes, etc).  Region references can be created using the standard
   NumPy slicing syntax.

 - A new get() method for HDF5 groups, which also allows the type of an
   object or link to be queried without first opening it.

 - Improved locking system which makes h5py faster in both multi-threaded and
   single-threaded applications.

 - Automatic creation of missing intermediate groups (HDF5 1.8)

 - Anonymous group and dataset creation (HDF5 1.8)

 - Option to enable cProfile support for the parts of h5py written in Cython

 - Many bug fixes and performance enhancements


Other changes
-------------

 - Old-style dictionary methods (listobjects, etc) will now issue
   DeprecationWarning, and will be removed in 1.4.

 - Dataset .value attribute is deprecated.  Use dataset[...] or dataset[()].

 - new_vlen(), get_vlen(), new_enum() and get_enum() are deprecated in favor
   of the functions h5py.special_dtype() and h5py.check_dtype(), which also
   support reference types.


Where to get it
---------------

* Main website, documentation:  http://h5py.alfven.org

* Downloads, bug tracker:       http://h5py.googlecode.com

* Mailing list (discussion and development): h5py at googlegroups.com

* Contact email: h5py at alfven.org


Requires
--------

* Linux, Mac OS-X or Windows

* Python 2.5 or 2.6

* NumPy 1.0.3 or later

* HDF5 1.6.5 or later (including 1.8); HDF5 is included with
  the Windows version.


