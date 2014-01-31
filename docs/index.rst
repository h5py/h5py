HDF5 for Python
===============

**This page on readthedocs.org is a migration test.**
**The main documentation for h5py is at** http://www.h5py.org/docs.

**This version of the documentation is incomplete, may be incorrect, and is
subject to unannounced change or removal.  Go to** http://www.h5py.org/docs
**for the current stable documentation.**

What's h5py?
------------

The h5py package is a Pythonic interface to the HDF5 binary data format.

`HDF5 <http://hdfgroup.org>`_ is an open-source library and file format for 
storing large amounts of numerical data, originally developed at NCSA.  It is 
widely used in the scientific community for everything from NASA's Earth
Observing System to the storage of data from laboratory experiments and 
simulations.

Over the past few years, HDF5 has rapidly emerged as the de-facto standard 
technology in Python to store large numerical datasets.  The h5py package
is a Pythonic, easy-to-use but full featured interface to HDF5.

The package is designed with two major goals in mind:

* Use the native HDF5 feature set only
* Use native Python and NumPy abstractions

The files you create can be read by anyone else using HDF5-enabled
software, whether they're using Python, IDL, MATLAB or another software
package.


Getting h5py
------------

Downloads are at http://www.h5py.org.  It can be tricky to install all the
C library dependencies for h5py, so check out the :ref:`install guide <install>`
first.


Getting help
-------------

Tutorial and reference documentation is available at http://docs.h5py.org.
We also have a mailing list `at Google Groups <http://groups.google.com/d/forum/h5py>`_.

The lead author of h5py, Andrew Collette, also wrote
`an O'Reilly book <http://shop.oreilly.com/product/0636920030249.do>`_
which provides a comprehensive, example-based tour of HDF5 and h5py.


Introductory info
-----------------

.. toctree::
    :maxdepth: 2

    quick
    build


Advanced topics
---------------

.. toctree::
    :maxdepth: 2
    
    config
    special
    strings
    refs
    mpi


Meta-info about the h5py project
--------------------------------

.. toctree::
    :maxdepth: 2

    contributing
    faq
    licenses
