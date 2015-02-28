HDF5 for Python
===============

The h5py package is a Pythonic interface to the HDF5 binary data format.

`HDF5 <http://hdfgroup.org>`_ is an open-source library and file format for 
storing large amounts of numerical data, originally developed at NCSA.  It is 
widely used in the scientific community for everything from NASA's Earth
Observing System to the storage of data from laboratory experiments and 
simulations.  Over the past few years, HDF5 has rapidly emerged as the de-facto
standard  technology in Python to store large numerical datasets.

This is the reference documentation for the h5py package.  Check out
the :ref:`quick` if you're new to h5py and HDF5.

The lead author of h5py, Andrew Collette, also wrote
`an O'Reilly book <http://shop.oreilly.com/product/0636920030249.do>`_
which provides a comprehensive, example-based introduction to using Python
and HDF5 together.

Getting h5py
------------

Downloads are at http://www.h5py.org.  It can be tricky to install all the
C library dependencies for h5py, so check out the :ref:`install guide <install>`
first.


Getting help
-------------

Tutorial and reference documentation is available here at http://docs.h5py.org.
We also have a mailing list `at Google Groups <http://groups.google.com/d/forum/h5py>`_.
Anyone is welcome to post; the list is read by both users and the core developers
of h5py.


Introductory info
-----------------

.. toctree::
    :maxdepth: 2

    quick
    build


High-level API reference
------------------------

.. toctree::
    :maxdepth: 2

    high/file
    high/group
    high/dataset
    high/attr
    high/dims


Advanced topics
---------------

.. toctree::
    :maxdepth: 2
    
    config
    special
    strings
    refs
    mpi
    swmr


Low-level API reference
-----------------------

.. toctree::
    :maxdepth: 2

    low


Meta-info about the h5py project
--------------------------------

.. toctree::
    :maxdepth: 2

    whatsnew/index
    contributing
    faq
    licenses
