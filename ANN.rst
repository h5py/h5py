Announcing HDF5 for Python (h5py) 2.2.0
=======================================

We are proud to announce that HDF5 for Python 2.2.0 is now available.
Thanks to everyone who helped put together this release!

The h5py package is a Pythonic interface to the HDF5 binary data format.

It lets you store huge amounts of numerical data, and easily manipulate that
data from NumPy. For example, you can slice into multi-terabyte datasets
stored on disk, as if they were real NumPy arrays. Thousands of datasets can 
be stored in a single file, categorized and tagged however you want.

H5py uses straightforward NumPy and Python metaphors, like dictionary and
NumPy array syntax. For example, you can iterate over datasets in a file, or
check out the .shape or .dtype attributes of datasets. You don't need to know
anything special about HDF5 to get started.

Documentation and download links are available at:

    http://www.h5py.org

Parallel HDF5
=============

This version of h5py introduces support for MPI/Parallel HDF5, using the
mpi4py package.  Parallel HDF5 is the "native" method in HDF5 for sharing
files and objects across multiple processes, in contrast to the "threading"
package or "multiprocessing".

There is a guide to using Parallel HDF5 at the h5py web site:

    http://www.h5py.org/docs/topics/mpi.html

Other new features
==================

* Support for Python 3.3
* Support for 16-bit "mini" floats
* Access to the HDF5 scale-offset filter
* Field names are now allowed when writing to a dataset
* Region references now preserve the shape of their selections
* File-resident "committed" types can be linked to datasets and attributes
* Make object mtime tracking optional
* A new "move" method on Group objects
* Many new options for Group.copy
* Access to HDF5's get_vfd_handle
* Many bug fixes

Acknowledgments
===============

Special thanks to:

    *  Thomas A Caswell
    *  Konrad Hinsen
    *  Darren Dale
    *  Matt Zwier
    *  Toon Verstraelen
    *  Noel Dawe
    *  Barry Wardel
    *  Bradley M. Froehle
    *  Dan Meliza
    *  Johannes Meixner
    *  John Benediktsson
    *  Matthew Turk
    *  syhw

And everyone else who posted a bug report, contributed on the mailing list,
or otherwise helped with this release.


