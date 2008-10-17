

Python bindings for the HDF5 library
====================================

The `HDF5 library`__ is a versatile, mature library designed for the storage
of numerical data.  The h5py package provides a simple, Pythonic interface to
HDF5.  A straightforward high-level interface allows the manipulation of
HDF5 files, groups and datasets using established Python and NumPy metaphors.

__ http://www.hdfgroup.com/HDF5

Additionally, the library offers a low-level interface which exposes the 
majority of the HDF5 C API in a Pythonic, object-oriented fashion.  This
interface is documented `separately`__.

__ http://h5py.alfven.org/docs

The major design goals for h5py are simplicity and interoperability.  You don't
need to learn any new concepts beyond the basic principles of HDF5 and NumPy
arrays, and the files you create with h5py can be opened with any other
HDF5-aware program.

* `Downloads for Unix/Windows, and bug tracker (Google Code)`__
* `Alternate Windows installers`__
* `Low-level API documentation`__

* Mail: "h5py" at the domain "alfven" dot "org"

__ http://h5py.googlecode.com
__ http://h5py.alfven.org/windows/
__ http://h5py.alfven.org/docs/

Contents:

.. toctree::
    :maxdepth: 2

    build
    quick
    datasets
    threads
    licenses



