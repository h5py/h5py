.. _faq:

FAQ
===


What datatypes are supported?
-----------------------------

Below is a complete list of types for which h5py supports reading, writing and
creating datasets. Each type is mapped to a native NumPy type.

Fully supported types:

=========================           ============================================    ================================
Type                                Precisions                                      Notes
=========================           ============================================    ================================
Bitfield                            1, 2, 4 or 8 byte, BE/LE                        Read as unsigned integers
Integer                             1, 2, 4 or 8 byte, BE/LE, signed/unsigned
Float                               2, 4, 8, 12, 16 byte, BE/LE
Complex                             8 or 16 byte, BE/LE                             Stored as HDF5 struct
Compound                            Arbitrary names and offsets
Strings (fixed-length)              Any length
Strings (variable-length)           Any length, ASCII or Unicode
Opaque (kind 'V')                   Any length
Boolean                             NumPy 1-byte bool                               Stored as HDF5 enum
Array                               Any supported type
Enumeration                         Any NumPy integer type                          Read/write as integers
References                          Region and object
Variable length array               Any supported type                              See :ref:`Special Types <vlen>`
=========================           ============================================    ================================

Other numpy dtypes, such as datetime64 and timedelta64, can optionally be
stored in HDF5 opaque data using :func:`opaque_dtype`.
h5py will read this data back with the same dtype, but other software probably
will not understand it.

Unsupported types:

=========================           ============================================
Type                                Status
=========================           ============================================
HDF5 "time" type
NumPy "U" strings                   No HDF5 equivalent
NumPy generic "O"                   Not planned
=========================           ============================================


What compression/processing filters are supported?
--------------------------------------------------

=================================== =========================================== ============================
Filter                              Function                                    Availability
=================================== =========================================== ============================
DEFLATE/GZIP                        Standard HDF5 compression                   All platforms
SHUFFLE                             Increase compression ratio                  All platforms
FLETCHER32                          Error detection                             All platforms
Scale-offset                        Integer/float scaling and truncation        All platforms
SZIP                                Fast, patented compression for int/float    * UNIX: if supplied with HDF5.
                                                                                * Windows: read-only
`LZF <http://h5py.org/lzf>`_        Very fast compression, all types            Ships with h5py, C source
                                                                                available
=================================== =========================================== ============================


What file drivers are available?
--------------------------------

A number of different HDF5 "drivers", which provide different modes of access
to the filesystem, are accessible in h5py via the high-level interface. The
currently supported drivers are:

=================================== =========================================== ============================
Driver                              Purpose                                     Notes
=================================== =========================================== ============================
sec2                                Standard optimized driver                   Default on UNIX/Windows
stdio                               Buffered I/O using stdio.h
core                                In-memory file (optionally backed to disk)
family                              Multi-file driver
mpio                                Parallel HDF5 file access
=================================== =========================================== ============================


What's the difference between h5py and PyTables?
------------------------------------------------

The two projects have different design goals. PyTables presents a database-like
approach to data storage, providing features like indexing and fast "in-kernel"
queries on dataset contents. It also has a custom system to represent data types.

In contrast, h5py is an attempt to map the HDF5 feature set to NumPy as closely
as possible. For example, the high-level type system uses NumPy dtype objects
exclusively, and method and attribute naming follows Python and NumPy
conventions for dictionary and array access (i.e. ".dtype" and ".shape"
attributes for datasets, ``group[name]`` indexing syntax for groups, etc).

Underneath the "high-level" interface to h5py (i.e. NumPy-array-like objects;
what you'll typically be using) is a large Cython layer which calls into C.
This "low-level" interface provides access to nearly all of the HDF5 C API.
This layer is object-oriented with respect to HDF5 identifiers, supports
reference counting, automatic translation between NumPy and HDF5 type objects,
translation between the HDF5 error stack and Python exceptions, and more.

This greatly simplifies the design of the complicated high-level interface, by
relying on the "Pythonicity" of the C API wrapping.

There's also a PyTables perspective on this question at the
`PyTables FAQ <http://www.pytables.org/FAQ.html#how-does-pytables-compare-with-the-h5py-project>`_.


Does h5py support Parallel HDF5?
--------------------------------

Starting with version 2.2, h5py supports Parallel HDF5 on UNIX platforms.
``mpi4py`` is required, as well as an MPIO-enabled build of HDF5.
Check out :ref:`parallel` for details.


Variable-length (VLEN) data
---------------------------

Starting with version 2.3, all supported types can be stored in variable-length
arrays (previously only variable-length byte and unicode strings were supported)
See :ref:`Special Types <vlen>` for use details.  Please note that since strings
in HDF5 are encoded as ASCII or UTF-8, NUL bytes are not allowed in strings.


Enumerated types
----------------
HDF5 enumerated types are supported. As NumPy has no native enum type, they
are treated on the Python side as integers with a small amount of metadata
attached to the dtype.

NumPy object types
------------------
Storage of generic objects (NumPy dtype "O") is not implemented and not
planned to be implemented, as the design goal for h5py is to expose the HDF5
feature set, not add to it.  However, objects picked to the "plain-text" protocol
(protocol 0) can be stored in HDF5 as strings.

Appending data to a dataset
---------------------------

The short response is that h5py is NumPy-like, not database-like. Unlike the
HDF5 packet-table interface (and PyTables), there is no concept of appending
rows. Rather, you can expand the shape of the dataset to fit your needs. For
example, if I have a series of time traces 1024 points long, I can create an
extendable dataset to store them:

    >>> dset = myfile.create_dataset("MyDataset", (10, 1024), maxshape=(None, 1024))
    >>> dset.shape
    (10,1024)

The keyword argument "maxshape" tells HDF5 that the first dimension of the
dataset can be expanded to any size, while the second dimension is limited to a
maximum size of 1024. We create the dataset with room for an initial ensemble
of 10 time traces. If we later want to store 10 more time traces, the dataset
can be expanded along the first axis:

    >>> dset.resize(20, axis=0)   # or dset.resize((20,1024))
    >>> dset.shape
    (20, 1024)

Each axis can be resized up to the maximum values in "maxshape". Things to note:

* Unlike NumPy arrays, when you resize a dataset the indices of existing data
  do not change; each axis grows or shrinks independently
* The dataset rank (number of dimensions) is fixed when it is created

Unicode
-------
As of h5py 2.0.0, Unicode is supported for file names as well as for objects
in the file. When object names are read, they are returned as Unicode by default.

However, HDF5 has no predefined datatype to represent fixed-width UTF-16 or
UTF-32 (NumPy format) strings. Therefore, the NumPy 'U' datatype is not supported.

Exceptions
----------

h5py tries to map the error codes from hdf5 to the corresponding
``Exception`` class on the Python side.  However the HDF5 group does
not consider the error codes to be public API so we can not guarantee
type stability of the exceptions raised.

Development
-----------

Building from Git
~~~~~~~~~~~~~~~~~

We moved to GitHub in December of 2012 (http://github.com/h5py/h5py).

We use the following conventions for branches and tags:

* master: integration branch for the next minor (or major) version
* 2.0, 2.1, 2.2, etc: bugfix branches for released versions
* tags 2.0.0, 2.0.1, etc: Released bugfix versions

To build from a Git checkout:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the project::

    $ git clone https://github.com/h5py/h5py.git
    $ cd h5py

(Optional) Choose which branch to build from (e.g. a stable branch)::

    $ git checkout 2.1

Build the project. If given, /path/to/hdf5 should point to a directory
containing a compiled, shared-library build of HDF5 (containing things like "include" and "lib")::

    $ python setup.py build [--hdf5=/path/to/hdf5]

(Optional) Run the unit tests::

    $ python setup.py test

Report any failing tests to the mailing list (h5py at googlegroups), or by filing a bug report at GitHub.
