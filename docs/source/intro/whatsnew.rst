What's new in h5py 2.0
======================

HDF5 for Python (h5py) 2.0 represents the first major refactoring of the h5py
codebase since the project's launch in 2008.  Many of the most important
changes are behind the scenes, and include changes to the way h5py interacts
with the HDF5 library and Python.  These changes have substantially
improved h5py's stability, and make it possible to use more modern versions
of HDF5 without compatibility concerns.  It is now also possible to use
h5py with Python 3.

Enhancements unlikely to affect compatibility
---------------------------------------------

* HDF5 1.8.3 through 1.8.7 now work correctly and are officially supported.

* Python 3.2 is officially supported by h5py!  Thanks especially to
  Darren Dale for getting this working.

* Fill values can now be specified when creating a dataset.  The fill time is
  H5D_FILL_TIME_IFSET for contiguous datasets, and H5D_FILL_TIME_ALLOC for
  chunked datasets.

* On Python 3, dictionary-style methods like Group.keys() and Group.values()
  return view-like objects instead of lists.

* Object and region references now work correctly in compound types.

* Zero-length dimensions for extendible axes are now allowed.

* H5py no longer attempts to auto-import ipython on startup.

* File format bounds can now be given when opening a high-level File object
  (keyword "libver").


Changes which may break existing code
-------------------------------------

Supported HDF5/Python versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* HDF5 1.6.X is no longer supported on any platform; following the release of
  1.6.10 some time ago, this branch is no longer maintained by The HDF Group.

* Python 2.6 or later is now required to run h5py.  This is a consequence of
  the numerous changes made to h5py for Python 3 compatibility.

* On Python 2.6, unittest2 is now required to run the test suite.

Group, Dataset and Datatype constructors have changed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under normal use, you should always use the create_group and create_dataset
functions to make new groups and datasets.  In h5py 2.0, the constructors have
been changed to accept a single low-level identifier.  This lets you "bind"
a new high-level Group or Dataset object to an existing low-level identifier
Creating objects in this fashion is now an official part of the API.

The File constructor remains unchanged and is still the correct mechanism for
opening and creating files.

Code which manually creates Group, Dataset or Datatype objects will have to
be modified to use create_group or create_dataset.  File-resident datatypes
can be created by assigning a NumPy dtype to a name
(e.g. mygroup["name"] = numpy.dtype('S10')).

Unicode is now used for object names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Older versions of h5py used byte strings to represent names in the file.
Starting with version 2.0, you may use either byte or unicode strings to create
objects, but object names (obj.name, etc) will always be returned as Unicode.

Code which may be affected:

* Anything which uses "isinstance" or explicit type checks on names, expecting
  "str" objects.  Such checks should be removed, or changed to compare to
  "basestring" instead.

* In Python 2.X, handling non-ascii names (e.g. writing to binary files) may
  suddenly break as the "implicit" encoding from unicode to bytes is via the
  ascii codec.  To fix this, you will need to explicitly encode any unicode
  strings which can't be represented as ascii.

Changes to scalar slicing code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a scalar dataset was accessed with the syntax ``dataset[()]``, h5py
incorrectly returned an ndarray.  H5py now correctly returns an array
scalar.  Using ``dataset[...]`` on a scalar dataset still returns an ndarray.

Array scalars now always returned when indexing a dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using datasets of compound type, retrieving a single element incorrectly
returned a tuple of values, rather than an instance of ``numpy.void_`` with the
proper fields populated.  Among other things, this meant you couldn't do
things like ``dataset[index][field]``.  H5py now always returns an array scalar,
except in the case of object dtypes (references, vlen strings).

Reading object-like data strips special type information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the past, reading multiple data points from dataset with vlen or reference
type returned an Numpy array with a "special dtype" (such as those created
by ``h5py.special_dtype()``).  In h5py 2.0, all such arrays now have a generic
Numpy object dtype (``numpy.dtype('O')``).  To get a copy of the dataset's
dtype, always use the dataset's dtype property directly (``mydataset.dtype``).

The selections module has been removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only numpy-style slicing arguments remain supported in the high level interface.
Existing code which uses the selections module should be refactored to use
numpy slicing (and ``numpy.s_`` as appropriate), or the standard C-style HDF5
dataspace machinery.

The H5Error exception class has been removed (along with h5py.h5e)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All h5py exceptions are now native Python exceptions, no longer inheriting
from H5Error.  RuntimeError is raised if h5py can't figure out what exception
is appropriate... every instance of this behavior is considered a bug.  If you
see h5py raising RuntimeError please report it so we can add the correct
mapping!

The old errors module (h5py.h5e) has also been removed.  There is no public
error-management API.

Long-deprecated dict methods have been removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aliases for Group/AttributeManager methods (e.g. listnames) have been removed.
Please use the standard Python dict interface (Python 2 or Python 3 as
appropriate) to interact with these objects.

Known issues
------------

* Thread support has been improved in h5py 2.0. However, we still recommend
  that for your own sanity you use locking to serialize access to files.

* There are reports of crashes related to storing object and region references.
  If this happens to you, please post on the mailing list or contact the h5py
  author directly.









