.. _refs:

Object and Region References
============================

In addition to soft and external links, HDF5 supplies one more mechanism to
refer to objects and data in a file.  HDF5 *references* are low-level pointers
to other objects.  The great advantage of references is that they can be
stored and retrieved as data; you can create an attribute or an entire dataset
of reference type.

References come in two flavors, object references and region references.
As the name suggests, object references point to a particular object in a file,
either a dataset, group or named datatype.  Region references always point to
a dataset, and additionally contain information about a certain selection
(*dataset region*) on that dataset.  For example, if you have a dataset
representing an image, you could specify a region of interest, and store it
as an attribute on the dataset.

Using object references
-----------------------

It's trivial to create a new object reference; every high-level object
in h5py has a read-only property "ref", which when accessed returns a new
object reference:

    >>> myfile = h5py.File('myfile.hdf5')
    >>> mygroup = myfile['/some/group']
    >>> ref = mygroup.ref
    >>> print ref
    <HDF5 object reference>

"Dereferencing" these objects is straightforward; use the same syntax as when
opening any other object:

    >>> mygroup2 = myfile[ref]
    >>> print mygroup2
    <HDF5 group "/some/group" (0 members)>

Using region references
-----------------------

Region references always contain a selection.  You create them using the 
dataset property "regionref" and standard NumPy slicing syntax:

    >>> myds = myfile.create_dataset('dset', (200,200))
    >>> regref = myds.regionref[0:10, 0:5]
    >>> print regref
    <HDF5 region reference>

The reference itself can now be used in place of slicing arguments to the
dataset:

    >>> subset = myds[regref]

There is one complication; since HDF5 region references don't express shapes
the same way as NumPy does, the data returned will be "flattened" into a
1-D array:

    >>> subset.shape
    (50,)

This is similar to the behavior of NumPy's fancy indexing, which returns
a 1D array for selections which don't conform to a regular grid.

In addition to storing a selection, region references inherit from object
references, and can be used anywhere an object reference is accepted.  In this
case the object they point to is the dataset used to create them.

Storing references in a dataset
-------------------------------

HDF5 treats object and region references as data.  Consequently, there is a
special HDF5 type to represent them.  However, NumPy has no equivalent type.
Rather than implement a special "reference type" for NumPy, references are
handled at the Python layer as plain, ordinary python objects.  To NumPy they
are represented with the "object" dtype (kind 'O').  A small amount of
metadata attached to the dtype tells h5py to interpret the data as containing
reference objects.

H5py contains a convenience function to create these "hinted dtypes" for you:

    >>> ref_dtype = h5py.special_dtype(ref=h5py.Reference)
    >>> type(ref_dtype)
    <type 'numpy.dtype'>
    >>> ref_dtype.kind
    'O'

The types accepted by this "ref=" keyword argument are h5py.Reference (for
object references) and h5py.RegionReference (for region references).

To create an array of references, use this dtype as you normally would:

    >>> ref_dataset = myfile.create_dataset("MyRefs", (100,), dtype=ref_dtype)

You can read from and write to the array as normal:

    >>> ref_dataset[0] = myfile.ref
    >>> print ref_dataset[0]
    <HDF5 object reference>

Storing references in an attribute
----------------------------------

Simply assign the reference to a name; h5py will figure it out and store it
with the correct type:

    >>> myref = myfile.ref
    >>> myfile.attrs["Root group reference"] = myref

Null references
---------------

When you create a dataset of reference type, the uninitialized elements are
"null" references.  H5py uses the truth value of a reference object to
indicate whether or not it is null:

    >>> print bool(myfile.ref)
    True
    >>> nullref = ref_dataset[50]
    >>> print bool(nullref)
    False


