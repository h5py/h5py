.. _dimensionscales:

================
Dimension Scales
================

Datasets are multidimensional arrays. HDF5 provides support for labeling the
dimensions and associating one or "dimension scales" with each dimension. A
dimension scale is simply another HDF5 dataset. In principle, the length of the
multidimensional array along the dimension of interest should be equal to the
length of the dimension scale, but HDF5 does not enforce this property.

The HDF5 library provides the H5DS API for working with dimension scales. H5py
provides low-level bindings to this API in :mod:`h5py.h5ds`. These low-level
bindings are in turn used to provide a high-level interface through the
``Dataset.dims`` property. Suppose we have the following data file::

    f = File('foo.h5', 'w')
    f['data'] = np.ones((4, 3, 2), 'f')

HDF5 allows the dimensions of ``data`` to be labeled, for example::

    f['data'].dims[0].label = 'z'
    f['data'].dims[2].label = 'x'

Note that the first dimension, which has a length of 4, has been labeled "z",
the third dimension (in this case the fastest varying dimension), has been
labeled "x", and the second dimension was given no label at all.

We can also use HDF5 datasets as dimension scales. For example, if we have::

    f['x1'] = [1, 2]
    f['x2'] = [1, 1.1]
    f['y1'] = [0, 1, 2]
    f['z1'] = [0, 1, 4, 9]

We are going to treat the ``x1``, ``x2``, ``y1``, and ``z1`` datasets as
dimension scales::

    f['data'].dims.create_scale(f['x1'])
    f['data'].dims.create_scale(f['x2'], 'x2 name')
    f['data'].dims.create_scale(f['y1'], 'y1 name')
    f['data'].dims.create_scale(f['z1'], 'z1 name')

When you create a dimension scale, you may provide a name for that scale. In
this case, the ``x1`` scale was not given a name, but the others were. Now we
can associate these dimension scales with the primary dataset::

    f['data'].dims[0].attach_scale(f['z1'])
    f['data'].dims[1].attach_scale(f['y1'])
    f['data'].dims[2].attach_scale(f['x1'])
    f['data'].dims[2].attach_scale(f['x2'])

Note that two dimension scales were associated with the third dimension of
``data``. You can also detach a dimension scale::

    f['data'].dims[2].detach_scale(f['x2'])

but for now, lets assume that we have both ``x1`` and ``x2`` still associated
with the third dimension of ``data``. You can attach a dimension scale to any
number of HDF5 datasets, you can even attach it to multiple dimensions of a
single HDF5 dataset.

Now that the dimensions of ``data`` have been labeled, and the dimension scales
for the various axes have been specified, we have provided much more context
with which ``data`` can be interpreted. For example, if you want to know the
labels for the various dimensions of ``data``::

    >>> [dim.label for dim in f['data'].dims]
    ['z', '', 'x']

If you want the names of the dimension scales associated with the "x" axis::

    >>> f['data'].dims[2].keys()
    ['', 'x2 name']

:meth:`items` and :meth:`values` methods are also provided. The dimension
scales themselves can also be accessed with::

    f['data'].dims[2][1]

or::

    f['data'].dims[2]['x2 name']

such that::

    >>> f['data'].dims[2][1] == f['x2']
    True

though, beware that if you attempt to index the dimension scales with a string,
the first dimension scale whose name matches the string is the one that will be
returned. There is no guarantee that the name of the dimension scale is unique.
