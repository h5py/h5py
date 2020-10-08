# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    High-level interface for creating HDF5 virtual datasets
"""

from copy import deepcopy as copy
from collections import namedtuple
from .selections import SimpleSelection, select
from .. import h5s, h5
from .. import version


class VDSmap(namedtuple('VDSmap', ('vspace', 'file_name',
                                   'dset_name', 'src_space'))):
    '''Defines a region in a virtual dataset mapping to part of a source dataset
    '''



vds_support = False
hdf5_version = version.hdf5_version_tuple[0:3]

if hdf5_version >= h5.get_config().vds_min_hdf5_version:
    vds_support = True


def _convert_space_for_key(space, key):
    """
    Converts the space with the given key. Mainly used to allow unlimited
    dimensions in virtual space selection.
    """
    key = key if isinstance(key, tuple) else (key,)
    type_code = space.get_select_type()

    # check for unlimited selections in case where selection is regular
    # hyperslab, which is the only allowed case for h5s.UNLIMITED to be
    # in the selection
    if type_code == h5s.SEL_HYPERSLABS and space.is_regular_hyperslab():
        rank = space.get_simple_extent_ndims()
        nargs = len(key)

        idx_offset = 0
        start, stride, count, block = space.get_regular_hyperslab()
        # iterate through keys. we ignore numeral indices. if we get a
        # slice, we check for an h5s.UNLIMITED value as the stop
        # if we get an ellipsis, we offset index by (rank - nargs)
        for i, sl in enumerate(key):
            if isinstance(sl, slice):
                if sl.stop == h5s.UNLIMITED:
                    counts = list(count)
                    idx = i + idx_offset
                    counts[idx] = h5s.UNLIMITED
                    count = tuple(counts)
            elif sl is Ellipsis:
                idx_offset = rank - nargs

        space.select_hyperslab(start, count, stride, block)


class VirtualSource(object):
    """Source definition for virtual data sets.

    Instantiate this class to represent an entire source dataset, and then
    slice it to indicate which regions should be used in the virtual dataset.

    path_or_dataset
        The path to a file, or an h5py dataset. If a dataset is given,
        no other parameters are allowed, as the relevant values are taken from
        the dataset instead.
    name
        The name of the source dataset within the file.
    shape
        A tuple giving the shape of the dataset.
    dtype
        Numpy dtype or string.
    maxshape
        The source dataset is resizable up to this shape. Use None for
        axes you want to be unlimited.
    """
    def __init__(self, path_or_dataset, name=None,
                 shape=None, dtype=None, maxshape=None):
        from .dataset import Dataset
        if isinstance(path_or_dataset, Dataset):
            failed = {k: v
                      for k, v in
                      {'name': name, 'shape': shape,
                       'dtype': dtype, 'maxshape': maxshape}.items()
                      if v is not None}
            if failed:
                raise TypeError("If a Dataset is passed as the first argument "
                                "then no other arguments may be passed.  You "
                                "passed {failed}".format(failed=failed))
            ds = path_or_dataset
            path = ds.file.filename
            name = ds.name
            shape = ds.shape
            dtype = ds.dtype
            maxshape = ds.maxshape
        else:
            path = path_or_dataset
            if name is None:
                raise TypeError("The name parameter is required when "
                                "specifying a source by path")
            if shape is None:
                raise TypeError("The shape parameter is required when "
                                "specifying a source by path")
            elif isinstance(shape, int):
                shape = (shape,)

            if isinstance(maxshape, int):
                maxshape = (maxshape,)

        self.path = path
        self.name = name
        self.dtype = dtype

        if maxshape is None:
            self.maxshape = shape
        else:
            self.maxshape = tuple([h5s.UNLIMITED if ix is None else ix
                                   for ix in maxshape])
        self.sel = SimpleSelection(shape)

    @property
    def shape(self):
        return self.sel.array_shape

    def __getitem__(self, key):
        tmp = copy(self)
        tmp.sel = select(self.shape, key, dataset=None)
        _convert_space_for_key(tmp.sel.id, key)
        return tmp

class VirtualLayout(object):
    """Object for building a virtual dataset.

    Instantiate this class to define a virtual dataset, assign to slices of it
    (using VirtualSource objects), and then pass it to
    group.create_virtual_dataset() to add the virtual dataset to a file.

    This class does not allow access to the data; the virtual dataset must
    be created in a file before it can be used.

    shape
        A tuple giving the shape of the dataset.
    dtype
        Numpy dtype or string.
    maxshape
        The virtual dataset is resizable up to this shape. Use None for
        axes you want to be unlimited.
    """
    def __init__(self, shape, dtype, maxshape=None):
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.dtype = dtype
        self.maxshape = (maxshape,) if isinstance(maxshape, int) else maxshape
        self.sources = []

    def __setitem__(self, key, source):
        sel = select(self.shape, key, dataset=None)
        _convert_space_for_key(sel.id, key)
        self.sources.append(VDSmap(sel.id,
                                   source.path,
                                   source.name,
                                   source.sel.id))
