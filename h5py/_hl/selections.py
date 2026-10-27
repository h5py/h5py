# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    High-level access to HDF5 dataspace selections
"""

import itertools

import numpy as np

from .base import product
from .. import h5s, h5r, _selector

def select(shape, args, dataset=None):
    """ High-level routine to generate a selection from arbitrary arguments
    to __getitem__.  The arguments should be the following:

    shape
        Shape of the "source" dataspace.

    args
        Either a single argument or a tuple of arguments.  See below for
        supported classes of argument.

    dataset
        A h5py.Dataset instance representing the source dataset.

    Argument classes:

    Single Selection instance
        Returns the argument.

    numpy.ndarray
        Must be a boolean mask.  Returns a PointSelection instance.

    RegionReference
        Returns a Selection instance.

    Indices, slices, ellipses, MultiBlockSlices only
        Returns a SimpleSelection instance

    Indices, slices, ellipses, lists or boolean index arrays
        Returns a FancySelection instance.
    """
    if not isinstance(args, tuple):
        args = (args,)

    # "Special" indexing objects
    if len(args) == 1:

        arg = args[0]
        if isinstance(arg, Selection):
            if arg.shape != shape:
                raise TypeError("Mismatched selection shape")
            return arg

        elif isinstance(arg, np.ndarray) and arg.dtype.kind == 'b':
            if arg.shape == shape:
                return PointSelection.from_mask(arg)
            # Allow 1D boolean array on the 1st dim
            elif arg.shape != shape[:1]:
                raise TypeError("Boolean indexing array has incompatible shape")


        elif isinstance(arg, h5r.RegionReference):
            if dataset is None:
                raise TypeError("Cannot apply a region reference without a dataset")
            sid = h5r.get_region(arg, dataset.id)
            if shape != sid.shape:
                raise TypeError("Reference shape does not match dataset shape")

            return Selection(shape, spaceid=sid)

    if dataset is not None:
        selector = dataset._selector
    else:
        space = h5s.create_simple(shape)
        selector = _selector.Selector(space)

    return selector.make_selection(args)


class Selection:

    """
        Base class for HDF5 dataspace selections.  Subclasses support the
        "selection protocol", which means they have at least the following
        members:

        __init__(shape)   => Create a new selection on "shape"-tuple
        __getitem__(args) => Perform a selection with the range specified.
                             What args are allowed depends on the
                             particular subclass in use.

        id (read-only) =>      h5py.h5s.SpaceID instance
        shape (read-only) =>   The shape of the dataspace.
        mshape  (read-only) => The shape of the selection region.
                               Not guaranteed to fit within "shape", although
                               the total number of points is less than
                               product(shape).
        nselect (read-only) => Number of selected points.  Always equal to
                               product(mshape).

        broadcast(target_shape) => Return an iterable which yields dataspaces
                                   for read, based on target_shape.

        The base class represents "unshaped" selections (1-D).
    """

    def __init__(self, shape, spaceid=None):
        """ Create a selection.  Shape may be None if spaceid is given. """
        if spaceid is not None:
            self._id = spaceid
            self._shape = spaceid.shape
        else:
            shape = tuple(shape)
            self._shape = shape
            self._id = h5s.create_simple(shape, (h5s.UNLIMITED,)*len(shape))
            self._id.select_all()

    @property
    def id(self):
        """ SpaceID instance """
        return self._id

    @property
    def shape(self):
        """ Shape of whole dataspace """
        return self._shape

    @property
    def nselect(self):
        """ Number of elements currently selected """
        return self._id.get_select_npoints()

    @property
    def mshape(self):
        """ Shape of selection (always 1-D for this class) """
        return (self.nselect,)

    @property
    def array_shape(self):
        """Shape of array to read/write (always 1-D for this class)"""
        return self.mshape

    # expand_shape and broadcast only really make sense for SimpleSelection
    def expand_shape(self, source_shape):
        if product(source_shape) != self.nselect:
            raise TypeError("Broadcasting is not supported for point-wise selections")
        return source_shape

    def broadcast(self, source_shape):
        """ Get an iterable for broadcasting """
        if product(source_shape) != self.nselect:
            raise TypeError("Broadcasting is not supported for point-wise selections")
        yield self._id

    def __getitem__(self, args):
        raise NotImplementedError("This class does not support indexing")

class PointSelection(Selection):

    """
        Represents a point-wise selection.  You can supply sequences of
        points to the three methods append(), prepend() and set(), or
        instantiate it with a single boolean array using from_mask().
    """
    def __init__(self, shape, spaceid=None, points=None):
        super().__init__(shape, spaceid)
        if points is not None:
            self._perform_selection(points, h5s.SELECT_SET)

    def _perform_selection(self, points, op):
        """ Internal method which actually performs the selection """
        points = np.atleast_2d(np.asarray(points, order='C', dtype='u8'))

        if self._id.get_select_type() != h5s.SEL_POINTS:
            op = h5s.SELECT_SET

        if len(points) == 0:
            self._id.select_none()
        else:
            self._id.select_elements(points, op)

    @classmethod
    def from_mask(cls, mask, spaceid=None):
        """Create a point-wise selection from a NumPy boolean array """
        if not (isinstance(mask, np.ndarray) and mask.dtype.kind == 'b'):
            raise TypeError("PointSelection.from_mask only works with bool arrays")

        points = np.transpose(mask.nonzero())
        return cls(mask.shape, spaceid, points=points)

    def append(self, points):
        """ Add the sequence of points to the end of the current selection """
        self._perform_selection(points, h5s.SELECT_APPEND)

    def prepend(self, points):
        """ Add the sequence of points to the beginning of the current selection """
        self._perform_selection(points, h5s.SELECT_PREPEND)

    def set(self, points):
        """ Replace the current selection with the given sequence of points"""
        self._perform_selection(points, h5s.SELECT_SET)


class SimpleSelection(Selection):

    """ A single "rectangular" (regular) selection composed of only slices
        and integer arguments.  Can participate in broadcasting.
    """

    @property
    def mshape(self):
        """ Shape of current selection """
        return self._sel[1]

    @property
    def array_shape(self):
        scalar = self._sel[3]
        return tuple(x for x, s in zip(self.mshape, scalar, strict=True) if not s)

    def __init__(self, shape, spaceid=None, hyperslab=None):
        super().__init__(shape, spaceid)
        if hyperslab is not None:
            self._sel = hyperslab
        else:
            # No hyperslab specified - select all
            rank = len(self.shape)
            self._sel = ((0,)*rank, self.shape, (1,)*rank, (False,)*rank)

    def expand_shape(self, source_shape):
        """Match the dimensions of an array to be broadcast to the selection

        The returned shape describes an array of the same size as the input
        shape, but its dimensions

        E.g. with a dataset shape (10, 5, 4, 2), writing like this::

            ds[..., 0] = np.ones((5, 4))

        The source shape (5, 4) will expand to (1, 5, 4, 1).
        Then the broadcast method below repeats that chunk 10
        times to write to an effective shape of (10, 5, 4, 1).
        """
        start, count, step, scalar = self._sel

        rank = len(count)
        remaining_src_dims = list(source_shape)

        eshape = []
        for idx in range(1, rank + 1):
            if len(remaining_src_dims) == 0 or scalar[-idx]:  # Skip scalar axes
                eshape.append(1)
            else:
                t = remaining_src_dims.pop()
                if t == 1 or count[-idx] == t:
                    eshape.append(t)
                else:
                    raise TypeError("Can't broadcast %s -> %s" % (source_shape, self.array_shape))  # array shape

        if any([n > 1 for n in remaining_src_dims]):
            # All dimensions from target_shape should either have been popped
            # to match the selection shape, or be 1.
            raise TypeError("Can't broadcast %s -> %s" % (source_shape, self.array_shape))  # array shape

        # We have built eshape backwards, so now reverse it
        return tuple(eshape[::-1])


    def broadcast(self, source_shape):
        """ Return an iterator over target dataspaces for broadcasting.

        Follows the standard NumPy broadcasting rules against the current
        selection shape (self.mshape).
        """
        if self.shape == ():
            if product(source_shape) != 1:
                raise TypeError("Can't broadcast %s to scalar" % source_shape)
            self._id.select_all()
            yield self._id
            return

        start, count, step, scalar = self._sel

        rank = len(count)
        tshape = self.expand_shape(source_shape)

        # Avoid ZeroDivisionError below (after the shape checks in expand_source)
        if any(d == 0 for d in count):
            return

        chunks = tuple(x//y for x, y in zip(count, tshape, strict=True))
        nchunks = product(chunks)

        if nchunks == 1:
            yield self._id
        else:
            sid = self._id.copy()
            sid.select_hyperslab((0,)*rank, tshape, step)
            for idx in range(nchunks):
                offset = tuple(x*y*z + s for x, y, z, s in zip(np.unravel_index(idx, chunks), tshape, step, start, strict=True))
                sid.offset_simple(offset)
                yield sid

    def broadcast_tiles(self, source_shape, itemsize, max_nbytes, chunks=None):
        """Plan a broadcast write through a buffer bigger than the source.

        broadcast() writes a small source, such as a scalar or one row, one
        source-sized block at a time, so filling a large selection takes one
        HDF5 call per block, which is very slow (issue #1067). This instead
        picks a larger tile shape, which the caller fills by broadcasting
        the source, and splits the selection into blocks no larger than the
        tile. Along the broadcast axes, block edges follow the dataset's
        chunk boundaries, so each chunk is covered by a single write.

        The tile is at most max_nbytes, or one chunk if that is bigger.

        Returns None if the source doesn't need repeating, or if the
        selection must be split but either isn't a plain hyperslab (e.g. it
        uses MultiBlockSlice) or no tile bigger than the source fits.
        Otherwise returns (tile_shape, blocks), where blocks yields
        (count, fspace) pairs: count is the shape of the tile's leading
        corner to write, and fspace is the matching file dataspace. The same
        fspace object is reused between iterations.
        """
        if self.shape == ():
            return None

        start, count, step, _ = self._sel
        rank = len(count)
        src_shape = self.expand_shape(source_shape)

        if any(n == 0 for n in count) or product(src_shape) == product(count):
            return None

        if chunks is None:
            # Contiguous layout: treat the whole dataset as one chunk
            chunks = self.shape
        else:
            max_nbytes = max(max_nbytes, itemsize * product(chunks))

        # Small enough to write in one go
        if itemsize * product(count) <= max_nbytes:
            return count, iter([(count, self._id)])

        # Only selections made of slices & integers can be split up using
        # start, count & step alone.
        sel_type = self._id.get_select_type()
        if sel_type == h5s.SEL_HYPERSLABS:
            if not (self._id.is_regular_hyperslab() and
                    self._id.get_regular_hyperslab() == (start, step, count, (1,) * rank)):
                return None
        elif sel_type != h5s.SEL_ALL:
            return None

        # Most selected points in one chunk, along each axis
        per_chunk = tuple(min(n, -(-c // s)) for n, c, s in zip(count, chunks, step, strict=True))
        bcast_axes = [d for d in range(rank) if src_shape[d] == 1 and count[d] > 1]

        # Start from one chunk along each broadcast axis
        tile = list(count)
        for d in bcast_axes:
            tile[d] = per_chunk[d]

        if itemsize * product(tile) <= max_nbytes:
            # Grow the tile by whole chunks, starting from the last axis
            for d in reversed(bcast_axes):
                fit = max_nbytes // (itemsize * product(tile) // tile[d])
                if fit >= count[d]:
                    tile[d] = count[d]
                else:
                    tile[d] = max(per_chunk[d], fit - fit % per_chunk[d])
                    break
        else:
            # The source is big enough that even this is too much: shrink
            # the tile, starting from the first axis
            for d in bcast_axes:
                fit = max_nbytes // (itemsize * product(tile) // tile[d])
                tile[d] = max(1, min(tile[d], fit))
                if itemsize * product(tile) <= max_nbytes:
                    break
            if product(tile) == product(src_shape):
                return None

        tile = tuple(tile)
        axis_blocks = [
            self._tile_axis(start[d], count[d], step[d], chunks[d], per_chunk[d], tile[d])
            for d in range(rank)
        ]

        def blocks():
            sid = self._id.copy()
            for parts in itertools.product(*axis_blocks):
                offset = tuple(s + i * st for s, (i, _), st in zip(start, parts, step, strict=True))
                block_count = tuple(n for _, n in parts)
                sid.select_hyperslab(offset, block_count, step)
                yield block_count, sid

        return tile, blocks()

    @staticmethod
    def _tile_axis(start, count, step, chunk, per_chunk, length):
        """Split the points selected along one axis into (first, n) blocks.

        Points are numbered 0 ... count-1 within the selection, and each
        block has at most `length` of them. Blocks are made of whole chunks
        if length >= per_chunk; otherwise each chunk is split into blocks.
        """
        if length >= count:
            return [(0, count)]

        # First selected point in each chunk the selection touches. Chunks
        # with no selected points (if step > chunk) give repeated values.
        first_chunk = start // chunk
        last_chunk = (start + (count - 1) * step) // chunk
        edges = sorted({max(0, -(-(k * chunk - start) // step))
                        for k in range(first_chunk, last_chunk + 1)})
        edges.append(count)

        if length >= per_chunk:
            group = length // per_chunk
            edges = edges[:-1:group] + [count]
            spans = zip(edges[:-1], edges[1:])
        else:
            spans = ((a, min(a + length, hi))
                     for lo, hi in zip(edges[:-1], edges[1:])
                     for a in range(lo, hi, length))

        return [(a, b - a) for a, b in spans]


class FancySelection(Selection):

    """
        Implements advanced NumPy-style selection operations in addition to
        the standard slice-and-int behavior.

        Indexing arguments may be ints, slices, lists of indices, or
        per-axis (1D) boolean arrays.

        Broadcasting is not supported for these selections.
    """

    @property
    def mshape(self):
        return self._mshape

    @property
    def array_shape(self):
        return self._array_shape

    def __init__(self, shape, spaceid=None, mshape=None, array_shape=None):
        super().__init__(shape, spaceid)
        if mshape is None:
            mshape = self.shape
        if array_shape is None:
            array_shape = mshape
        self._mshape = mshape
        self._array_shape = array_shape

    def expand_shape(self, source_shape):
        if not source_shape == self.array_shape:
            raise TypeError("Broadcasting is not supported for complex selections")
        return source_shape

    def broadcast(self, source_shape):
        if not source_shape == self.array_shape:
            raise TypeError("Broadcasting is not supported for complex selections")
        yield self._id


def guess_shape(sid):
    """ Given a dataspace, try to deduce the shape of the selection.

    Returns one of:
        * A tuple with the selection shape, same length as the dataspace
        * A 1D selection shape for point-based and multiple-hyperslab selections
        * None, for unselected scalars and for NULL dataspaces
    """

    sel_class = sid.get_simple_extent_type()    # Dataspace class
    sel_type = sid.get_select_type()            # Flavor of selection in use

    if sel_class == h5s.NULL:
        # NULL dataspaces don't support selections
        return None

    elif sel_class == h5s.SCALAR:
        # NumPy has no way of expressing empty 0-rank selections, so we use None
        if sel_type == h5s.SEL_NONE: return None
        if sel_type == h5s.SEL_ALL: return tuple()

    elif sel_class != h5s.SIMPLE:
        raise TypeError("Unrecognized dataspace class %s" % sel_class)

    # We have a "simple" (rank >= 1) dataspace

    N = sid.get_select_npoints()
    rank = len(sid.shape)

    if sel_type == h5s.SEL_NONE:
        return (0,)*rank

    elif sel_type == h5s.SEL_ALL:
        return sid.shape

    elif sel_type == h5s.SEL_POINTS:
        # Like NumPy, point-based selections yield 1D arrays regardless of
        # the dataspace rank
        return (N,)

    elif sel_type != h5s.SEL_HYPERSLABS:
        raise TypeError("Unrecognized selection method %s" % sel_type)

    # We have a hyperslab-based selection

    if N == 0:
        return (0,)*rank

    bottomcorner, topcorner = (np.array(x) for x in sid.get_select_bounds())

    # Shape of full selection box
    boxshape = topcorner - bottomcorner + np.ones((rank,))

    def get_n_axis(sid, axis):
        """ Determine the number of elements selected along a particular axis.

        To do this, we "mask off" the axis by making a hyperslab selection
        which leaves only the first point along the axis.  For a 2D dataset
        with selection box shape (X, Y), for axis 1, this would leave a
        selection of shape (X, 1).  We count the number of points N_leftover
        remaining in the selection and compute the axis selection length by
        N_axis = N/N_leftover.
        """

        if(boxshape[axis]) == 1:
            return 1

        start = bottomcorner.copy()
        start[axis] += 1
        count = boxshape.copy()
        count[axis] -= 1

        # Throw away all points along this axis
        masked_sid = sid.copy()
        masked_sid.select_hyperslab(tuple(start), tuple(count), op=h5s.SELECT_NOTB)

        N_leftover = masked_sid.get_select_npoints()

        return N//N_leftover


    shape = tuple(get_n_axis(sid, x) for x in range(rank))

    if product(shape) != N:
        # This means multiple hyperslab selections are in effect,
        # so we fall back to a 1D shape
        return (N,)

    return shape
