# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

# We use __getitem__ side effects, which pylint doesn't like.
# pylint: disable=pointless-statement

"""
    High-level access to HDF5 dataspace selections
"""
from operator import index
import numpy as np

from .base import product
from .. import h5s, h5r


def select(shape, args, dsid):
    """ High-level routine to generate a selection from arbitrary arguments
    to __getitem__.  The arguments should be the following:

    shape
        Shape of the "source" dataspace.

    args
        Either a single argument or a tuple of arguments.  See below for
        supported classes of argument.

    dsid
        A h5py.h5d.DatasetID instance representing the source dataset.

    Argument classes:

    Single Selection instance
        Returns the argument.

    numpy.ndarray
        Must be a boolean mask.  Returns a PointSelection instance.

    RegionReference
        Returns a Selection instance.

    Indices, slices, ellipses only
        Returns a SimpleSelection instance

    Indices, slices, ellipses, lists or boolean index arrays
        Returns a FancySelection instance.
    """
    if not isinstance(args, tuple):
        args = (args,)

    ellipsis_ix = nargs = len(args)

    # "Special" indexing objects
    if nargs == 1:
        arg = args[0]
        if isinstance(arg, SelectionBase):
            if arg.shape != shape:
                raise TypeError("Mismatched selection shape")
            return arg

        elif isinstance(arg, np.ndarray) and arg.dtype.kind == 'b':
            return PointSelection(shape, mask=arg)

        elif isinstance(arg, h5r.RegionReference):
            sid = h5r.get_region(arg, dsid)
            if shape != sid.shape:
                raise TypeError("Reference shape does not match dataset shape")

            return OpaqueSelection(shape, spaceid=sid)

    rank = len(shape)
    dim_ix = 0
    seq_dim = seq_arr = None
    starts = [0] * rank
    counts = [1] * rank
    steps  = [1] * rank
    scalar = [False] * rank
    seen_ellipsis = False

    for a in args:
        if a is Ellipsis:
            # [...] (Ellipsis -> fill as many other dimensions as needed)
            if seen_ellipsis:
                raise ValueError("Only one ellipsis may be used.")
            seen_ellipsis = True
            ellipsis_ix = dim_ix
            nargs -= 1
            dim_ix += rank - nargs  # Skip ahead to the remaining dimensions
            continue

        l = shape[dim_ix]

        if isinstance(a, slice):
            # [0:10]
            starts[dim_ix], counts[dim_ix], steps[dim_ix] = _translate_slice(a, l)
        else:
            try:
                a = index(a)
            except Exception:
                # [[0, 1, 2]] - list/array (potentially): fancy indexing
                arr = np.asarray(a)
                if arr.shape == (0,):
                    # Empty coordinate list -> like slice(0, 0)
                    # asarray([]) -> float, so check this before dtype
                    counts[dim_ix] = 0

                else:
                    arr = _validate_sequence_arg(arr, l)
                    if seq_dim is not None:
                        raise TypeError("Only one indexing vector or array is currently allowed for advanced selection")

                    seq_dim = dim_ix
                    seq_arr = arr

            else:
                # [0] - simple integer indices
                if a < 0:
                    a += l

                if not 0 <= a < l:
                    _out_of_range(a, l)

                starts[dim_ix] = a
                scalar[dim_ix] = True

        dim_ix += 1

    if nargs == 0:
        return AllSelection(shape)
    elif nargs < rank:
        # Fill in ellipsis or trailing dimensions
        ellipsis_end = ellipsis_ix + (rank - nargs)
        counts[ellipsis_ix:ellipsis_end] = shape[ellipsis_ix:ellipsis_end]
    elif nargs > rank:
        raise ValueError(f"{nargs} indexing arguments for {rank} dimensions")

    if seq_dim is not None:
        return FancySelection(shape, starts=tuple(starts), counts=tuple(counts),
                               steps=tuple(steps), scalar=tuple(scalar),
                              seq_dim=seq_dim, seq_arr=seq_arr)
    else:
        return SimpleSelection(shape, starts=tuple(starts), counts=tuple(counts),
                               steps=tuple(steps), scalar=tuple(scalar))

def _validate_sequence_arg(arr: np.ndarray, l: int):
    # Array must be integers...
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("Coordinate arrays must be integers")

    # ... 1D...
    if arr.ndim != 1:
        raise TypeError("Coordinate arrays for indexing must be 1D")

    # (Convert negative indices to positive)
    arr[arr < 0] += l

    # ... all values in 0 <= a < l ...
    oob = (arr < 0) | (arr > l)
    if np.any(oob):
        _out_of_range(arr[oob], l)

    # ... increasing, unique values.
    if np.any(np.diff(arr) < 1):
        raise TypeError("Indexing elements must be in increasing order & unique")

    return arr

def _out_of_range(val, l):
    if l == 0:
        msg = f"Index ({val}) out of range (empty dimension)"
    else:
        msg = f"Index ({val}) out of range (0-{l - 1})"
    raise IndexError(msg)

class SelectionBase:
    shape = ()  # Override in subclasses

    @property
    def id(self):
        id = h5s.create_simple(self.shape)
        self.apply_to_spaceid(id)
        return id

    def apply_to_spaceid(self, spaceid):
        raise NotImplementedError

    @property
    def mshape(self):
        raise NotImplementedError

    @property
    def nselect(self):
        raise NotImplementedError


class AllSelection(SelectionBase):
    def __init__(self, shape):
        self.shape = shape

    @property
    def mshape(self):
        return self.shape

    @property
    def nselect(self):
        return product(self.shape)

    def apply_to_spaceid(self, spaceid: h5s.SpaceID):
        spaceid.select_all()

    def broadcast(self, target_shape):
        """ Return an iterator over target dataspaces for broadcasting.

        Follows the standard NumPy broadcasting rules against the current
        selection shape (self.mshape).
        """
        if self.shape == ():
            if np.product(target_shape) != 1:
                raise TypeError("Can't broadcast %s to scalar" % target_shape)
            space = h5s.create(h5s.SCALAR)
            space.select_all()
            yield space
            return

        rank = len(self.shape)

        start = [0] * rank
        count = self.shape
        step = [1] * rank

        target = list(target_shape)

        tshape = []
        for idx in range(1,rank+1):
            if len(target) == 0:
                tshape.append(1)
            else:
                t = target.pop()
                if t == 1 or count[-idx] == t:
                    tshape.append(t)
                else:
                    raise TypeError("Can't broadcast %s -> %s" % (target_shape, self.mshape))

        if any([n > 1 for n in target]):
            # All dimensions from target_shape should either have been popped
            # to match the selection shape, or be 1.
            raise TypeError("Can't broadcast %s -> %s" % (target_shape, self.mshape))

        tshape.reverse()
        tshape = tuple(tshape)

        chunks = tuple(x//y for x, y in zip(count, tshape))
        nchunks = product(chunks)

        if nchunks == 1:
            yield self.id
        else:
            sid = self.id.copy()
            sid.select_hyperslab((0,)*rank, tshape)
            for idx in range(nchunks):
                offset = tuple(x*y*z + s for x, y, z, s in zip(np.unravel_index(idx, chunks), tshape, step, start))
                sid.offset_simple(offset)
                yield sid

class OpaqueSelection(SelectionBase):

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
        self.shape = shape
        self._id = spaceid

    @property
    def id(self):
        """ SpaceID instance """
        return self._id

    @property
    def nselect(self):
        """ Number of elements currently selected """
        return self._id.get_select_npoints()

    @property
    def mshape(self):
        """ Shape of selection (always 1-D for this class) """
        return (self.nselect,)

    def broadcast(self, target_shape):
        """ Get an iterable for broadcasting """
        if product(target_shape) != self.nselect:
            raise TypeError("Broadcasting is not supported for point-wise selections")
        yield self._id


class PointSelection(SelectionBase):
    """
        Represents a point-wise selection.  You can supply sequences of
        points to the three methods append(), prepend() and set(), or a
        single boolean array to the constructor.
    """
    def __init__(self, shape, mask=None):
        self.shape = shape
        if mask is not None:
            self.points = self.mask_to_points(mask)
        else:
            # Empty list of points
            self.points = np.zeros((0, len(shape)), dtype=np.uint64)

    def apply_to_spaceid(self, spaceid):
        """ Internal method which actually performs the selection """
        points = np.asarray(self.points, order='C', dtype='u8')
        if len(points.shape) == 1:
            points.shape = (1,points.shape[0])

        if len(points) == 0:
            spaceid.select_none()
        else:
            spaceid.select_elements(points, h5s.SELECT_SET)

    def mask_to_points(self, arg):
        """ Perform point-wise selection from a NumPy boolean array """
        if not (isinstance(arg, np.ndarray) and arg.dtype.kind == 'b'):
            raise TypeError("PointSelection only works with bool arrays")
        if not arg.shape == self.shape:
            raise TypeError("Boolean indexing array has incompatible shape")

        return np.transpose(arg.nonzero())

    def append(self, points):
        """ Add the sequence of points to the end of the current selection """
        self.points = np.concatenate((self.points, points), axis=0)

    def prepend(self, points):
        """ Add the sequence of points to the beginning of the current selection """
        self.points = np.concatenate((self.points, points), axis=0)

    def set(self, points):
        """ Replace the current selection with the given sequence of points"""
        self.points = points

    @property
    def nselect(self):
        return self.points.shape[0]

    @property
    def mshape(self):
        return (self.nselect,)


class SimpleSelection(SelectionBase):

    """ A single "rectangular" (regular) selection composed of only slices
        and integer arguments.  Can participate in broadcasting.
    """

    @property
    def mshape(self):
        """ Shape of current selection """
        return self._mshape

    @property
    def nselect(self):
        return product(self._mshape)

    def __init__(self, shape, starts, counts, steps, scalar):
        self.shape = shape
        self.starts = starts
        self.counts = counts
        self.steps = steps
        self.scalar = scalar
        self._mshape = tuple([x for x, y in zip(counts, scalar) if not y])

    def apply_to_spaceid(self, spaceid):
        spaceid.select_hyperslab(self.starts, self.counts, self.steps)

    def broadcast(self, target_shape):
        """ Return an iterator over target dataspaces for broadcasting.

        Follows the standard NumPy broadcasting rules against the current
        selection shape (self.mshape).
        """
        if self.shape == ():
            if np.product(target_shape) != 1:
                raise TypeError("Can't broadcast %s to scalar" % target_shape)
            space = h5s.create(h5s.SCALAR)
            space.select_all()
            yield space
            return

        start = self.starts
        count = self.counts
        step = self.steps
        scalar = self.scalar

        rank = len(count)
        target = list(target_shape)

        tshape = []
        for idx in range(1,rank+1):
            if len(target) == 0 or scalar[-idx]:     # Skip scalar axes
                tshape.append(1)
            else:
                t = target.pop()
                if t == 1 or count[-idx] == t:
                    tshape.append(t)
                else:
                    raise TypeError("Can't broadcast %s -> %s" % (target_shape, self.mshape))

        if any([n > 1 for n in target]):
            # All dimensions from target_shape should either have been popped
            # to match the selection shape, or be 1.
            raise TypeError("Can't broadcast %s -> %s" % (target_shape, self.mshape))

        tshape.reverse()
        tshape = tuple(tshape)

        chunks = tuple(x//y for x, y in zip(count, tshape))
        nchunks = product(chunks)

        if nchunks == 1:
            yield self.id
        else:
            sid = self.id.copy()
            sid.select_hyperslab((0,)*rank, tshape, self.steps)
            for idx in range(nchunks):
                offset = tuple(x*y*z + s for x, y, z, s in zip(np.unravel_index(idx, chunks), tshape, step, start))
                sid.offset_simple(offset)
                yield sid


class FancySelection(SelectionBase):

    """
        Implements advanced NumPy-style selection operations in addition to
        the standard slice-and-int behavior.

        Indexing arguments may be ints, slices, lists of indicies, or
        per-axis (1D) boolean arrays.

        Broadcasting is not supported for these selections.
    """

    @property
    def mshape(self):
        return self._mshape

    @property
    def nselect(self):
        return product(self._mshape)

    def __init__(self, shape, starts, counts, steps, scalar, seq_dim, seq_arr):
        self.shape = shape
        self.counts = counts
        self.starts = starts
        self.steps = steps
        self.scalar = scalar
        self.seq_dim = seq_dim
        self.seq_arr = seq_arr

        # Find shape of selection
        mshape = list(counts)
        mshape[seq_dim] = len(seq_arr)
        self._mshape = tuple([x for x, y in zip(mshape, scalar) if not y])

    def apply_to_spaceid(self, spaceid):
        # Apply the selection by making several HDF5 hyperslab selections
        var_starts = list(self.starts)
        counts = self.counts
        steps = self.steps
        seq_dim = self.seq_dim
        spaceid.select_none()
        for coord in self.seq_arr:
            var_starts[seq_dim] = coord
            spaceid.select_hyperslab(tuple(var_starts), counts, steps, op=h5s.SELECT_OR)

    def broadcast(self, target_shape):
        if not target_shape == self.mshape:
            raise TypeError("Broadcasting is not supported for complex selections")
        yield self.id


def _translate_slice(exp, length):
    """ Given a slice object, return a 3-tuple
        (start, count, step)
        for use with the hyperslab selection routines
    """
    start, stop, step = exp.indices(length)
        # Now if step > 0, then start and stop are in [0, length];
        # if step < 0, they are in [-1, length - 1] (Python 2.6b2 and later;
        # Python issue 3004).

    if step < 1:
        raise ValueError("Step must be >= 1 (got %d)" % step)
    if stop < start:
        # list/tuple and numpy consider stop < start to be an empty selection
        return 0, 0, 1

    count = 1 + (stop - start - 1) // step

    return start, count, step

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

    if np.product(shape) != N:
        # This means multiple hyperslab selections are in effect,
        # so we fall back to a 1D shape
        return (N,)

    return shape
