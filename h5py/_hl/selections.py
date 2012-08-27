#+
# 
# This file is part of h5py, a low-level Python interface to the HDF5 library.
# 
# Copyright (C) 2008 Andrew Collette
# http://h5py.alfven.org
# License: BSD  (See LICENSE.txt for full license)
# 
# $Date$
# 
#-

"""
    High-level access to HDF5 dataspace selections
"""

import numpy as np

from h5py import h5s, h5r

# Selection types for hyperslabs
from h5py.h5s import SELECT_SET  as SET
from h5py.h5s import SELECT_OR   as OR
from h5py.h5s import SELECT_AND  as AND
from h5py.h5s import SELECT_XOR  as XOR
from h5py.h5s import SELECT_NOTB as NOTB
from h5py.h5s import SELECT_NOTA as NOTA

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

    # "Special" indexing objects
    if len(args) == 1:

        arg = args[0]
        if isinstance(arg, Selection):
            if arg.shape != shape:
                raise TypeError("Mismatched selection shape")
            return arg

        elif isinstance(arg, np.ndarray):
            sel = PointSelection(shape)
            sel[arg]
            return sel

        elif isinstance(arg, h5r.RegionReference):
            sid = h5r.get_region(arg, dsid)
            if shape != sid.shape:
                raise TypeError("Reference shape does not match dataset shape")
                
            return Selection(shape, spaceid=sid)

    for a in args:
        if not isinstance(a, slice) and a is not Ellipsis:
            try:
                int(a)
            except Exception:
                sel = FancySelection(shape)
                sel[args]
                return sel
    
    sel = SimpleSelection(shape)
    sel[args]
    return sel

class _RegionProxy(object):

    """
        Thin proxy object which takes __getitem__-style index arguments and
        produces RegionReference objects.  Example:

        >>> dset = myfile['dataset']
        >>> myref = dset.regionref[0:100,20:30]
        >>> data = dset[myref]

    """

    def __init__(self, dsid):
        """ Supply a h5py.h5d.DatasetID instance """
        self.id = dsid

    def __getitem__(self, args):
        """ Takes arbitrary selection terms and produces a RegionReference
        object.  Selection must be compatible with the dataset.
        """
        selection = select(self.id.shape, args)
        return h5r.create(self.id, '.', h5r.DATASET_REGION, selection.id)

class Selection(object):

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

    def broadcast(self, target_shape):
        """ Get an iterable for broadcasting """
        if np.product(target_shape) != self.nselect:
            raise TypeError("Broadcasting is not supported for point-wise selections")
        yield self._id

    def __getitem__(self, args):
        raise NotImplementedError("This class does not support indexing")

class PointSelection(Selection):

    """
        Represents a point-wise selection.  You can supply sequences of
        points to the three methods append(), prepend() and set(), or a
        single boolean array to __getitem__.
    """

    def _perform_selection(self, points, op):

        points = np.asarray(points, order='C', dtype='u8')
        if len(points.shape) == 1:
            points.shape = (1,points.shape[0])

        if self._id.get_select_type() != h5s.SEL_POINTS:
            op = h5s.SELECT_SET

        if len(points) == 0:
            self._id.select_none()
        else:
            self._id.select_elements(points, op)

    def __getitem__(self, arg):
        """ Perform point-wise selection from a NumPy boolean array """
        if not (isinstance(arg, np.ndarray) and arg.dtype.kind == 'b'):
            raise TypeError("PointSelection __getitem__ only works with bool arrays")
        if not arg.shape == self.shape:
            raise TypeError("Boolean indexing array has incompatible shape")

        points = np.transpose(arg.nonzero())
        self.set(points)
        return self

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
        return self._mshape

    def __init__(self, shape, *args, **kwds):
        Selection.__init__(self, shape, *args, **kwds)
        rank = len(self.shape)
        self._sel = ((0,)*rank, self.shape, (1,)*rank, (False,)*rank)
        self._mshape = self.shape

    def __getitem__(self, args):

        if not isinstance(args, tuple):
            args = (args,)
  
        if self.shape == ():
            if len(args) > 0 and args[0] not in (Ellipsis, ()):
                raise TypeError("Invalid index for scalar dataset (only ..., () allowed)")
            self._id.select_all()
            return self

        start, count, step, scalar = _handle_simple(self.shape,args)

        self._id.select_hyperslab(start, count, step)

        self._sel = (start, count, step, scalar)

        self._mshape = tuple(x for x, y in zip(count, scalar) if not y)

        return self


    def broadcast(self, target_shape):
        """ Return an iterator over target dataspaces for broadcasting.

        Follows the standard NumPy broadcasting rules against the current
        selection shape (self.mshape).
        """
        if self.shape == ():
            if np.product(target_shape) != 1:
                raise TypeError("Can't broadcast %s to scalar" % target_shape)
            self._id.select_all()
            yield self._id
            return

        start, count, step, scalar = self._sel

        rank = len(count)
        target = list(target_shape)

        tshape = []
        for idx in xrange(1,rank+1):
            if len(target) == 0 or scalar[-idx]:     # Skip scalar axes
                tshape.append(1)
            else:
                t = target.pop()
                if t == 1 or count[-idx] == t:
                    tshape.append(t)
                else:
                    raise TypeError("Can't broadcast %s -> %s" % (target_shape, count))
        tshape.reverse()
        tshape = tuple(tshape)

        chunks = tuple(x/y for x, y in zip(count, tshape))
        nchunks = long(np.product(chunks))

        if nchunks == 1:
            yield self._id
        else:
            sid = self._id.copy()
            sid.select_hyperslab((0,)*rank, tshape, step)
            for idx in xrange(nchunks):
                offset = tuple(x*y*z + s for x, y, z, s in zip(np.unravel_index(idx, chunks), tshape, step, start))
                sid.offset_simple(offset)
                yield sid


class HyperSelection(Selection):

    """
        Represents multiple overlapping rectangular selections, combined
        with set-like operators.  Result is a 1D shape, as with boolean array
        selection.  Broadcasting is not supported for these selections.

        When created, the entire dataspace is selected.  To make
        adjustments to the selection, use the standard NumPy slicing
        syntax, either via __getitem__ (as with simple selections) or via
        __setitem__ and one of the supported operators:

            >>> sel = HyperSelection((10,20))  # Initially 200 points
            >>> sel[:,5:15] = False            # Now 100 points
            >>> sel[:,10]   = True             # Now 110 points
            >>> sel[...]    = XOR              # Now 90 points

        Legal operators (in the h5py.selections module) are:
           
        SET
            New selection, wiping out any old one
       
        AND, XOR, OR (or True)
            Logical AND/XOR/OR between new and old selection

        NOTA
            Select only regions in new selection which don't intersect the old

        NOTB (or False)
            Select only regions in old selection which don't intersect the new
 
    """

    def __getitem__(self, args):
        self[args] = SET
        return self

    def __setitem__(self, args, op):

        if not isinstance(args, tuple):
            args = (args,)
 
        start, count, step, scalar = _handle_simple(self.shape, args)

        if not op in (SET, OR, AND, XOR, NOTB, NOTA, True, False):
            raise ValueError("Illegal selection operator")

        if op is True:
            op = OR
        elif op is False:
            op = NOTB

        seltype = self._id.get_select_type()

        if seltype == h5s.SEL_ALL:
            self._id.select_hyperslab((0,)*len(self.shape), self.shape, op=h5s.SELECT_SET)
       
        elif seltype == h5s.SEL_NONE:
            if op in (SET, OR, XOR, NOTA):
                op = SET
            else:
                return

        self._id.select_hyperslab(start, count, step, op=op)

class FancySelection(Selection):

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

    def __init__(self, shape, *args, **kwds):
        Selection.__init__(self, shape, *args, **kwds)
        self._mshape = self.shape

    def __getitem__(self, args):

        if not isinstance(args, tuple):
            args = (args,)

        args = _expand_ellipsis(args, len(self.shape))

        # First build up a dictionary of (position:sequence) pairs

        sequenceargs = {}
        for idx, arg in enumerate(args):
            if not isinstance(arg, slice):
                if hasattr(arg, 'dtype') and arg.dtype == np.dtype('bool'):
                    if len(arg.shape) != 1:
                        raise TypeError("Boolean indexing arrays must be 1-D")
                    arg = arg.nonzero()[0]
                try:
                    sequenceargs[idx] = list(arg)
                except TypeError:
                    pass
                else:
                    if sorted(arg) != list(arg):
                        raise TypeError("Indexing elements must be in increasing order")

        if len(sequenceargs) > 1:
            # TODO: fix this with broadcasting
            raise TypeError("Only one indexing vector or array is currently allowed for advanced selection")
        if len(sequenceargs) == 0:
            # TODO: fallback to standard selection
            raise TypeError("Advanced selection inappropriate")

        vectorlength = len(sequenceargs.values()[0])
        if not all(len(x) == vectorlength for x in sequenceargs.values()):
            raise TypeError("All sequence arguments must have the same length %s" % sequenceargs)

        # Now generate a vector of selection lists,
        # consisting only of slices and ints

        argvector = []
        for idx in xrange(vectorlength):
            entry = list(args)
            for position, seq in sequenceargs.iteritems():
                entry[position] = seq[idx]
            argvector.append(entry)

        # "OR" all these selection lists together to make the final selection

        self._id.select_none()
        for idx, vector in enumerate(argvector):
            start, count, step, scalar = _handle_simple(self.shape, vector)
            self._id.select_hyperslab(start, count, step, op=h5s.SELECT_OR)

        # Final shape excludes scalars, except where
        # they correspond to sequence entries

        mshape = list(count)
        for idx in xrange(len(mshape)):
            if idx in sequenceargs:
                mshape[idx] = len(sequenceargs[idx])
            elif scalar[idx]:
                mshape[idx] = 0

        self._mshape = tuple(x for x in mshape if x != 0)

    def broadcast(self, target_shape):
        if not target_shape == self.mshape:
            raise TypeError("Broadcasting is not supported for complex selections")
        yield self._id

def _expand_ellipsis(args, rank):
    """ Expand ellipsis objects and fill in missing axes.
    """
    n_el = sum(1 for arg in args if arg is Ellipsis)
    if n_el > 1:
        raise ValueError("Only one ellipsis may be used.")
    elif n_el == 0 and len(args) != rank:
        args = args + (Ellipsis,)

    final_args = []
    n_args = len(args)
    for idx, arg in enumerate(args):

        if arg is Ellipsis:
            final_args.extend( (slice(None,None,None),)*(rank-n_args+1) )
        else:
            final_args.append(arg)

    if len(final_args) > rank:
        raise TypeError("Argument sequence too long")

    return final_args

def _handle_simple(shape, args):
    """ Process a "simple" selection tuple, containing only slices and
        integer objects.  Return is a 4-tuple with tuples for start,
        count, step, and a flag which tells if the axis is a "scalar"
        selection (indexed by an integer).

        If "args" is shorter than "shape", the remaining axes are fully
        selected.
    """
    args = _expand_ellipsis(args, len(shape))

    start = []
    count = []
    step  = []
    scalar = []

    for arg, length in zip(args, shape):
        if isinstance(arg, slice):
            x,y,z = _translate_slice(arg, length)
            s = False
        else:
            try:
                x,y,z = _translate_int(int(arg), length)
                s = True
            except TypeError:
                raise TypeError('Illegal index "%s" (must be a slice or number)' % arg)
        start.append(x)
        count.append(y)
        step.append(z)
        scalar.append(s)

    return tuple(start), tuple(count), tuple(step), tuple(scalar)

def _translate_int(exp, length):
    """ Given an integer index, return a 3-tuple
        (start, count, step)
        for hyperslab selection
    """
    if exp < 0:
        exp = length+exp

    if not 0<=exp<length:
        raise ValueError("Index (%s) out of range (0-%s)" % (exp, length-1))

    return exp, 1, 1

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
    if stop == start:
        raise ValueError("Zero-length selections are not allowed")
    if stop < start:
        raise ValueError("Reverse-order selections are not allowed")

    count = 1 + (stop - start - 1) // step

    return start, count, step





