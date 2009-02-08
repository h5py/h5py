
"""
    High-level access to HDF5 dataspace selections
"""
import numpy as np

from h5py import h5s

# Selection types for hyperslabs
from h5py.h5s import SELECT_SET  as SET
from h5py.h5s import SELECT_OR   as OR
from h5py.h5s import SELECT_AND  as AND
from h5py.h5s import SELECT_XOR  as XOR
from h5py.h5s import SELECT_NOTB as NOTB
from h5py.h5s import SELECT_NOTA as NOTA

def select(shape, args):
    """ Automatically determine the correct selection class, perform the
        selection, and return the selection instance.  Args may be a single
        argument or a tuple of arguments.
    """
    if not isinstance(args, tuple):
        args = (args,)

    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, Selection):
            if arg.shape == shape:
                return arg
            raise TypeError("Mismatched selection shape")
        elif isinstance(arg, np.ndarray):
            sel = PointSelection(shape)
            sel[arg]
            return sel

    for a in args:
        if not isinstance(a, slice) and a != Ellipsis:
            try:
                int(a)
            except Exception:
                sel = FancySelection(shape)
                sel[args]
                return sel
    
    sel = SimpleSelection(shape)
    sel[args]
    return sel

class Selection(object):

    """
        Base class for HDF5 dataspace selections.  Subclasses support the
        "selection protocol", which means they have at least the following
        members:
        
        __init__(shape)   => Create a new selection on "shape"-tuple
        __getitem__(args) => Perform a selection with the range specified.
                             What args are allowed depends on the
                             particular subclass in use.

        shape (read-only) =>   The shape of the dataspace.
        mshape  (read-only) => The shape of the selection region. 
                               Not guaranteed to fit within "shape", although
                               the total number of points is less than
                               product(shape).
        nselect (read-only) => Number of selected points.  Always equal to
                               product(mshape).

        broadcast(target_shape) => Return an iterable which yields dataspaces
                                   for read, based on target_shape.
    """

    def __init__(self, shape):
        shape = tuple(shape)
        self._id = h5s.create_simple(shape, (h5s.UNLIMITED,)*len(shape))
        self._id.select_all()
        self._shape = shape

    @property
    def shape(self):
        """ Shape of whole dataspace """
        return self._shape

    @property
    def nselect(self):
        seltype = self._id.get_select_type()

        if seltype == h5s.SEL_POINTS:
            return self._id.get_select_npoints()

        elif seltype == h5s.SEL_HYPERSLABS:
            return self._id.get_select_hyper_nblocks()

        elif seltype == h5s.SEL_ALL:
            return np.product(self.shape)

        elif seltype == h5s.SEL_NONE:
            return 0

        raise TypeError("Selection invalid")

class _Selection_1D(Selection):

    """
        Base class for selections which result in a 1-D shape, as with
        NumPy indexing via boolean mask arrays.
    """

    @property
    def mshape(self):
        return (self.nselect,)

    def broadcast(self, target_shape):
        """ Get an iterable for broadcasting """
        if np.product(target_shape) != self.nselect:
            raise TypeError("Broadcasting is not supported for point-wise selections")

        yield self._id

class PointSelection(_Selection_1D):

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

    def __init__(self, shape):
        Selection.__init__(self, shape)
        rank = len(self.shape)
        self._sel = ((0,)*rank, self.shape, (1,)*rank, (False,)*rank)
        self.mshape = self.shape

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

        self.mshape = tuple(x for x, y in zip(count, scalar) if not y)

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
        nchunks = np.product(chunks)

        sid = self._id.copy()
        sid.select_hyperslab((0,)*rank, tshape, step)

        for idx in xrange(nchunks):
            offset = tuple(x*y*z + s for x, y, z, s in zip(np.unravel_index(idx, chunks), tshape, step, start))
            sid.offset_simple(offset)
            yield sid


class FancySelection(Selection):

    """
        Implements advanced NumPy-style selection operations in addition to
        the standard slice-and-int behavior.

        Indexing arguments may be ints, slices, lists of indicies, or
        per-axis (1D) boolean arrays.

        Broadcasting is not supported for these selections.
    """
    def __init__(self, shape):
        Selection.__init__(self, shape)
        self._mshape = shape

    @property
    def mshape(self):
        return self._mshape

    def __getitem__(self, args):

        if not isinstance(args, tuple):
            args = (args,)

        args = _expand_ellipsis(args, len(self.shape))

        self._id.select_all()

        def perform_selection(start, count, step, idx, op=h5s.SELECT_AND):
            """ Performs a selection using start/count/step in the given axis.

            All other axes have their full range selected.  The selection is
            added to the current dataspace selection using the given operator,
            defaulting to AND.

            All arguments are ints.
            """

            start = tuple(0 if i != idx else start for i, x in enumerate(self.shape))
            count = tuple(x if i != idx else count for i, x in enumerate(self.shape))
            step  = tuple(1 if i != idx else step  for i, x in enumerate(self.shape))

            self._id.select_hyperslab(start, count, step, op=op)

        def validate_number(num, length):
            """ Validate a list member for the given axis length
            """
            try:
                num = long(num)
            except TypeError:
                raise TypeError("Illegal index: %r" % num)
            if num > length-1:
                raise IndexError('Index out of bounds: %d' % num)
            if num < 0:
                raise IndexError('Negative index not allowed: %d' % num)

        mshape = []

        for idx, (exp, length) in enumerate(zip(args, self.shape)):

            if isinstance(exp, slice):
                start, count, step = _translate_slice(exp, length)
                perform_selection(start, count, step, idx)
                mshape.append(count)

            else:

                if isinstance(exp, np.ndarray) and exp.kind == 'b':
                    exp = list(exp.nonzero()[0])

                try:
                    exp = list(exp)     
                except TypeError:
                    exp = [exp]         # Handle scalar index as a list of length 1
                    mshape.append(0)    # Keep track of scalar index for NumPy
                else:
                    mshape.append(len(exp))

                if len(exp) == 0:
                    raise TypeError("Empty selections are not allowed (axis %d)" % idx)

                last_idx = -1
                for select_idx in xrange(len(exp)+1):

                    # This crazy piece of code performs a list selection
                    # using HDF5 hyperslabs.
                    # For each index, perform a "NOTB" selection on every
                    # portion of *this axis* which falls *outside* the list
                    # selection.  For this to work, the input array MUST be
                    # monotonically increasing.

                    if select_idx < last_idx:
                        raise ValueError("Selection lists must be in increasing order")
                    validate_number(select_idx, length)

                    if select_idx == 0:
                        start = 0
                        count = exp[0]
                    elif select_idx == len(exp):
                        start = exp[-1]+1
                        count = length-start
                    else:
                        start = exp[select_idx-1]+1
                        count = exp[select_idx] - start
                    if count > 0:
                        perform_selection(start, count, 1, idx, op=h5s.SELECT_NOTB)

                    last_idx = select_idx

        self._mshape = tuple(x for x in mshape if x != 0)

    def broadcast(self, target_shape):
        if not target_shape == self.mshape:
            raise TypeError("Broadcasting is not supported for complex selections")
        yield self._id

def _expand_ellipsis(args, rank):
    """ Expand ellipsis objects and fill in missing axes.
    """
    n_el = list(args).count(Ellipsis)
    if n_el > 1:
        raise ValueError("Only one ellipsis may be used.")
    elif n_el == 0 and len(args) != rank:
        args = args + (Ellipsis,)

    final_args = []
    n_args = len(args)
    for idx, arg in enumerate(args):

        if arg == Ellipsis:
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
    start, stop, step = exp.start, exp.stop, exp.step
    start = 0 if start is None else int(start)
    stop = length if stop is None else int(stop)
    step = 1 if step is None else int(step)

    if step < 1:
        raise ValueError("Step must be >= 1 (got %d)" % step)
    if stop == start:
        raise ValueError("Zero-length selections are not allowed")
    if stop < start:
        raise ValueError("Reverse-order selections are not allowed")
    if start < 0:
        start = length+start
    if stop < 0:
        stop = length+stop

    if not 0 <= start <= (length-1):
        raise ValueError("Start index %s out of range (0-%d)" % (start, length-1))
    if not 1 <= stop <= length:
        raise ValueError("Stop index %s out of range (1-%d)" % (stop, length))

    count = (stop-start)//step
    if (stop-start) % step != 0:
        count += 1

    if start+count > length:
        raise ValueError("Selection out of bounds (%d; axis has %d)" % (start+count,length))

    return start, count, step

def CoordsList(*args, **kwds):

    raise NotImplementedError("CoordsList indexing is unavailable as of 1.1.\n"
                              "Please use the selections module instead")




