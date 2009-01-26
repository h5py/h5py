
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

def is_simple(args):
    for arg in args:
        if not isinstance(arg, slice):
            try:
                long(arg)
            except Exception:
                return False
    return True

class Selection(object):

    """
        Base class for HDF5 dataspace selections
    """

    def __init__(self, shape):
        shape = tuple(shape)
        self._id = h5s.create_simple(shape, (h5s.UNLIMITED,)*len(shape))
        self._shape = shape

    @property
    def shape(self):
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

    def shape_broadcast(self, shape):
        """ Stub broadcasting method """
        if not shape == self.shape:
            raise TypeError("Broadcasting is only supported for simple selections")
        yield self._id

class PointSelection(Selection):

    """
        Represents a point-wise selection.
    """

    def _perform_selection(self, points, op):

        points = np.asarray(points, order='C')
        if len(points.shape) == 1:
            points.shape = (1,points.shape[0])

        if self._id.get_select_type() != h5s.SEL_POINTS:
            op = h5s.SELECT_SET

        self._id.select_elements(points, op)

    def append(self, points):
        self._perform_selection(points, h5s.SELECT_APPEND)

    def prepend(self, points):
        self._perform_selection(points, h5s.SELECT_PREPEND)

    def set(self, points):
        self._perform_selection(points, h5s.SELECT_SET)


class RectSelection(Selection):

    """ A single "rectangular" (regular) selection composed of only slices
        and integer arguments.  Can participate in broadcasting.
    """

    def __init__(self, *args, **kwds):
        Selection.__init__(self, *args, **kwds)
        rank = len(self.shape)
        self._sel = ((0,)*rank, self.shape, (1,)*rank, (False,)*rank)
        self.mshape = self.shape

    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = (args,)
  
        start, count, step, scalar = self._handle_args(args)

        self._id.select_hyperslab(start, count, step)

        self._sel = (start, count, step, scalar)

        self.mshape = tuple(x for x, y in zip(count, scalar) if not y)

        return self._id


    def shape_broadcast(self, target_shape):
        """ Return an iterator over target dataspaces for broadcasting """

        # count = (10,10,10)
        # cshape = (1,1,5)

        start, count, step, scalar = self._sel

        rank = len(count)
        target = list(target_shape)

        tshape = []
        for idx in xrange(1,rank+1):
            if len(target) == 0 or scalar[-idx]:     # Skip scalar axes
                tshape.append(1)
            else:
                t = target.pop()
                if count[-idx] == t or t == 1:
                    tshape.append(t)
                else:
                    raise TypeError("Can't broadcast %s -> %s [%s,%s,%s] %s\n%s" % (target_shape, count, count[-idx], t, -idx, tshape, self._sel))
        tshape.reverse()
        tshape = tuple(tshape)

        chunks = tuple(x/y for x, y in zip(count, tshape))

        #print tshape, chunks

        nchunks = np.product(chunks)

        sid = self._id.copy()
        sid.select_hyperslab((0,)*rank, tshape, step)

        for idx in xrange(nchunks):
            offset = tuple(x*y*z + s for x, y, z, s in zip(np.unravel_index(idx, chunks), tshape, step, start))
            sid.offset_simple(offset)
            yield sid

    def _handle_args(self, args):
        """ Process a "simple" selection tuple, containing only slices and
            integer objects.  Return is a 3-tuple with start, count, step tuples.

            If "args" is shorter than "shape", the remaining axes are fully
            selected.
        """
        args = _broadcast(args, len(self.shape))

        start = []
        count = []
        step  = []
        scalar = []

        for arg, length in zip(args, self.shape):
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

class HyperSelection(RectSelection):

    """
        Represents multiple overlapping rectangular selections, combined
        with set-like operators.

        When created, the entire dataspace is selected.  To make
        adjustments to the selection, use the standard NumPy slicing
        syntax::

            >>> sel = HyperSelection((10,20,20))  # Initially 10 x 20 x 20
            >>> sel[:,5:15,:] = SET               # Now 10 x 10 x 20
            >>> sel[0:5,:,:] = AND                # Now  5 x 10 x 10

        Legal operators (in the h5py.selections module) are:
            
        SET
            New selection, wiping out any old one
        
        OR (or True), AND, XOR
            Logical OR/AND/XOR between new and old selection

        NOTA
            Select only regions in new selection which don't intersect the old

        NOTB (or False)
            Select only regions in old selection which don't intersect the new
  
    """

    def __setitem__(self, args, op):

        if not isinstance(args, tuple):
            args = (args,)
  
        start, count, step = self._handle_args(args)

        if not op in (SET, OR, AND, XOR, NOTB, NOTA, True, False):
            raise ValueError("Illegal selection operator")

        if op is True:
            op = OR
        elif op is False:
            op = NOTB

        seltype == self._id.get_select_type()

        if seltype == h5s.SEL_ALL:
            self._id.select_hyperslab((0,)*len(self.shape), self.shape, op=h5s.SELECT_SET)
        
        elif seltype == h5s.SEL_NONE:
            if op in (SET, OR, XOR, NOTA):
                op = SET
            else:
                return

        self._id.select_hyperslab(start, count, step, op=op)



class FancySelection(HyperSelection):

    """
        Implements advanced, NumPy-style selection operations.

        Indexing arguments may be ints, slices, lists of indicies, or
        boolean arrays (1-D).  The only permitted operation is SET.

        Intended for internal use by the Dataset __getitem__ machinery.
    """

    def __setitem__(self, args, op):

        if op != SET:
            raise ValueError("The only permitted operation is SET")
        if not isinstance(args, tuple):
            args = (args,)

        args = _broadcast(args, len(self.shape))

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

            elif isinstance(exp, np.ndarray) and exp.kind == 'b':

                raise NotImplementedError() # TODO: bool vector

            else:

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

        self.mshape = tuple(x for x in mshape if x != 0)

def _broadcast(args, rank):
    """ Expand ellipsis objects and fill in missing axes.  Returns the
    new args tuple.
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



