
"""
    Utility functions for high-level modules.
"""
from __future__ import with_statement
from h5py import h5s

from posixpath import basename, normpath
import numpy

CHUNK_BASE = 16*1024    # Multiplier by which chunks are adjusted
MIN_CHUNK = 8*1024      # Soft lower limit (8k)
MAX_CHUNK = 1024*1024   # Hard upper limit (1M)

def hbasename(name):
    """ Basename function with more readable handling of trailing slashes"""
    bname = normpath(name)
    bname = basename(bname)
    if bname == '':
        bname = '/'
    return bname

def guess_chunk(shape, typesize):
    """ Guess an appropriate chunk layout for a dataset, given its shape and
        the size of each element in bytes.  Will allocate chunks only as large
        as MAX_SIZE.  Chunks are generally close to some power-of-2 fraction of
        each axis, slightly favoring bigger values for the last index.
    """

    ndims = len(shape)
    if ndims == 0:
        raise ValueError("Chunks not allowed for scalar datasets.")

    chunks = numpy.array(shape, dtype='=f8')

    # Determine the optimal chunk size in bytes using a PyTables expression.
    # This is kept as a float.
    dset_size = numpy.product(chunks)*typesize
    target_size = CHUNK_BASE * (2**numpy.log10(dset_size/(1024.*1024)))

    if target_size > MAX_CHUNK:
        target_size = MAX_CHUNK
    elif target_size < MIN_CHUNK:
        target_size = MIN_CHUNK

    idx = 0
    while True:
        # Repeatedly loop over the axes, dividing them by 2.  Stop when:
        # 1a. We're smaller than the target chunk size, OR
        # 1b. We're within 50% of the target chunk size, AND
        #  2. The chunk is smaller than the maximum chunk size

        chunk_bytes = numpy.product(chunks)*typesize

        if (chunk_bytes < target_size or \
         abs(chunk_bytes-target_size)/target_size < 0.5) and \
         chunk_bytes < MAX_CHUNK:
            break

        chunks[idx%ndims] = numpy.ceil(chunks[idx%ndims] / 2.0)
        idx += 1

    return tuple(long(x) for x in chunks)

class FlatIndexer(object):

    """
        Utility class which encapsulates a 1-D selection into an n-D array.

    """

    def __init__(self, shape, args):
        """ Shape must be a tuple; args must be iterable.
        """
        if shape == ():
            raise TypeError("Can't slice into a scalar array.")

        try:
            args = tuple(iter(args))
        except TypeError:
            args = (args,)

        points = []

        scalarok = False
        for arg in args:
            if isinstance(arg, slice):
                points.extend(xrange(*arg.indices(numpy.product(shape))))
            else:
                try:
                    points.append(long(arg))
                except TypeError:
                    raise ValueError("Illegal index (ints, longs or slices only)")
                scalarok = True

        self.coords = numpy.array([numpy.unravel_index(x, shape) for x in points])

        # A scalar value should result for a single integer index.
        self.scalar = True if scalarok and len(args) == 1 else False


def slice_select(space, args):
    """ Perform a selection on the given HDF5 dataspace, using a tuple
        of Python extended slice objects.  The dataspace may be scalar or
        simple.  The following selection mechanisms are implemented:

        1. select_all:
            0-tuple
            1-tuple containing Ellipsis

        2. Hyperslab selection
            n-tuple (n>1) containing slice/integer/Ellipsis objects

        3. Discrete element selection
            1-tuple containing boolean array or FlatIndexer

        The return value is a 2-tuple:
        1. Appropriate memory dataspace to use for new array
        2. Boolean indicating if the slice should result in a scalar quantity
    """

    if len(args) == 0 or (len(args) == 1 and args[0] is Ellipsis):
        # The only safe way to access a scalar dataspace
        space.select_all()
        return space.copy(), False
    else:
        if space.get_simple_extent_type() == h5s.SCALAR:
            raise TypeError('Can\'t slice a scalar dataset (only fields and "..." allowed)')

    if len(args) == 1:
        argval = args[0]

        if isinstance(argval, numpy.ndarray):
            # Boolean array indexing is handled by discrete element selection
            # It never results in a scalar value
            indices = numpy.transpose(argval.nonzero())
            space.select_elements(indices)
            return h5s.create_simple((len(indices),)), False

        if isinstance(argval, FlatIndexer):
            # Flat indexing also uses discrete selection
            # Scalar determination is made by the indexer
            space.select_elements(argval.coords)
            npoints = space.get_select_elem_npoints()
            return h5s.create_simple((npoints,)), argval.scalar

    # Proceed to hyperslab selection

    shape = space.shape
    rank = len(shape)

    # First expand (at most 1) ellipsis object

    n_el = list(args).count(Ellipsis)
    if n_el > 1:
        raise ValueError("Only one ellipsis may be used.")
    elif n_el == 0 and len(args) != rank:
        args = args + (Ellipsis,)  # Simple version of NumPy broadcasting

    final_args = []
    n_args = len(args)

    for idx, arg in enumerate(args):

        if arg == Ellipsis:
            final_args.extend( (slice(None,None,None),)*(rank-n_args+1) )
        else:
            final_args.append(arg)

    # Step through the expanded argument list and handle each axis

    start = []
    count = []
    stride = []
    simple = []
    for idx, (length, exp) in enumerate(zip(shape,final_args)):

        if isinstance(exp, slice):

            # slice.indices() method is limited to long ints

            start_, stop_, step_ = exp.start, exp.stop, exp.step
            start_ = 0 if start_ is None else int(start_)
            stop_ = length if stop_ is None else int(stop_)
            step_ = 1 if step_ is None else int(step_)

            if start_ < 0:
                raise ValueError("Negative start index not allowed (got %d)" % start_)
            if step_ < 1:
                raise ValueError("Step must be >= 1 (got %d)" % step_)
            if stop_ < 0:
                raise ValueError("Negative stop index not allowed (got %d)" % stop_)

            count_ = (stop_-start_)//step_
            if (stop_-start_) % step_ != 0:
                count_ += 1

            if start_+count_ > length:
                raise ValueError("Selection out of bounds on axis %d" % idx)

            simple_ = False

        else:
            try:
                exp = long(exp)
            except TypeError:
                raise TypeError("Illegal index on axis %d: %r" % (idx, exp))

            if exp > length-1:
                raise IndexError('Index %d out of bounds: "%d" (should be <= %d)' % (idx, exp, length-1))

            start_ = exp
            step_ = 1
            count_ = 1
            simple_ = True

        start.append(start_)
        count.append(count_)
        stride.append(step_)
        simple.append(simple_)

    space.select_hyperslab(tuple(start), tuple(count), tuple(stride))

    # According to the NumPy rules, dimensions which are specified as an int
    # do not result in a length-1 axis.
    mem_shape = tuple(x for x, smpl in zip(count, simple) if not smpl) 

    return h5s.create_simple(mem_shape), all(simple)

def strhdr(line, char='-'):
    """ Print a line followed by an ASCII-art underline """
    return line + "\n%s\n" % (char*len(line))

def strlist(lst, keywidth=10):
    """ Print a list of (key: value) pairs, with column alignment. """
    format = "%-"+str(keywidth)+"s %s\n"

    outstr = ''
    for key, val in lst:
        outstr += format % (key+':',val)

    return outstr







