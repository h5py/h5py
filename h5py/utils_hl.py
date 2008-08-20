
"""
    Utility functions for high-level modules.
"""
from __future__ import with_statement
from h5py import h5s

from posixpath import basename, normpath
import numpy

def hbasename(name):
    """ Basename function with more readable handling of trailing slashes"""
    bname = normpath(name)
    bname = basename(bname)
    if bname == '':
        bname = '/'
    return bname

class FlatIndexer(object):

    """
        Utility class which encapsulates a 1-D selection into an n-D array.

    """

    def __init__(self, shape, args):
        """ Shape must be a tuple; args must be iterable.
        """
        try:
            args = iter(args)
        except TypeError:
            args = (args,)

        points = []

        for arg in args:
            if isinstance(arg, slice):
                points.extend(xrange(*arg.indices(numpy.product(shape))))
            elif isinstance(arg, int) or isinstance(arg, long):
                points.append(arg)
            else:
                raise ValueError("Illegal index (ints, longs or slices only)")

        self.coords = numpy.array([numpy.unravel_index(x, shape) for x in points])

def slice_select(space, args):
    """ Perform a selection on the given HDF5 dataspace, using a tuple
        of Python extended slice objects.  The dataspace may be scalar or
        simple.  The slice argument may be:

        0-tuple:
            Entire dataspace selected (compatible with scalar)

        1-tuple:
            1. A single Ellipsis: entire dataspace selected
            2. A single integer or slice (row-broadcasting)
            3. A NumPy array: element-wise selection
            4. A FlatIndexer instance containing a coordinate list

        n-tuple:
            1. slice objects
            2. Ellipsis objects
            3. Integers

        The return value is the appropriate memory dataspace to use.
    """

    if len(args) == 0 or (len(args) == 1 and args[0] is Ellipsis):
        space.select_all()
        return space.copy()

    if len(args) == 1:
        argval = args[0]

        if isinstance(argval, numpy.ndarray):
            # Catch element-wise selection
            indices = numpy.transpose(argval.nonzero())
            space.select_elements(indices)
            return h5s.create_simple((len(indices),))

        if isinstance(argval, FlatIndexer):
            space.select_elements(argval.coords)
            npoints = space.get_select_elem_npoints()
            return h5s.create_simple((npoints,))

        # Single-index obj[0] access is always equivalent to obj[0,...].
        # Pack it back up and send it to the hyperslab machinery
        args = (argval, Ellipsis)

    # Proceed to hyperslab selection

    shape = space.shape
    rank = len(shape)

    start = []
    count = []
    stride = []

    # Expand integers and ellipsis arguments to slices
    for dim, arg in enumerate(args):

        if isinstance(arg, int) or isinstance(arg, long):
            if arg < 0:
                raise ValueError("Negative indices are not allowed.")
            start.append(arg)
            count.append(1)
            stride.append(1)

        elif isinstance(arg, slice):

            # slice.indices() method clips, so do it the hard way...

            # Start
            if arg.start is None:
                ss=0
            else:
                if arg.start < 0:
                    raise ValueError("Negative dimensions are not allowed")
                ss=arg.start

            # Stride
            if arg.step is None:
                st = 1
            else:
                if arg.step <= 0:
                    raise ValueError("Only positive step sizes allowed")
                st = arg.step

            # Count
            if arg.stop is None:
                cc = shape[dim]/st
            else:
                if arg.stop < 0:
                    raise ValueError("Negative dimensions are not allowed")
                cc = (arg.stop-ss)/st
                if ((arg.stop-ss) % st) != 0:
                    cc += 1   # Be careful with integer division!
            if cc == 0:
                raise ValueError("Zero-length selections are not allowed")

            start.append(ss)
            stride.append(st)
            count.append(cc)

        elif arg == Ellipsis:
            nslices = rank-(len(args)-1)
            if nslices <= 0:
                continue
            for x in range(nslices):
                idx = dim+x
                start.append(0)
                count.append(shape[dim+x])
                stride.append(1)

        else:
            raise ValueError("Bad slice type %s" % repr(arg))

    space.select_hyperslab(tuple(start), tuple(count), tuple(stride))
    return h5s.create_simple(tuple(count))

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







