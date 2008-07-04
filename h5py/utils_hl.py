
"""
    Utility functions for high-level modules.
"""
from posixpath import basename, normpath

def hbasename(name):
    """ Basename function with more readable handling of trailing slashes"""
    bname = normpath(name)
    bname = basename(bname)
    if bname == '':
        bname = '/'
    return bname

def slicer(shape, args):
    """ Parse a tuple containing Numpy-style extended slices.
        Shape should be a Numpy-style shape tuple.

        Arguments may contain:

        1. Integer/long indices
        2. Slice objects
        3. Exactly one ellipsis object
        4. Strings representing field names (zero or more, in any order)

        Return is a 4-tuple containing sub-tuples:
        (start, count, stride, names)

        start:  tuple with starting indices
        count:  how many elements to select along each axis
        stride: selection pitch for each axis
        names:  field names (i.e. for compound types)
    """

    rank = len(shape)

    if not isinstance(args, tuple):
        args = (args,)
    args = list(args)

    slices = []
    names = []

    # Sort arguments
    for entry in args[:]:
        if isinstance(entry, str):
            names.append(entry)
        else:
            slices.append(entry)

    start = []
    count = []
    stride = []

    # Hack to allow Numpy-style row indexing
    if len(slices) == 1 and slices[0] != Ellipsis:
        args.append(Ellipsis)

    # Expand integers and ellipsis arguments to slices
    for dim, arg in enumerate(slices):

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
            if cc == 0:
                raise ValueError("Zero-length selections are not allowed")

            start.append(ss)
            stride.append(st)
            count.append(cc)

        elif arg == Ellipsis:
            nslices = rank-(len(slices)-1)
            if nslices <= 0:
                continue
            for x in range(nslices):
                idx = dim+x
                start.append(0)
                count.append(shape[dim+x])
                stride.append(1)

        else:
            raise ValueError("Bad slice type %s" % repr(arg))

    return (tuple(start), tuple(count), tuple(stride), tuple(names))

