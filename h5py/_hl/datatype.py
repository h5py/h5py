# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Implements high-level access to committed datatypes in the file.
"""

from __future__ import absolute_import

import posixpath as pp
from numpy import dtype

from ..h5t import TypeID, create_opaque as create_opaque_ll
from .base import HLObject, with_phil


def create_opaque(dt_in):
    """ (dtype dt_in, bytes tag=None)

    Register a NumPy dtype for use with h5py. Types registered in this way
    will be stored as a custom opaque type, with a special tag to map it to
    the corresponding NumPy type.

    Opaque types with this tag will be mapped to NumPy types in the same way.

    The default tag is generated via the code:
    ``b"NUMPY:" + dt_in.descr[0][1].encode()``.
    """
    dt_in = dtype(dt_in)
    return Datatype(create_opaque_ll(dt_in))


class Datatype(HLObject):

    """
        Represents an HDF5 named datatype stored in a file.

        To store a datatype, simply assign it to a name in a group:

        >>> MyGroup["name"] = numpy.dtype("f")
        >>> named_type = MyGroup["name"]
        >>> assert named_type.dtype == numpy.dtype("f")
    """

    @property
    @with_phil
    def dtype(self):
        """Numpy dtype equivalent for this datatype"""
        return self.id.dtype

    @with_phil
    def __init__(self, bind):
        """ Create a new Datatype object by binding to a low-level TypeID.
        """
        if not isinstance(bind, TypeID):
            raise ValueError("%s is not a TypeID" % bind)
        HLObject.__init__(self, bind)

    @with_phil
    def __repr__(self):
        if not self.id:
            return "<Closed HDF5 named type>"
        if self.name is None:
            namestr = '("anonymous")'
        else:
            name = pp.basename(pp.normpath(self.name))
            namestr = '"%s"' % (name if name != '' else '/')
        return '<HDF5 named type %s (dtype %s)>' % \
            (namestr, self.dtype.str)
