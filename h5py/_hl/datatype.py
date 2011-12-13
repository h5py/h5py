import posixpath as pp

from ..h5t import TypeID
from .base import HLObject

class Datatype(HLObject):

    """
        Represents an HDF5 named datatype stored in a file.

        To store a datatype, simply assign it to a name in a group:

        >>> MyGroup["name"] = numpy.dtype("f")
        >>> named_type = MyGroup["name"]
        >>> assert named_type.dtype == numpy.dtype("f")
    """

    @property
    def dtype(self):
        """Numpy dtype equivalent for this datatype"""
        return self.id.dtype

    def __init__(self, bind):
        """ Create a new Datatype object by binding to a low-level TypeID.
        """
        if not isinstance(bind, TypeID):
            raise ValueError("%s is not a TypeID" % bind)
        HLObject.__init__(self, bind)

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
