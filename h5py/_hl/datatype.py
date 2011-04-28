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

    def __init__(self, grp, name, bind=None):
        """ Private constructor.
        """
        id_ = bind if bind is not None else h5t.open(grp.id, name)
        HLObject.__init__(self, id_)

    def __repr__(self):
        if not self.id:
            return "<Closed HDF5 named type>"
        namestr = '"%s"' % _extras.basename(self.name) if self.name is not None else "(anonymous)"
        return '<HDF5 named type %s (dtype %s)>' % \
            (namestr, self.dtype.str)
