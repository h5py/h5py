import numpy

from h5py import h5s, h5t, h5a
from . import base
from .dataset import readtime_dtype

class AttributeManager(base.DictCompat, base.CommonStateObject):

    """ 
        Allows dictionary-style access to an HDF5 object's attributes.

        These are created exclusively by the library and are available as
        a Python attribute at <object>.attrs

        Like Group objects, attributes provide a minimal dictionary-
        style interface.  Anything which can be reasonably converted to a
        Numpy array or Numpy scalar can be stored.

        Attributes are automatically created on assignment with the
        syntax <obj>.attrs[name] = value, with the HDF5 type automatically
        deduced from the value.  Existing attributes are overwritten.

        To modify an existing attribute while preserving its type, use the
        method modify().  To specify an attribute of a particular type and
        shape, use create().
    """

    def __init__(self, parent):
        """ Private constructor.
        """
        self._id = parent.id

    def __getitem__(self, name):
        """ Read the value of an attribute.
        """
        attr = h5a.open(self._id, self._e(name))

        tid = attr.get_type()

        rtdt = readtime_dtype(attr.dtype, [])

        arr = numpy.ndarray(attr.shape, dtype=rtdt, order='C')
        attr.read(arr)

        if len(arr.shape) == 0:
            return arr[()]
        return arr

    def __setitem__(self, name, value):
        """ Set a new attribute, overwriting any existing attribute.

        The type and shape of the attribute are determined from the data.  To
        use a specific type or shape, or to preserve the type of an attribute,
        use the methods create() and modify().
        """
        self.create(name, data=value, dtype=base.guess_dtype(value))

    def __delitem__(self, name):
        """ Delete an attribute (which must already exist). """
        h5a.delete(self._id, self._e(name))

    def create(self, name, data, shape=None, dtype=None):
        """ Create a new attribute, overwriting any existing attribute.

        name
            Name of the new attribute (required)
        data
            An array to initialize the attribute (required)
        shape
            Shape of the attribute.  Overrides data.shape if both are
            given, in which case the total number of points must be unchanged.
        dtype
            Data type of the attribute.  Overrides data.dtype if both
            are given.
        """
        # TODO: REMOVE WHEN UNICODE VLENS IMPLEMENTED
        # Hack to support Unicode values (scalars only)
        #if isinstance(data, unicode):
        #    unicode_hack = True
        #    data = data.encode('utf8')
        #else:
        #    unicode_hack = False

        if data is not None:
            data = numpy.asarray(data, order='C', dtype=dtype)
            if shape is None:
                shape = data.shape
            elif numpy.product(shape) != numpy.product(data.shape):
                raise ValueError("Shape of new attribute conflicts with shape of data")
                
            if dtype is None:
                dtype = data.dtype

        if dtype is None:
            dtype = numpy.dtype('f')
        if shape is None:
            raise ValueError('At least one of "shape" or "data" must be given')

        data = data.reshape(shape)

        space = h5s.create_simple(shape)
        htype = h5t.py_create(dtype, logical=True)

        # TODO: REMOVE WHEN UNICODE VLENS IMPLEMENTED
        #if unicode_hack:
        #    htype.set_cset(h5t.CSET_UTF8)

        if name in self:
            h5a.delete(self._id, self._e(name))

        attr = h5a.create(self._id, self._e(name), htype, space)
        if data is not None:
            attr.write(data)

    def modify(self, name, value):
        """ Change the value of an attribute while preserving its type.

        Differs from __setitem__ in that if the attribute already exists, its
        type is preserved.  This can be very useful for interacting with
        externally generated files.

        If the attribute doesn't exist, it will be automatically created.
        """
        if not name in self:
            self[name] = value
        else:
            value = numpy.asarray(value, order='C')

            attr = h5a.open(self._id, self._e(name))

            # Allow the case of () <-> (1,)
            if (value.shape != attr.shape) and not \
               (numpy.product(value.shape)==1 and numpy.product(attr.shape)==1):
                raise TypeError("Shape of data is incompatible with existing attribute")
            attr.write(value)

    def __len__(self):
        """ Number of attributes attached to the object. """
        # I expect we will not have more than 2**32 attributes
        return h5a.get_num_attrs(self._id)

    def __iter__(self):
        """ Iterate over the names of attributes. """
        attrlist = []
        def iter_cb(name, *args):
            attrlist.append(self._d(name))
        h5a.iterate(self._id, iter_cb)

        for name in attrlist:
            yield name

    def __contains__(self, name):
        """ Determine if an attribute exists, by name. """
        return h5a.exists(self._id, self._e(name))

    def __repr__(self):
        if not self._id:
            return "<Attributes of closed HDF5 object>"
        return "<Attributes of HDF5 object at %s>" % id(self._id)
