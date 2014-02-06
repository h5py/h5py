# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

import numpy
import collections

import h5py
from h5py import h5s, h5t, h5a
from . import base
from .base import phil, with_phil
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

    @with_phil
    def __getitem__(self, name):
        """ Read the value of an attribute.
        """
        attr = h5a.open(self._id, self._e(name))

        if attr.get_space().get_simple_extent_type() == h5s.NULL:
            raise IOError("Empty attributes cannot be read")

        tid = attr.get_type()

        rtdt = readtime_dtype(attr.dtype, [])

        arr = numpy.ndarray(attr.shape, dtype=rtdt, order='C')
        attr.read(arr)

        if len(arr.shape) == 0:
            return arr[()]
        return arr

    @with_phil
    def __setitem__(self, name, value):
        """ Set a new attribute, overwriting any existing attribute.

        The type and shape of the attribute are determined from the data.  To
        use a specific type or shape, or to preserve the type of an attribute,
        use the methods create() and modify().
        """
        self.create(name, data=value, dtype=base.guess_dtype(value))

    @with_phil
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

        with phil:
            if data is not None:
                data = numpy.asarray(data, order='C', dtype=dtype)
                if shape is None:
                    shape = data.shape
                elif numpy.product(shape) != numpy.product(data.shape):
                    raise ValueError("Shape of new attribute conflicts with shape of data")

                if dtype is None:
                    dtype = data.dtype

            if isinstance(dtype, h5py.Datatype):
                htype = dtype.id
                dtype = htype.dtype
            else:
                if dtype is None:
                    dtype = numpy.dtype('f')
                htype = h5t.py_create(dtype, logical=True)

            if shape is None:
                raise ValueError('At least one of "shape" or "data" must be given')

            data = data.reshape(shape)

            space = h5s.create_simple(shape)

            if name in self:
                h5a.delete(self._id, self._e(name))

            attr = h5a.create(self._id, self._e(name), htype, space)

            if data is not None:
                try:
                    attr.write(data)
                except:
                    attr._close()
                    h5a.delete(self._id, self._e(name))
                    raise

    def modify(self, name, value):
        """ Change the value of an attribute while preserving its type.

        Differs from __setitem__ in that if the attribute already exists, its
        type is preserved.  This can be very useful for interacting with
        externally generated files.

        If the attribute doesn't exist, it will be automatically created.
        """
        with phil:
            if not name in self:
                self[name] = value
            else:
                value = numpy.asarray(value, order='C')

                attr = h5a.open(self._id, self._e(name))

                if attr.get_space().get_simple_extent_type() == h5s.NULL:
                    raise IOError("Empty attributes can't be modified")

                # Allow the case of () <-> (1,)
                if (value.shape != attr.shape) and not \
                   (numpy.product(value.shape) == 1 and numpy.product(attr.shape) == 1):
                    raise TypeError("Shape of data is incompatible with existing attribute")
                attr.write(value)

    @with_phil
    def __len__(self):
        """ Number of attributes attached to the object. """
        # I expect we will not have more than 2**32 attributes
        return h5a.get_num_attrs(self._id)

    def __iter__(self):
        """ Iterate over the names of attributes. """
        with phil:
            attrlist = []

            def iter_cb(name, *args):
                attrlist.append(self._d(name))
            h5a.iterate(self._id, iter_cb)

        for name in attrlist:
            yield name

    @with_phil
    def __contains__(self, name):
        """ Determine if an attribute exists, by name. """
        return h5a.exists(self._id, self._e(name))

    @with_phil
    def __repr__(self):
        if not self._id:
            return "<Attributes of closed HDF5 object>"
        return "<Attributes of HDF5 object at %s>" % id(self._id)

collections.MutableMapping.register(AttributeManager)
