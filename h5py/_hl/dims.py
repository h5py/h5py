import numpy

from h5py import h5ds
from . import base
from .dataset import Dataset, readtime_dtype


class Dimension(object):

    @property
    def label(self):
        return h5ds.get_label(self._id, self._dimension)
    @label.setter
    def label(self, val):
        h5ds.set_label(self._id, self._dimension, val)

    def __init__(self, id, dimension):
        self._id = id
        self._dimension = dimension

    def __getitem__(self, item):
        if isinstance(item, int):
            scales = []
            def f(dsid):
                scales.append(Dataset(dsid))
            h5ds.iterate(self._id, self._dimension, f, 0)
            return scales[item]
        else:
            def f(dsid):
                if h5ds.get_scale_name(dsid) == item:
                    return Dataset(dsid)
            res = h5ds.iterate(self._id, self._dimension, f, 0)
            if res is None:
                raise KeyError('%s not found' % item)
            return res

    def attach_scale(self, dset):
        h5ds.attach_scale(self._id, dset.id, self._dimension)

    def keys(self):
        scales = []
        def f(dsid):
            scales.append(dsid)
        h5ds.iterate(self._id, self._dimension, f, 0)
        return [h5ds.get_scale_name(scale) for scale in scales]


class DimensionManager(base.DictCompat, base.CommonStateObject):

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

    def __getitem__(self, index):
        """ Return a Dimension object
        """
        return Dimension(self._id, index)

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

    def __repr__(self):
        if not self._id:
            return "<Attributes of closed HDF5 object>"
        return "<Attributes of HDF5 object at %s>" % id(self._id)

    def create_scale(self, dset, name=''):
        h5ds.set_scale(dset.id, name)
