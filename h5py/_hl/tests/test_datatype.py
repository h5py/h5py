import numpy as np

from .common import ut, TestCase

from h5py import File

class BaseType(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()

class TestRepr(BaseType):

    """
        Feature: repr() works sensibly on datatype objects
    """

    def test_repr(self):
        """ repr() on datatype objects """
        self.f['foo'] = np.dtype('S10')
        dt = self.f['foo']
        self.assertIsInstance(repr(dt), basestring)
        self.f.close()
        self.assertIsInstance(repr(dt), basestring)
