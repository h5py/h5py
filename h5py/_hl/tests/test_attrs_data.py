import numpy as np

from .common import TestCase, ut

from h5py.highlevel import File

class BaseAttrs(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')
 
    def tearDown(self):
        if self.f:
            self.f.close()

class TestScalar(BaseAttrs):

    """
        Feature: Scalar types map correctly to array scalars
    """

    @ut.expectedFailure
    def test_compound(self):
        """ Compound scalars are read as numpy.void """
        dt = np.dtype([('a','b'),('i','f')])
        data = np.array((1,4.2), dtype=dt)
        self.f.attrs['x'] = data
        out = self.f.attrs['x']
        self.assertIsInstance(out, np.void)
        self.assertEqual(out, data)
        self.assertEqual(out['b'], data['b'])


