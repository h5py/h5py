import pytest

from .common import TestCase
from h5py import File


class SampleException(Exception):
    pass

def throwing(name, obj):
    print(name, obj)
    raise SampleException("throwing exception")

class TestVisit(TestCase):
    def test_visit(self):
        fname = self.mktemp()
        fid = File(fname, 'w')
        fid.create_dataset('foo', (100,), dtype='uint8')
        with pytest.raises(SampleException, match='throwing exception'):
            fid.visititems(throwing)
        fid.close()

class Visitor:
    """class for excersize 'visit' and 'visititems'"""

    def __init__(self):
        self._names = []

    def __call__(self, name, obj=None):
        self._names.append(name)

    @property
    def names(self):
        return self._names

class TestLexicographic(TestCase):
    def setUp(self):
        """populate example hdf5 file, with track_order=True"""

        self.f = File(self.mktemp(), 'w-', track_order=True)
        self.f.create_dataset('b', (10,))

        grp = self.f.create_group('B', track_order=True)
        grp.create_dataset('b', (10,))
        grp.create_dataset('a', (10,))
        # check that example is non trivial: key order is not sorted
        assert list(grp) != sorted(grp)

        grp = self.f.create_group('z', track_order=True)
        grp.create_dataset('b', (10,))
        grp.create_dataset('a', (10,))
        assert list(grp) != sorted(grp)

        self.f.create_dataset('a', (10,))
        assert list(self.f) != sorted(self.f)

        # create some links
        self.f['A/x'] = self.f['B/b']
        self.f['y'] = self.f['z/a']

    def test_visit(self):
        """check that File.visit iterates in lexicographic order"""
        v = Visitor()
        self.f.visit(v)
        assert v.names == sorted(v.names)

    def test_visit_links(self):
        """check that File.visit_links iterates in lexicographic order"""
        v = Visitor()
        self.f.visit_links(v)
        assert v.names == sorted(v.names)

    def test_visititems(self):
        """check that File.visititems iterates in lexicographic order"""
        v = Visitor()
        self.f.visititems(v)
        assert v.names == sorted(v.names)

    def test_visititems_links(self):
        """check that File.visititems_links iterates in lexicographic order"""
        v = Visitor()
        self.f.visititems_links(v)
        assert v.names == sorted(v.names)
