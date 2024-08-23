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
    """ Class for exercise 'visit' and 'visititems' methods """

    def __init__(self):
        self._names = []

    def __call__(self, name, obj=None):
        self._names.append(name)

    @property
    def names(self):
        return self._names

class TestLexicographic(TestCase):
    """ Test ascending lexicographic order traversal of the 'visit*' methods.

        This semantics is set by the following default args in
        h5py.h5o.visit(..., idx_type=H5_INDEX_NAME, order=H5_ITER_INC, ...)
        h5py.h5l.visit(..., idx_type=H5_INDEX_NAME, order=H5_ITER_INC, ...)
    """

    import operator
    split_parts = operator.methodcaller('split', '/')

    def setUp(self):
        """ Populate example hdf5 file, with track_order=True """

        self.f = File(self.mktemp(), 'w-', track_order=True)
        self.f.create_dataset('b', (10,))

        grp = self.f.create_group('B', track_order=True)
        grp.create_dataset('b', (10,))
        grp.create_dataset('a', (10,))

        grp = self.f.create_group('z', track_order=True)
        grp.create_dataset('b', (10,))
        grp.create_dataset('a', (10,))

        self.f.create_dataset('a', (10,))
        # note that 'z-' < 'z/...' but traversal order is ['z', 'z/...', 'z-']
        self.f.create_dataset('z-', (10,))

        # create some links
        self.f['A/x'] = self.f['B/b']
        self.f['y'] = self.f['z/a']
        self.f['A$'] = self.f['y']
        self.f['A/B/C'] = self.f['A']
        self.f['A/a'] = self.f['A']

        # create vistor
        self.v = Visitor()

    def test_nontrivial_sort_visit(self):
        """check that test example is not trivially sorted"""
        self.f.visit(self.v)
        assert self.v.names != sorted(self.v.names)

    def test_visit(self):
        """check that File.visit iterates in lexicographic order"""
        self.f.visit(self.v)
        assert self.v.names == sorted(self.v.names, key=self.split_parts)

    def test_visit_links(self):
        """check that File.visit_links iterates in lexicographic order"""
        self.f.visit_links(self.v)
        assert self.v.names == sorted(self.v.names, key=self.split_parts)

    def test_visititems(self):
        """check that File.visititems iterates in lexicographic order"""
        self.f.visititems(self.v)
        assert self.v.names == sorted(self.v.names, key=self.split_parts)

    def test_visititems_links(self):
        """check that File.visititems_links iterates in lexicographic order"""
        self.f.visititems_links(self.v)
        assert self.v.names == sorted(self.v.names, key=self.split_parts)

    def test_visit_group(self):
        """check that Group.visit iterates in lexicographic order"""
        self.f['A'].visit(self.v)
        assert self.v.names == sorted(self.v.names, key=self.split_parts)
