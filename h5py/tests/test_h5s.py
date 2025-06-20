import pytest

from h5py import h5s, version
from h5py._selector import Selector

class Helper:
    def __init__(self, shape: tuple):
        self.shape = shape

    def __getitem__(self, item) -> h5s.SpaceID:
        if not isinstance(item, tuple):
            item = (item,)
        space = h5s.create_simple(self.shape)
        sel = Selector(space)
        sel.make_selection(item)
        return space


def test_same_shape():
    s1 = Helper((5, 6))[:3, :4]
    s2 = Helper((5, 6))[2:, 2:]
    assert s1.select_shape_same(s2)

    s3 = Helper((5, 6))[:4, :3]
    assert not s1.select_shape_same(s3)

def test_select_copy():
    s1 = h5s.create_simple((50,))
    s1.select_hyperslab((5,), (1,), block=(5,), op=h5s.SELECT_SET)
    s2 = h5s.create_simple((50,))
    s2.select_copy(s1)
    assert (s1.get_select_hyper_blocklist() == s2.get_select_hyper_blocklist()).all()

def test_combine_select():
    # combine_select should create a new space without modifying the original
    s1 = h5s.create_simple((50,))
    s1.select_hyperslab((5,), (1,), block=(5,), op=h5s.SELECT_SET)

    s2 = h5s.create_simple((50,))
    s2.select_hyperslab((20,), (1,), block=(2,), op=h5s.SELECT_SET)

    s3 = s1.combine_select(s2, op=h5s.SELECT_OR)
    s2.select_hyperslab((5,), (1,), block=(5,), op=h5s.SELECT_OR)
    assert (s3.get_select_hyper_blocklist() == s2.get_select_hyper_blocklist()).all()
    assert not (s3.get_select_hyper_blocklist() == s1.get_select_hyper_blocklist()).all()

def test_modify_select():
    # combine_select should create a new space without modifying the original
    s1 = h5s.create_simple((50,))
    s1.select_hyperslab((5,), (1,), block=(5,), op=h5s.SELECT_SET)

    s2 = h5s.create_simple((50,))
    s2.select_hyperslab((20,), (1,), block=(2,), op=h5s.SELECT_SET)

    s1.modify_select(s2, op=h5s.SELECT_OR)
    s2.select_hyperslab((5,), (1,), block=(5,), op=h5s.SELECT_OR)
    assert (s1.get_select_hyper_blocklist() == s2.get_select_hyper_blocklist()).all()
