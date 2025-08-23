
from h5py import h5s
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
