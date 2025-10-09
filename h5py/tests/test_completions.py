import h5py

from .common import TestCase, make_name


class TestCompletions(TestCase):

    def test_root_group_completions(self):
        # Test completions on top-level file.
        with h5py.File(self.mktemp(), 'w') as f:
            f.create_group('h')
            f.create_group('g')
            f.create_group('data2')
            f.create_dataset('data', [1, 2, 3])

            # Test that order is alphabetical, and that there is no
            # internal ordering between groups and datasets.
            self.assertEqual(
                f._ipython_key_completions_(),
                ['data', 'data2', 'g', 'h'],
            )

    def test_subgroup_completions(self):
        g = self.f.create_group(make_name())
        g.create_dataset('g_data2', [4, 5, 6])
        g.create_dataset('g_data1', [1, 2, 3])
        g.create_dataset('g_data3', [7, 8, 9])
        self.assertEqual(
            g._ipython_key_completions_(),
            ['g_data1', 'g_data2', 'g_data3'],
        )

    def test_attrs_completions(self):
        attrs = self.f.create_group(make_name()).attrs

        # Write out of alphabetical order to test that completions come back in
        # alphabetical order, as opposed to, say, insertion order.
        attrs['b'] = 1
        attrs['a'] = 2
        attrs['c'] = 3
        self.assertEqual(
            attrs._ipython_key_completions_(),
            ['a', 'b', 'c']
        )
