from .common import TestCase


class TestCompletions(TestCase):

    def test_group_completions(self):
        # Test completions on top-level file.
        g = self.f.create_group('g')
        self.f.create_group('h')
        self.f.create_dataset('data', [1, 2, 3])
        self.assertEqual(
            self.f._ipython_key_completions_(),
            [u'data', u'g', u'h'],
        )

        self.f.create_group('data2', [1, 2, 3])
        self.assertEqual(
            self.f._ipython_key_completions_(),
            [u'data', u'data2', u'g', u'h'],
        )

        # Test on subgroup.
        g.create_dataset('g_data1', [1, 2, 3])
        g.create_dataset('g_data2', [4, 5, 6])
        self.assertEqual(
            g._ipython_key_completions_(),
            [u'g_data1', u'g_data2'],
        )

        g.create_dataset('g_data3', [7, 8, 9])
        self.assertEqual(
            g._ipython_key_completions_(),
            [u'g_data1', u'g_data2', u'g_data3'],
        )

    def test_attrs_completions(self):
        attrs = self.f.attrs

        # Write out of alphabetical order to test that completions come back in
        # alphabetical order, as opposed to, say, insertion order.
        attrs['b'] = 1
        attrs['a'] = 2
        self.assertEqual(
            attrs._ipython_key_completions_(),
            [u'a', u'b']
        )

        attrs['c'] = 3
        self.assertEqual(
            attrs._ipython_key_completions_(),
            [u'a', u'b', u'c']
        )
