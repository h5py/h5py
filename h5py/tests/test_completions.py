from .common import TestCase, name


class TestCompletions(TestCase):

    def test_group_completions(self):
        # Test completions on top-level file.
        g = self.f.create_group(name("g"))
        self.f.create_group(name("h"))
        self.f.create_dataset(name("data"), [1, 2, 3])
        self.assertEqual(
            self.f._ipython_key_completions_(),
            [name("data"), name("g"), name("h")],
        )

        self.f.create_group(name('data2'), [1, 2, 3])
        self.assertEqual(
            self.f._ipython_key_completions_(),
            [name("data"), name("data2"), name("g"), name("h")],
        )

        # Test on subgroup.
        g.create_dataset('g_data1', [1, 2, 3])
        g.create_dataset('g_data2', [4, 5, 6])
        self.assertEqual(
            g._ipython_key_completions_(),
            ['g_data1', 'g_data2'],
        )

        g.create_dataset('g_data3', [7, 8, 9])
        self.assertEqual(
            g._ipython_key_completions_(),
            ['g_data1', 'g_data2', 'g_data3'],
        )

    def test_attrs_completions(self):
        attrs = self.f.attrs

        # Write out of alphabetical order to test that completions come back in
        # alphabetical order, as opposed to, say, insertion order.
        attrs[name('b')] = 1
        attrs[name('a')] = 2
        self.assertEqual(
            attrs._ipython_key_completions_(),
            [name('a'), name('b')]
        )

        attrs[name('c')] = 3
        self.assertEqual(
            attrs._ipython_key_completions_(),
            [name('a'), name('b'), name('c')]
        )
