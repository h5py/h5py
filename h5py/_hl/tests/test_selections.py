
"""
    Tests for the (internal) selections module
"""

import numpy as np
import h5py
import h5py._hl.selections2 as sel

from common import TestCase, ut

class TestTypeGeneration(TestCase):

    """
        Internal feature: Determine output types from dataset dtype and fields.
    """

    def test_simple(self):
        """ Non-compound types are handled appropriately """
        dt = np.dtype('i')
        out, format = sel.read_dtypes(dt, ())
        self.assertEqual(out, format)
        self.assertEqual(out, np.dtype('i'))

    def test_simple_fieldexc(self):
        """ Field names for non-field types raises ValueError """
        dt = np.dtype('i')
        with self.assertRaises(ValueError):
            out, format = sel.read_dtypes(dt, ('a',))

    def test_compound_simple(self):
        """ Compound types with elemental subtypes """
        dt = np.dtype( [('a','i'), ('b','f'), ('c','|S10')] )

        # Implicit selection of all fields -> all fields
        out, format = sel.read_dtypes(dt, ())
        self.assertEqual(out, format)
        self.assertEqual(out, dt)
       
        # Explicit selection of fields -> requested fields
        out, format = sel.read_dtypes(dt, ('a','b'))
        self.assertEqual(out, format)
        self.assertEqual(out, np.dtype( [('a','i'), ('b','f')] ))

        # Explicit selection of exactly one field -> no fields
        out, format = sel.read_dtypes(dt, ('a',))
        self.assertEqual(out, np.dtype('i'))
        self.assertEqual(format, np.dtype( [('a','i')] ))

    def test_objects(self):
        """ Metadata is stripped from object types """
        dt = h5py.special_dtype(ref=h5py.Reference)
        
        out, format = sel.read_dtypes(dt, ())
        self.assertEqual(out, format)
        self.assertEqual(format, dt)
        self.assertTrue(out.fields is None)
        self.assertTrue(format.fields is not None)

    def test_compound_objects(self):
        """ Metadata is stripped from output in compound types"""
        reftype = h5py.special_dtype(ref=h5py.Reference)

        dt = np.dtype( [('a','i'), ('b',reftype), ('c','|S10')] )

        out, format = sel.read_dtypes(dt, ())
        self.assertEqual(out, format)
        self.assertEqual(format, dt)
        self.assertTrue(all(x[0].fields is None for x in out.fields.values() if x[0].kind == 'O'))


class TestScalarSliceRules(TestCase):

    """
        Internal feature: selections rules for scalar datasets
    """

    def setUp(self):
        self.f = h5py.File(self.mktemp(), 'w')
        self.dsid = self.f.create_dataset('x', ()).id

    def tearDown(self):
        if self.f:
            self.f.close()

    def test_args(self):
        """ Permissible arguments for scalar slicing """
        shape, selection = sel.read_selections_scalar(self.dsid, ())
        self.assertEqual(shape, None)
        self.assertEqual(selection.get_select_npoints(), 1)

        shape, selection = sel.read_selections_scalar(self.dsid, (Ellipsis,))
        self.assertEqual(shape, ())
        self.assertEqual(selection.get_select_npoints(), 1)
        
        with self.assertRaises(ValueError):
            shape, selection = sel.read_selections_scalar(self.dsid, (1,))











