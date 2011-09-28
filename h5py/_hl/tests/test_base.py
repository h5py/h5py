
"""
    Common high-level operations test

    Tests features common to all high-level objects, like the .name property.
"""

from h5py import File
from .common import ut, TestCase

class BaseTest(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()

class TestName(BaseTest):

    """
        Feature: .name attribute returns the object name
    """

    def test_anonymous(self):
        """ Anomymous objects have name None """
        grp = self.f.create_group(None)
        self.assertIs(grp.name, None)
