import numpy as np
import h5py

from multiprocessing import JoinableQueue, Process

from .common import ut, TestCase


@ut.skipUnless(h5py.version.hdf5_version_tuple < (1, 9, 178), 'SWMR is available. Skipping backwards compatible tests')
class TestSwmrNotAvailable(TestCase):
    """ Test backwards compatibility behaviour when using SWMR functions with
    an older version of HDF5 which does not have this feature available.
    Skip this test if SWMR features *are* available in the HDF5 library.
    """

    def setUp(self):
        TestCase.setUp(self)
        self.data = np.arange(13).astype('f')
        self.group = self.f.create_group('group')

    def test_open_swmr_raises(self):
        fname = self.f.filename
        self.f.close()

        with self.assertRaises(ValueError):
            self.f = h5py.File(fname, 'r', swmr=True)

    def test_refresh_raises(self):
        """ If the SWMR feature is not available then Dataset.refresh() should throw an AttributeError
        """
        with self.assertRaises(AttributeError):
            self.group.refresh()

    def test_flush_raises(self):
        """ If the SWMR feature is not available the Dataset.flush() should
        throw an AttributeError
        """
        with self.assertRaises(AttributeError):
            self.group.flush()

    def test_swmr_mode_raises(self):
        with self.assertRaises(AttributeError):
            self.f.swmr_mode

@ut.skipUnless(h5py.version.hdf5_version_tuple >= (1, 9, 178), 'SWMR requires HDF5 >= 1.9.178')
class TestDatasetSwmrRead(TestCase):
    """ Testing SWMR functions when reading a dataset.
    Skip this test if the HDF5 library does not have the SWMR features.
    """

    def setUp(self):
        TestCase.setUp(self)
        self.data = np.arange(13).astype('f')
        self.group = self.f.create_group('group')
        self.dset = self.group.create_dataset('data', chunks=(13,), maxshape=(None,), data=self.data)

        self.attribute_value = 'test'
        self.group.attrs.create('attribute', self.attribute_value)
        fname = self.f.filename

        self.f.close()

        self.f = h5py.File(fname, 'r', swmr=True)
        self.group = self.f['group']
        self.dset = self.group['data']
        self.attr = self.group.attrs['attribute']

    def test_initial_swmr_mode_on(self):
        """ Verify that the file is initially in SWMR mode"""
        self.assertTrue(self.f.swmr_mode)

    def test_read_data(self):
        self.assertEqual(self.attr, self.attribute_value)
        self.assertArrayEqual(self.dset, self.data)
        pass

    def test_refresh(self):
        self.group.refresh()

    def test_force_swmr_mode_on_raises(self):
        """ Verify when reading a file cannot be forcibly switched to swmr mode.
        When reading with SWMR the file must be opened with swmr=True."""
        with self.assertRaises(Exception):
            self.f.swmr_mode = True
        self.assertTrue(self.f.swmr_mode)

    def test_force_swmr_mode_off_raises(self):
        """ Switching SWMR write mode off is only possible by closing the file.
        Attempts to forcibly switch off the SWMR mode should raise a ValueError.
        """
        with self.assertRaises(ValueError):
            self.f.swmr_mode = False
        self.assertTrue(self.f.swmr_mode)


def writer_loop(queue: JoinableQueue):
    stop = False
    file = None

    while not stop:
        item = queue.get()

        assert 'action' in item
        action = item['action']

        parameters = {}
        if 'parameters' in item:
            parameters = item['parameters']

        if action == 'create_file':
            assert 'fname' in parameters

            file = h5py.File(parameters['fname'], 'w', libver='latest')
            file.swmr_mode = True
        elif action == 'close_file':
            file.close()
        elif action == 'stop':
            stop = True
        else:
            assert False

        if action == 'stop':
            stop = True

        queue.task_done()


@ut.skipUnless(h5py.version.hdf5_version_tuple >= (1, 9, 178), 'SWMR requires HDF5 >= 1.9.178')
class TestDatasetSwmrWriteRead(TestCase):
    """ Testing SWMR functions when reading a dataset.
    Skip this test if the HDF5 library does not have the SWMR features.
    """

    def setUp(self):
        """ First setup a file with a small chunked and empty dataset.
        No data written yet.
        """

        # Note that when creating the file, the swmr=True is not required for
        # write, but libver='latest' is required.

        fname = self.mktemp()

        writer_queue = JoinableQueue()
        writer_process = Process(target=writer_loop, args=(writer_queue,), daemon=True)
        writer_process.start()

        parameters = {'fname': fname}
        writer_queue.put({'action': 'create_file', 'parameters': parameters})
        writer_queue.join()

        writer_queue.put({'action': 'close_file'})
        writer_queue.join()

        writer_queue.put({'action': "stop"})
        writer_queue.join()

        writer_process.join()

        self.f_write = h5py.File(fname, 'w', libver='latest')
        self.f_write.swmr_mode = True

        self.f_read = h5py.File(fname, 'r', swmr=True)

        self.group_write = self.f_write.create_group('group')

        self.data = np.arange(13).astype('f')

        self.dset_write = self.group_write.create_dataset('data', chunks=(13,), maxshape=(None,), data=self.data)

        self.attribute_value = 'test'
        self.group_write.attrs.create('attribute', self.attribute_value)
        self.attr_write = self.group_write.attrs['attribute']

        pass

    def tearDown(self):
        self.f_read.close()
        self.f_write.close()

    def test_update_attribute(self):
        """ Extend and flush a SWMR dataset
        """

        # check file writer data and attr
        self.assertArrayEqual(self.dset_write, self.data)
        self.assertEqual(self.attr_write, self.attribute_value)

        # check that file reader has knowledge of this
        self.group_read = self.f_read['group']
        self.assertTrue('attribute' in self.group_read.attrs)

        self.attr_read = self.group_read.attrs['attribute']
        self.assertEqual(self.attr_read, self.attribute_value)

        self.new_attribute_value = 'test2'

        self.group_write.attrs['attribute'] = self.new_attribute_value
        self.attr_write = self.group_write.attrs['attribute']
        self.assertEqual(self.attr_write, self.new_attribute_value)

        self.attr_read = self.group_read.attrs['attribute']
        self.assertEqual(self.attr_read, self.new_attribute_value)

        self.group_write.flush()

        self.attr_read = self.group_read.attrs['attribute']
        self.assertEqual(self.attr_read, self.new_attribute_value)

        self.group_read.refresh()

        pass
