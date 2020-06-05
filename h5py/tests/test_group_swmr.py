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
    group = None

    while not stop:
        item = queue.get()

        assert 'action' in item
        action = item['action']

        # get parameter if any
        parameters = {}
        if 'parameters' in item:
            parameters = item['parameters']

        if action == 'create_file':
            assert 'fname' in parameters

            fname = parameters['fname']

            file = h5py.File(fname, 'w', libver='latest')

            group = file.create_group('group')

            pass

        elif action == 'set_swmr_mode':
            file.swmr_mode = True

        elif action == 'create_dataset':
            assert 'name' in parameters
            assert 'value' in parameters

            name = parameters['name']
            data = parameters['value']

            group.create_dataset(name, maxshape=data.shape, data=data)

        elif action == 'update_dataset':
            assert 'name' in parameters
            assert 'value' in parameters

            name = parameters['name']
            data = parameters['value']

            group[name][:] = data

        elif action == 'create_attribute':
            assert 'name' in parameters
            assert 'value' in parameters

            name = parameters['name']
            value = parameters['value']

            if isinstance(value, str):
                value = np.string_(value)

            group.attrs.create(name, value)

        elif action == 'update_attribute':
            assert 'name' in parameters
            assert 'value' in parameters

            name = parameters['name']
            value = parameters['value']

            if isinstance(value, str):
                value = np.string_(value)

            group.attrs[name] = value

        elif action == 'flush_file':
            file.flush()

        elif action == 'flush_group':
            group.flush()

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

        self.fname = self.mktemp()

        self.writer_queue = JoinableQueue()
        self.writer_process = Process(target=writer_loop, args=(self.writer_queue,), daemon=True)
        self.writer_process.start()

    def test_create_open_read_update_file(self):
        """ Update and read dataset and
         an attribute in group with SWMR mode
        """

        self.data = np.arange(13).astype('f')
        self.new_data = np.arange(13).astype('f') + 2

        writer_queue = self.writer_queue

        parameters = {'fname': self.fname}
        writer_queue.put({'action': 'create_file', 'parameters': parameters})
        writer_queue.join()

        parameters = {'name': 'data', 'value': self.data}
        writer_queue.put({'action': 'create_dataset', 'parameters': parameters})
        writer_queue.join()

        # create attributes to test
        attributes = [
            {'name': 'attr_bool', 'value': False, 'new_value': True},
            {'name': 'attr_int', 'value': 1, 'new_value': 2},
            {'name': 'attr_float', 'value': 1.4, 'new_value': 3.2},
            {'name': 'attr_string', 'value': 'test', 'new_value': 'essai'},
        ]

        for attribute in attributes:
            attribute_name = attribute['name']
            attribute_value = attribute['value']

            parameters = {'name': attribute_name, 'value': attribute_value}
            writer_queue.put({'action': 'create_attribute', 'parameters': parameters})
            writer_queue.join()

        # try opening the file in swmr

        file = None
        with self.assertRaises(OSError):
            file = h5py.File(self.fname, 'r', libver='latest', swmr=True)

        writer_queue.put({'action': 'set_swmr_mode'})
        writer_queue.join()

        # open file and check group
        file = h5py.File(self.fname, 'r', libver='latest', swmr=True)
        self.assertIn('group', file)

        # check attributes

        group = file['group']

        for attribute in attributes:
            attribute_name = attribute['name']
            attribute_value = attribute['value']
            attribute_new_value = attribute['new_value']

            self.assertIn(attribute_name, group.attrs)

            read_value = group.attrs[attribute_name]
            if isinstance(attribute_value, str):
                read_value = read_value.decode()
            self.assertEqual(read_value, attribute_value)

            parameters = {'name': attribute_name, 'value': attribute_new_value}
            writer_queue.put({'action': 'update_attribute', 'parameters': parameters})
            writer_queue.join()

            read_value = group.attrs[attribute_name]
            if isinstance(attribute_value, str):
                read_value = read_value.decode()
            self.assertEqual(read_value, attribute_value)

            writer_queue.put({'action': 'flush_group'})
            writer_queue.join()

            # check that read group attribute has not changed
            read_value = group.attrs[attribute_name]
            if isinstance(attribute_value, str):
                read_value = read_value.decode()
            self.assertEqual(read_value, attribute_value)

            group.refresh()

            # check that read group attribute has changed
            read_value = group.attrs[attribute_name]
            if isinstance(attribute_value, str):
                read_value = read_value.decode()
            self.assertEqual(read_value, attribute_new_value)

        # check that dataset has been recorder
        data = group['data']
        self.assertArrayEqual(data[:], self.data)

        # update data
        parameters = {'name': 'data', 'value': self.new_data}
        writer_queue.put({'action': 'update_dataset', 'parameters': parameters})
        writer_queue.join()

        # check that data has not been updated
        data = group['data']
        self.assertArrayEqual(data[:], self.data)

        # flush group
        writer_queue.put({'action': 'flush_group'})
        writer_queue.join()

        # check that data has not been updated
        data = group['data']
        self.assertArrayEqual(data[:], self.data)

        # refresh group, this won't update dataset
        group.refresh()

        # check that data has not been updated
        data = group['data']
        self.assertArrayEqual(data[:], self.data)

        # refresh dataset, this will update data
        data.refresh()

        # check that data has been updated
        self.assertArrayEqual(data[:], self.new_data)

        writer_queue.put({'action': 'close_file'})
        writer_queue.join()

        file.close()

        pass

    def tearDown(self):

        self.writer_queue.put({'action': "stop"})
        self.writer_queue.join()

        self.writer_process.join()
