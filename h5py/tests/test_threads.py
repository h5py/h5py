# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Tests the h5py.File object.
"""

import threading
import h5py


def test_thread_hdf5_silence_error_membership(tmp_path, capfd):
    """Verify the error printing is squashed in all threads.

    No console messages should be shown from membership tests
    """
    path = str((tmp_path / 'test.h5').resolve())
    def test():
        with h5py.File(path, 'w') as newfile:
            try:
                doesnt_exist = newfile['doesnt_exist'].value
            except KeyError:
                pass

    th = threading.Thread(target=test)
    th.start()
    th.join()

    captured = capfd.readouterr()
    assert captured.err == ''
    assert captured.out == ''


def test_thread_hdf5_silence_error_attr(tmp_path, capfd):
    """Verify the error printing is squashed in all threads.

    No console messages should be shown for non-existing attributes
    """
    path = str((tmp_path / 'test.h5').resolve())
    def test():
        with h5py.File(path, 'w') as newfile:
            newfile['newdata'] = [1,2,3]
            try:
                nonexistent_attr = newfile['newdata'].attrs['nonexistent_attr']
            except KeyError:
                pass

    th = threading.Thread(target=test)
    th.start()
    th.join()

    captured = capfd.readouterr()
    assert captured.err == ''
    assert captured.out == ''