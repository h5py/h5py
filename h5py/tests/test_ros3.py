# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Read-only S3 virtual file driver (VFD) test module.
"""

import h5py
import pytest

pytestmark = [
    pytest.mark.skipif(
        h5py.version.hdf5_version_tuple < (1, 10, 6) or not h5py.h5.get_config().ros3,
        reason="ros3 driver not available"),
    pytest.mark.nonetwork
]


def test_ros3():
    """ ROS3 driver and options """

    with h5py.File("https://dandiarchive.s3.amazonaws.com/ros3test.hdf5", 'r',
                   driver='ros3') as f:
        assert f
        assert 'mydataset' in f.keys()
        assert f["mydataset"].shape == (100,)


def test_ros3_s3_fails():
    """ROS3 exceptions for s3:// location"""
    with pytest.raises(ValueError, match='AWS region required for s3:// location'):
        h5py.File('s3://fakebucket/fakekey', 'r', driver='ros3')

    with pytest.raises(ValueError, match=r'^foo://wrong/scheme: S3 location must begin with'):
        h5py.File('foo://wrong/scheme', 'r', driver='ros3')


def test_ros3_s3uri():
    """Use S3 URI with ROS3 driver"""
    with h5py.File('s3://dandiarchive/ros3test.hdf5', 'r', driver='ros3',
                   aws_region=b'us-east-2') as f:
        assert f
        assert 'mydataset' in f.keys()
        assert f["mydataset"].shape == (100,)
