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
from h5py._hl.files import make_fapl
import pytest


pytestmark = [
    pytest.mark.skipif(
        not h5py.h5.get_config().ros3,
        reason="ros3 driver not available")
]


@pytest.mark.network
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {},
            id="HDF5-v1",
            marks=pytest.mark.skipif(
                h5py.version.hdf5_version_tuple >= (2, 0, 0),
                reason="Requires HDF5 < 2.0",
            ),
        ),
        pytest.param(
            {"aws_region": b"us-east-2"},
            id="HDF5-v2",
            marks=pytest.mark.skipif(
                h5py.version.hdf5_version_tuple < (2, 0, 0),
                reason="Requires HDF5 >= 2.0",
            ),
        ),
    ],
)
def test_ros3(kwargs):
    """ ROS3 driver and options """

    with h5py.File("https://dandiarchive.s3.amazonaws.com/ros3test.hdf5", 'r',
                   driver='ros3', **kwargs) as f:
        assert f
        assert 'mydataset' in f.keys()
        assert f["mydataset"].shape == (100,)


@pytest.mark.parametrize(
    "exc,match_exc",
    [
        pytest.param(
            ValueError,
            [
                "AWS region required for s3:// location",
                r"^foo://wrong/scheme: S3 location must begin with",
            ],
            id="HDF5-v1",
            marks=pytest.mark.skipif(
                h5py.version.hdf5_version_tuple >= (2, 0, 0),
                reason="Requires HDF5 < 2.0",
            ),
        ),
        pytest.param(
            OSError,
            [None, "can't parse object key from path"],
            id="HDF5-v2",
            marks=pytest.mark.skipif(
                h5py.version.hdf5_version_tuple < (2, 0, 0),
                reason="Requires HDF5 >= 2.0",
            ),
        ),
    ],
)
def test_ros3_s3_fails(exc, match_exc):
    """ROS3 exceptions for s3:// location"""
    with pytest.raises(exc, match=match_exc[0]):
        h5py.File('s3://fakebucket/fakekey', 'r', driver='ros3')

    with pytest.raises(exc, match=match_exc[1]):
        h5py.File('foo://wrong/scheme', 'r', driver='ros3')


@pytest.mark.network
def test_ros3_s3uri():
    """Use S3 URI with ROS3 driver"""
    with h5py.File('s3://dandiarchive/ros3test.hdf5', 'r', driver='ros3',
                   aws_region=b'us-east-2') as f:
        assert f
        assert 'mydataset' in f.keys()
        assert f["mydataset"].shape == (100,)


@pytest.mark.skipif(h5py.version.hdf5_version_tuple < (1, 14, 2),
                    reason='AWS S3 access token support in HDF5 >= 1.14.2')
def test_ros3_temp_token():
    """Set and get S3 access token"""
    token = b'#0123FakeToken4567/8/9'
    fapl = make_fapl('ros3', libver=None, rdcc_nslots=None, rdcc_nbytes=None,
                     rdcc_w0=None, locking=None, page_buf_size=None, min_meta_keep=None,
                     min_raw_keep=None, alignment_threshold=1, alignment_interval=1,
                     meta_block_size=None, session_token=token)
    assert token, fapl.get_fapl_ros3_token()
