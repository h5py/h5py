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
from h5py import h5p
from h5py._hl.files import make_fapl
import pytest


pytestmark = [
    pytest.mark.skipif(
        not h5py.h5.get_config().ros3,
        reason="ros3 driver not available")
]

MiB = 1024 * 1024

# ROS3 I/O block caching defaults documented in HDF5's H5FDros3.h. HDF5 applies
# these to every parameter not given to H5Pset_fapl_ros3_block_caching(), so
# they are also what h5py's own defaults must resolve to.
DEFAULT_BLOCK_SIZE = 16 * MiB
DEFAULT_BLOCK_CACHE_SIZE = 128 * MiB

block_caching = pytest.mark.skipif(
    h5py.version.hdf5_version_tuple < (2, 2, 0),
    reason='ROS3 block caching requires HDF5 >= 2.2.0')


def ros3_fapl(**kwds):
    """A ros3 file access property list with only ``kwds`` customized"""
    return make_fapl('ros3', libver=None, rdcc_nslots=None, rdcc_nbytes=None,
                     rdcc_w0=None, locking=None, page_buf_size=None,
                     min_meta_keep=None, min_raw_keep=None,
                     alignment_threshold=1, alignment_interval=1,
                     meta_block_size=None, **kwds)


@pytest.mark.network
@pytest.mark.parametrize("driver", [None, "ros3"])
@pytest.mark.parametrize(
    "url",
    [
        "s3://dandiarchive/ros3test.hdf5",
        "https://dandiarchive.s3.amazonaws.com/ros3test.hdf5"
    ]
)
def test_ros3(driver, url):
    """ ROS3 driver and options """

    with h5py.File(url, "r", aws_region=b"us-east-2", driver=driver) as f:
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


@pytest.mark.skipif(h5py.version.hdf5_version_tuple < (1, 14, 2),
                    reason='AWS S3 access token support in HDF5 >= 1.14.2')
def test_ros3_temp_token():
    """Set and get S3 access token"""
    token = b'#0123FakeToken4567/8/9'
    fapl = ros3_fapl(session_token=token)
    assert fapl.get_fapl_ros3_token() == token


@block_caching
@pytest.mark.parametrize(
    "block_size,block_cache_size,lock_superblock",
    [
        (1 * MiB, 8 * MiB, True),
        (4 * MiB, 4 * MiB, False),
        (2 * MiB, 40 * MiB, False),
    ]
)
def test_ros3_block_caching_roundtrip(block_size, block_cache_size,
                                      lock_superblock):
    """Set and get all ROS3 block caching parameters"""
    fapl = ros3_fapl()
    fapl.set_fapl_ros3_block_caching(block_size, block_cache_size,
                                     lock_superblock)
    assert fapl.get_fapl_ros3_block_caching() == (block_size, block_cache_size,
                                                  lock_superblock)


@block_caching
def test_ros3_block_caching_hdf5_defaults():
    """Unspecified block caching parameters fall back to the HDF5 defaults"""
    fapl = ros3_fapl()
    fapl.set_fapl_ros3_block_caching()
    config = fapl.get_fapl_ros3_block_caching()

    assert config.block_size == DEFAULT_BLOCK_SIZE
    assert config.block_cache_size == DEFAULT_BLOCK_CACHE_SIZE
    assert config.lock_superblock is True


@block_caching
@pytest.mark.parametrize(
    "kwds,expected",
    [
        ({'block_size': 1 * MiB},
         (1 * MiB, DEFAULT_BLOCK_CACHE_SIZE, True)),
        ({'block_cache_size': 40 * MiB},
         (DEFAULT_BLOCK_SIZE, 40 * MiB, True)),
        ({'lock_superblock': False},
         (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_CACHE_SIZE, False)),
    ]
)
def test_ros3_block_caching_partial(kwds, expected):
    """Setting one block caching parameter defaults the other two"""
    fapl = ros3_fapl()
    fapl.set_fapl_ros3_block_caching(**kwds)
    assert fapl.get_fapl_ros3_block_caching() == expected


@block_caching
@pytest.mark.parametrize(
    "kwds,expected",
    [
        ({"block_size": 1 * MiB}, (1 * MiB, DEFAULT_BLOCK_CACHE_SIZE, True)),
        ({"block_cache_size": 40 * MiB}, (DEFAULT_BLOCK_SIZE, 40 * MiB, True)),
        (
            {"lock_superblock": False},
            (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_CACHE_SIZE, False),
        ),
        (
            {
                "block_size": 2 * MiB,
                "block_cache_size": 32 * MiB,
                "lock_superblock": False,
            },
            (2 * MiB, 32 * MiB, False),
        ),
    ],
)
def test_ros3_block_caching_file_keywords(kwds, expected):
    """h5py.File() block caching keywords reach the ros3 driver"""
    assert ros3_fapl(**kwds).get_fapl_ros3_block_caching() == expected


@block_caching
def test_ros3_block_caching_untouched_by_default():
    """Block caching stays enabled when no keyword asks for anything"""
    config = ros3_fapl().get_fapl_ros3_block_caching()

    assert config.block_size > 0
    assert config.block_cache_size > 0
    assert config.lock_superblock is True


@block_caching
@pytest.mark.parametrize(
    "kwds,expected",
    [
        ({'block_size': 0}, (0, DEFAULT_BLOCK_CACHE_SIZE, True)),
        ({'block_cache_size': 0}, (DEFAULT_BLOCK_SIZE, 0, True)),
    ]
)
def test_ros3_block_caching_disabled(kwds, expected):
    """A zero block or cache size disables ROS3 block caching"""
    assert ros3_fapl(**kwds).get_fapl_ros3_block_caching() == expected


@block_caching
def test_ros3_block_size_clamped():
    """HDF5 clamps the block size down to the block cache size"""
    fapl = ros3_fapl(block_size=8 * MiB, block_cache_size=4 * MiB)
    assert fapl.get_fapl_ros3_block_caching() == (4 * MiB, 4 * MiB, True)


@block_caching
def test_ros3_block_caching_needs_ros3_driver():
    """Block caching is rejected on a property list without the ros3 driver"""
    fapl = h5p.create(h5p.FILE_ACCESS)
    fapl.set_fapl_sec2()

    with pytest.raises(ValueError, match='driver is not set'):
        fapl.set_fapl_ros3_block_caching(1 * MiB)

    with pytest.raises(ValueError, match='driver is not set'):
        fapl.get_fapl_ros3_block_caching()


@pytest.mark.skipif(h5py.version.hdf5_version_tuple >= (2, 2, 0),
                    reason='Requires HDF5 < 2.2.0')
@pytest.mark.parametrize(
    "kwds",
    [
        {'block_size': 1 * MiB},
        {'block_cache_size': 40 * MiB},
        {'lock_superblock': False},
    ]
)
def test_ros3_block_caching_unsupported(kwds):
    """Block caching keywords are refused when HDF5 is too old"""
    with pytest.raises(ValueError, match='HDF5 >= 2.2.0 required'):
        ros3_fapl(**kwds)


@pytest.mark.parametrize("driver", [None, 'sec2'])
def test_ros3_block_caching_wrong_driver(driver):
    """Block caching keywords are not silently swallowed by other drivers"""
    with pytest.raises(TypeError, match='block_size'):
        make_fapl(driver, libver=None, rdcc_nslots=None, rdcc_nbytes=None,
                  rdcc_w0=None, locking=None, page_buf_size=None,
                  min_meta_keep=None, min_raw_keep=None,
                  alignment_threshold=1, alignment_interval=1,
                  meta_block_size=None, block_size=1 * MiB)


@block_caching
@pytest.mark.network
@pytest.mark.parametrize(
    "kwds",
    [
        pytest.param({}, id="hdf5-defaults"),
        # A block cache holding only two 4 KiB blocks forces the driver to
        # evict and re-fetch while reading this file.
        pytest.param({"block_size": 4096, "block_cache_size": 8192}, id="tiny-cache"),
        pytest.param(
            {"block_size": 4096, "block_cache_size": 8192, "lock_superblock": False},
            id="tiny-cache-unlocked",
        ),
        pytest.param({"block_size": 0}, id="disabled"),
    ],
)
def test_ros3_block_caching_read(kwds):
    """Reading a file is unaffected by its block caching configuration"""
    url = 'https://dandiarchive.s3.amazonaws.com/ros3test.hdf5'

    with h5py.File(url, 'r', aws_region=b'us-east-2', **kwds) as f:
        assert list(f.keys()) == ['mydataset', 'subgroup', 'subgroup2']

        dset = f['mydataset']
        assert dset.shape == (100,)
        assert dset.dtype == 'int32'
        assert dset.attrs['temperature'] == 99.5
        assert (dset[...] == 0).all()
