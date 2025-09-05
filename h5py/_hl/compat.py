"""
Compatibility module for high-level h5py
"""
import os
import sys
from ..version import hdf5_built_version_tuple

# HDF5 supported passing paths as UTF-8 for Windows from 1.10.6, but this
# was broken again in 1.14.4 - https://github.com/HDFGroup/hdf5/issues/5037 .
# The change was reverted in 1.14.6.
if (1, 14, 4) <= hdf5_built_version_tuple < (1, 14, 6):
    WINDOWS_ENCODING = "mbcs"
else:
    WINDOWS_ENCODING = "utf-8"


def filename_encode(filename):
    """
    Encode filename for use in the HDF5 library.

    Due to how HDF5 handles filenames on different systems, this should be
    called on any filenames passed to the HDF5 library. See the documentation on
    filenames in h5py for more information.
    """
    filename = os.fspath(filename)
    if sys.platform == "win32" and isinstance(filename, str):
        return filename.encode(WINDOWS_ENCODING, "strict")
    else:
        return os.fsencode(filename)


def filename_decode(filename):
    """
    Decode filename used by HDF5 library.

    Due to how HDF5 handles filenames on different systems, this should be
    called on any filenames passed from the HDF5 library. See the documentation
    on filenames in h5py for more information.
    """
    if not isinstance(filename, (str, bytes)):
        raise TypeError(f"expect bytes or str, not {type(filename).__name__}")

    if sys.platform == "win32" and isinstance(filename, bytes):
        return filename.decode(WINDOWS_ENCODING, "strict")
    else:
        return os.fsdecode(filename)
