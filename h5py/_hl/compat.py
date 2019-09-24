"""
Compatibility module for high-level h5py
"""
import sys
from os import fspath, fsencode, fsdecode

WINDOWS_ENCODING = "mbcs"


def filename_encode(filename):
    """
    Encode filename for use in the HDF5 library.

    Due to how HDF5 handles filenames on different systems, this should be
    called on any filenames passed to the HDF5 library. See the documentation on
    filenames in h5py for more information.
    """
    filename = fspath(filename)
    if sys.platform == "win32":
        if isinstance(filename, str):
            return filename.encode(WINDOWS_ENCODING, "strict")
        return filename
    return fsencode(filename)


def filename_decode(filename):
    """
    Decode filename used by HDF5 library.

    Due to how HDF5 handles filenames on different systems, this should be
    called on any filenames passed from the HDF5 library. See the documentation
    on filenames in h5py for more information.
    """
    if sys.platform == "win32":
        if isinstance(filename, bytes):
            return filename.decode(WINDOWS_ENCODING, "strict")
        elif isinstance(filename, str):
            return filename
        else:
            raise TypeError("expect bytes or str, not %s" % type(filename).__name__)
    return fsdecode(filename)
