import re
import ctypes
from ctypes import byref
import os
import os.path as op
import sys

COMPAT_VERSION = (1, 8, 4)

def detect(libdirs):
    """
    Detect the current version of HDF5, and return it as a tuple
    (major, minor, release).

    If the version can't be determined, defaults to COMPAT_VERSION.

    libdirs: list of library paths to search for libhdf5.so
    """

    if sys.platform.startswith('win'):
        regexp = re.compile('^hdf5.dll$')
    else:
        regexp = re.compile(r'^libhdf5.so')

    try:
        path = None

        for d in libdirs:
            try:
                candidates = [x for x in os.listdir(d) if regexp.match(x)]
                if len(candidates) != 0:
                    candidates.sort(key=lambda x: len(x))   # Prefer libfoo.so to libfoo.so.X.Y.Z
                    path = op.abspath(op.join(d, candidates[0]))
            except Exception:
                pass   # We skip invalid entries, because that's what the C compiler does

        if path is None:
            path = "libhdf5.so"

        lib = ctypes.cdll.LoadLibrary(path)

    except Exception as e:

        print("Failed to detect HDF5 version; defaulting to %s" % ".".join(str(x) for x in COMPAT_VERSION))
        print(e)
        return COMPAT_VERSION

    major = ctypes.c_uint()
    minor = ctypes.c_uint()
    release = ctypes.c_uint()

    lib.H5get_libversion(byref(major), byref(minor), byref(release))

    vers = (int(major.value), int(minor.value), int(release.value))

    print("Autodetected HDF5 %s" % (vers,))

    return vers


