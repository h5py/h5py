
"""
    Implements a new custom Distutils command for handling library
    configuration.

    The "configure" command here doesn't directly affect things like
    config.pxi; rather, it exists to provide a set of attributes that are
    used by the build_ext replacement in setup_build.py.

    Options from the command line and environment variables are stored
    between invocations in a pickle file.  This allows configuring the library
    once and e.g. calling "build" and "test" without recompiling everything
    or explicitly providing the same options every time.

    This module also contains the auto-detection logic for figuring out
    the currently installed HDF5 version.
"""

from distutils.cmd import Command
import os
import os.path as op
import re
import sys
import json


def load_stashed_config():
    """ Load settings dict from the pickle file """
    try:
        with open('h5config.json', 'r') as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise TypeError
    except Exception:
        return {}
    return cfg


def stash_config(dct):
    """Save settings dict to the pickle file."""
    with open('h5config.json', 'w') as f:
        json.dump(dct, f)


def validate_version(s):
    """Ensure that s contains an X.Y.Z format version string, or ValueError."""
    m = re.match('(\d+)\.(\d+)\.(\d+)$', s)
    if m:
        return tuple(int(x) for x in m.groups())
    raise ValueError(f"HDF5 version string {s!r} not in X.Y.Z format")


def mpi_enabled():
    return os.environ.get('HDF5_MPI') == "ON"


class BuildConfig:
    def __init__(self, hdf5_includedirs, hdf5_libdirs, hdf5_define_macros, hdf5_version, mpi):
        self.hdf5_includedirs = hdf5_includedirs
        self.hdf5_libdirs = hdf5_libdirs
        self.hdf5_define_macros = hdf5_define_macros
        self.hdf5_version = hdf5_version
        self.mpi = mpi

    @classmethod
    def from_env(cls):
        mpi = mpi_enabled()
        h5_inc, h5_lib, h5_macros = cls._find_hdf5_compiler_settings(mpi)

        h5_version_s = os.environ.get('HDF5_VERSION')
        if h5_version_s:
            h5_version = validate_version(h5_version_s)
        else:
            h5_version = autodetect_version(h5_lib)

        return cls(h5_inc, h5_lib, h5_macros, h5_version, mpi)

    @staticmethod
    def _find_hdf5_compiler_settings(mpi=False):
        """Get compiler settings from environment or pkgconfig.

        Returns (include_dirs, lib_dirs, define_macros)
        """
        hdf5 = os.environ.get('HDF5_DIR')
        hdf5_includedir = os.environ.get('HDF5_INCLUDEDIR')
        hdf5_libdir = os.environ.get('HDF5_LIBDIR')
        hdf5_pkgconfig_name = os.environ.get('HDF5_PKGCONFIG_NAME')

        if sum([
            bool(hdf5_includedir or hdf5_libdir),
            bool(hdf5),
            bool(hdf5_pkgconfig_name)
        ]) > 1:
            raise ValueError(
                "Specify only one of: HDF5 lib/include dirs, HDF5 prefix dir, "
                "or HDF5 pkgconfig name"
            )

        if hdf5_includedir or hdf5_libdir:
            inc_dirs = [hdf5_includedir] if hdf5_includedir else []
            lib_dirs = [hdf5_libdir] if hdf5_libdir else []
            return (inc_dirs, lib_dirs, [])

        # Specified a prefix dir (e.g. '/usr/local')
        if hdf5:
            inc_dirs = [op.join(hdf5, 'include')]
            lib_dirs = [op.join(hdf5, 'lib')]
            if sys.platform.startswith('win'):
                lib_dirs.append(op.join(hdf5, 'bin'))
            return (inc_dirs, lib_dirs, [])

        # Specified a name to be looked up in pkgconfig
        if hdf5_pkgconfig_name:
            import pkgconfig
            if not pkgconfig.exists(hdf5_pkgconfig_name):
                raise ValueError(
                    f"No pkgconfig information for {hdf5_pkgconfig_name}"
                )
            pc = pkgconfig.parse(hdf5_pkgconfig_name)
            return (pc['include_dirs'], pc['library_dirs'], pc['define_macros'])

        # Fallback: query pkgconfig for default hdf5 names
        import pkgconfig
        pc_name = 'hdf5-openmpi' if mpi else 'hdf5'
        pc = {}
        try:
            if pkgconfig.exists(pc_name):
                pc = pkgconfig.parse(pc_name)
        except EnvironmentError:
            if os.name != 'nt':
                print(
                    "Building h5py requires pkg-config unless the HDF5 path "
                    "is explicitly specified", file=sys.stderr
                )
                raise

        return (
            pc.get('include_dirs', []),
            pc.get('library_dirs', []),
            pc.get('define_macros', []),
        )

    def as_dict(self):
        return {
            'hdf5_includedirs': self.hdf5_includedirs,
            'hdf5_libdirs': self.hdf5_libdirs,
            'hdf5_define_macros': self.hdf5_define_macros,
            'hdf5_version': list(self.hdf5_version),  # list() to match the JSON
            'mpi': self.mpi,
        }

    def changed(self):
        """Has the config changed since the last build?"""
        return self.as_dict() != load_stashed_config()

    def record_built(self):
        """Record config after a successful build"""
        stash_config(self.as_dict())

    def summarise(self):
        def fmt_dirs(l):
            return '\n'.join((['['] + [f'  {d!r}' for d in l] + [']'])) if l else '[]'

        print('*' * 80)
        print(' ' * 23 + "Summary of the h5py configuration")
        print('')
        print("HDF5 include dirs:", fmt_dirs(self.hdf5_includedirs))
        print("HDF5 library dirs:", fmt_dirs(self.hdf5_libdirs))
        print("     HDF5 Version:", repr(self.hdf5_version))
        print("      MPI Enabled:", self.mpi)
        print(" Rebuild Required:", self.changed())
        print('')
        print('*' * 80)


def autodetect_version(libdirs):
    """
    Detect the current version of HDF5, and return X.Y.Z version string.

    Intended for Unix-ish platforms (Linux, OS X, BSD).
    Does not support Windows. Raises an exception if anything goes wrong.

    config: configuration for the build (configure command)
    """
    import re
    import ctypes
    from ctypes import byref

    if sys.platform.startswith('darwin'):
        default_path = 'libhdf5.dylib'
        regexp = re.compile(r'^libhdf5.dylib')
    elif sys.platform.startswith('win') or \
        sys.platform.startswith('cygwin'):
        default_path = 'hdf5.dll'
        regexp = re.compile(r'^hdf5.dll')
    else:
        default_path = 'libhdf5.so'
        regexp = re.compile(r'^libhdf5.so')

    path = None
    for d in libdirs:
        try:
            candidates = [x for x in os.listdir(d) if regexp.match(x)]
        except Exception:
            continue   # Skip invalid entries

        if len(candidates) != 0:
            candidates.sort(key=lambda x: len(x))   # Prefer libfoo.so to libfoo.so.X.Y.Z
            path = op.abspath(op.join(d, candidates[0]))
            break

    if path is None:
        path = default_path

    major = ctypes.c_uint()
    minor = ctypes.c_uint()
    release = ctypes.c_uint()

    print("Loading library to get version:", path)

    try:
        lib = ctypes.cdll.LoadLibrary(path)
        lib.H5get_libversion(byref(major), byref(minor), byref(release))
    except Exception:
        print("error: Unable to load dependency HDF5, make sure HDF5 is installed properly")
        raise

    return int(major.value), int(minor.value), int(release.value)
