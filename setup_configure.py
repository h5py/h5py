
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
    try:
        tpl = tuple(int(x) for x in s.split('.'))
        if len(tpl) != 3:
            raise ValueError
    except Exception:
        raise ValueError("HDF5 version string must be in X.Y.Z format")


def get_env_options():
    # The keys here match the option attributes on *configure*
    return {
        'hdf5': os.environ.get('HDF5_DIR'),
        'hdf5_includedir': os.environ.get('HDF5_INCLUDEDIR'),
        'hdf5_libdir': os.environ.get('HDF5_LIBDIR'),
        'hdf5_pkgconfig_name': os.environ.get('HDF5_PKGCONFIG_NAME'),
        'hdf5_version': os.environ.get('HDF5_VERSION'),
        'mpi': os.environ.get('HDF5_MPI'),
    }


class configure(Command):

    """
        Configure build options for h5py: custom path to HDF5, version of
        the HDF5 library, and whether MPI is enabled.

        Options can come from either command line options or environment
        variables (but specifying the same option in both is an error).
        Options not specified will be loaded from the previous configuration,
        so they are 'sticky' (except hdf5-version).

        When options change, the rebuild_required attribute is set, and
        may only be reset by calling reset_rebuild().  The custom build_ext
        command does this.
    """

    description = "Configure h5py build options"

    user_options = [('hdf5=', 'h', 'Custom path prefix to HDF5'),
                    ('hdf5-version=', '5', 'HDF5 version "X.Y.Z"'),
                    ('hdf5-includedir=', 'i', 'path to HDF5 headers'),
                    ('hdf5-libdir=', 'l', 'path to HDF5 library'),
                    ('hdf5-pkgconfig-name=', 'p', 'name of HDF5 pkgconfig file'),
                    ('mpi', 'm', 'Enable MPI building'),
                    ('reset', 'r', 'Reset config options') ]

    def initialize_options(self):
        self.hdf5 = None
        self.hdf5_version = None
        self.hdf5_includedir = None
        self.hdf5_libdir = None
        self.hdf5_pkgconfig_name = None
        self.mpi = None
        self.reset = None

    def finalize_options(self):
        # Merge environment options with command-line
        for setting, env_val in get_env_options().items():
            if env_val is not None:
                if getattr(self, setting) is not None:
                    raise ValueError(
                        f"Provide {setting} in command line or environment "
                        f"variable, not both."
                    )
                setattr(self, setting, env_val)

        if sum([
            bool(self.hdf5_includedir or self.hdf5_libdir),
            bool(self.hdf5),
            bool(self.hdf5_pkgconfig_name)
        ]) > 1:
            raise ValueError(
                "Specify only one of: HDF5 lib/include dirs, HDF5 prefix dir, "
                "or HDF5 pkgconfig name"
            )

        # Check version number format
        if self.hdf5_version is not None:
            validate_version(self.hdf5_version)

    def reset_rebuild(self):
        """ Mark this configuration as built """
        dct = load_stashed_config()
        dct['rebuild'] = False
        stash_config(dct)


    def _find_hdf5_compiler_settings(self, olds, mpi):
        """Returns (include_dirs, lib_dirs, define_macros)"""
        # Specified lib/include dirs explicitly
        if self.hdf5_includedir or self.hdf5_libdir:
            inc_dirs = [self.hdf5_includedir] if self.hdf5_includedir else []
            lib_dirs = [self.hdf5_libdir] if self.hdf5_libdir else []
            return (inc_dirs, lib_dirs, [])

        # Specified a prefix dir (e.g. '/usr/local')
        if self.hdf5:
            inc_dirs = [op.join(self.hdf5, 'include')]
            lib_dirs = [op.join(self.hdf5, 'lib')]
            if sys.platform.startswith('win'):
                lib_dirs.append(op.join(self.hdf5, 'bin'))
            return (inc_dirs, lib_dirs, [])

        # Specified a name to be looked up in pkgconfig
        if self.hdf5_pkgconfig_name:
            import pkgconfig
            if not pkgconfig.exists(self.hdf5_pkgconfig_name):
                raise ValueError(
                    f"No pkgconfig information for {self.hdf5_pkgconfig_name}"
                )
            pc = pkgconfig.parse(self.hdf5_pkgconfig_name)
            return (pc['include_dirs'], pc['library_dirs'], pc['define_macros'])

        # Re-use previously specified settings
        if olds.get('hdf5_includedirs') and olds.get('hdf5_libdirs'):
            return (
                olds['hdf5_includedirs'],
                olds['hdf5_libdirs'],
                olds.get('hdf5_define_macros', []),
            )

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

    def run(self):
        """ Distutils calls this when the command is run """

        # Step 1: Load previous settings and combine with current ones
        oldsettings = {} if self.reset else load_stashed_config()

        if self.mpi is None:
            self.mpi = oldsettings.get('mpi', False)

        self.hdf5_includedirs, self.hdf5_libdirs, self.hdf5_define_macros = \
            self._find_hdf5_compiler_settings(oldsettings, self.mpi)

        # Don't use the HDF5 version saved previously - that may be referring
        # to another library. It should always be specified or autodetected.
        # The HDF5 version is persisted only so we can check if it changed.
        if self.hdf5_version is None:
            self.hdf5_version = autodetect_version(self.hdf5_libdirs)

        # Step 2: determine if a rebuild is needed & save the settings

        current_settings = {
            'hdf5_includedirs': self.hdf5_includedirs,
            'hdf5_libdirs': self.hdf5_libdirs,
            'hdf5_define_macros': self.hdf5_define_macros,
            'hdf5_version': self.hdf5_version,
            'mpi': self.mpi,
            'rebuild': False,
        }

        self.rebuild_required = current_settings['rebuild'] = (
            # If we haven't built since a previous config change
            oldsettings.get('rebuild')
            # If the config has changed now
            or current_settings != oldsettings
            # Corner case: If options reset, but only if they previously
            # had non-default values (to handle multiple resets in a row)
            or bool(self.reset and any(load_stashed_config().values()))
        )

        stash_config(current_settings)

        # Step 3: print the resulting configuration to stdout

        def fmt_dirs(l):
            return '\n'.join((['['] + [f'  {d!r}' for d in l] + [']'])) if l else '[]'

        print('*' * 80)
        print(' ' * 23 + "Summary of the h5py configuration")
        print('')
        print("HDF5 include dirs:", fmt_dirs(self.hdf5_includedirs))
        print("HDF5 library dirs:", fmt_dirs(self.hdf5_libdirs))
        print("     HDF5 Version:", repr(self.hdf5_version))
        print("      MPI Enabled:", self.mpi)
        print(" Rebuild Required:", self.rebuild_required)
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

    return "{0}.{1}.{2}".format(int(major.value), int(minor.value), int(release.value))
