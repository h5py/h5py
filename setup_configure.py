
"""
    Implements a new custom setuptools command for handling library
    configuration.

    The "configure" command exists to provide a set of attributes that are
    used by the build_ext replacement in setup_build.py.

    Options from the command line and environment variables are stored
    between invocations in a pickle file.  This allows configuring the library
    once and e.g. calling "build" and "test" without recompiling everything
    or explicitly providing the same options every time.

    This module also contains the auto-detection logic for figuring out
    the currently installed HDF5 version.
"""

from typing import NamedTuple
import os
import os.path as op
import platform
import re
import sys
import json
from pathlib import Path

from dataclasses import dataclass, field, replace
from enum import auto, Flag

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

class VersionTuple(NamedTuple):
    major: int
    minor: int
    micro: int

    @classmethod
    def from_version_string(cls, s:str, /) -> "VersionTuple":
        if (m := re.match(r'(?P<major>\d+)\.(?P<minor>\d+)\.(?P<micro>\d+)(?:\.\d+)?$', s)) is None:
            raise ValueError(f"version string {s!r} not in X.Y.Z[.P] format")
        return VersionTuple(
            major=int(m.group("major")),
            minor=int(m.group("minor")),
            micro=int(m.group("micro")),
        )

    def as_version_string(self) -> str:
        return f"{self.major}.{self.minor}.{self.micro}"

def mpi_enabled() -> bool:
    return os.environ.get('HDF5_MPI') == "ON"


@dataclass(frozen=True, slots=True, kw_only=True)
class CompilerSettings:
    include_dirs: list[str] = field(default_factory=list)
    lib_dirs: list[str] = field(default_factory=list)
    define_macros: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True, kw_only=True)
class HDF5:
    version: VersionTuple
    settings: CompilerSettings


@dataclass(frozen=True, slots=True, kw_only=True)
class MSMPI:
    is_enabled: bool
    settings: CompilerSettings = CompilerSettings()

    @classmethod
    def from_env(cls) -> "MSMPI":
        if os.environ.get('H5PY_MSMPI') != 'ON':
            return cls(is_enabled=False)

        import platform
        bitness, _ = platform.architecture()
        if bitness == '64bit':
            mpi_lib_envvar = 'MSMPI_LIB64'
        else:
            mpi_lib_envvar = 'MSMPI_LIB32'

        missing_defs: list[str] = []
        if (MSMPI_INC := os.environ.get("MSMPI_INC")) is None:
            missing_defs.append("MSMPI_INC")
        if (MSMPI_LIB := os.environ.get(mpi_lib_envvar)) is None:
            missing_defs.append(mpi_lib_envvar)

        if missing_defs:
            raise AssertionError(
                "Environment isn't properly configured. "
                "The following environment variables need to be defined and are not: "
                f"{', '.join(missing_defs)}"
            )

        # no-op conversions to force narrowing at type check time
        MSMPI_INC = str(MSMPI_INC)
        MSMPI_LIB = str(MSMPI_LIB)

        return cls(
            is_enabled=True,
            settings=CompilerSettings(
                include_dirs=MSMPI_INC.split(';'),
                lib_dirs=MSMPI_LIB.split(';'),
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LibFlags:
    mpi: bool
    ros3: bool
    direct_vfd: bool

@dataclass(frozen=True, slots=True, kw_only=True)
class PartialLibFlags:
    # None represents an unknown state here
    mpi: bool
    ros3: bool | None
    direct_vfd: bool | None

    def is_complete(self) -> bool:
        return self.ros3 is not None and self.direct_vfd is not None

    def as_complete(self) -> LibFlags:
        if not self.is_complete():
            raise AssertionError
        return LibFlags(
            mpi=self.mpi,
            ros3=bool(self.ros3),
            direct_vfd=bool(self.direct_vfd),
        )


class BuildConfig:
    def __init__(self, hdf5: HDF5, flags: LibFlags) -> None:
        self.hdf5 = hdf5
        self.flags = flags
        if self.flags.mpi:
            self.msmpi = MSMPI.from_env()
        else:
            self.msmpi = MSMPI(is_enabled=False)

    @classmethod
    def from_env(cls) -> "BuildConfig":
        h5py_ros3 = os.environ.get('H5PY_ROS3')
        h5py_direct_vfd = os.environ.get('H5PY_DIRECT_VFD')
        flags = PartialLibFlags(
            mpi=mpi_enabled(),
            ros3=(h5py_ros3 == '1') if h5py_ros3 is not None else None,
            direct_vfd=(h5py_direct_vfd == '1') if h5py_direct_vfd is not None else None,
        )
        hdf5_settings = cls._find_hdf5_compiler_settings(flags.mpi)
        hdf5_version_t: VersionTuple | None
        if (hdf5_version_s := os.environ.get('HDF5_VERSION')) is not None:
            hdf5_version_t = VersionTuple.from_version_string(hdf5_version_s)
        else:
            hdf5_version_t = None

        if (
            hdf5_version_t is not None
            and not flags.mpi
            and flags.is_complete()
        ):
            # if we know config, don't use wrapper, it may not be supported
            return cls(
                hdf5=HDF5(version=hdf5_version_t, settings=hdf5_settings),
                flags=flags.as_complete(),
            )

        hdf5_wrapper = HDF5LibWrapper(hdf5_settings.lib_dirs)

        if hdf5_version_t is None:
            hdf5_version_t = hdf5_wrapper.autodetect_version()

        hdf5 = HDF5(version=hdf5_version_t, settings=hdf5_settings)

        if flags.mpi and not hdf5_wrapper.has_mpi_support():
            raise RuntimeError("MPI support not detected")

        if h5py_ros3 is None:
            flags = replace(flags, ros3=hdf5_wrapper.has_ros3_support())

        if h5py_direct_vfd is None:
            flags = replace(flags, direct_vfd=hdf5_wrapper.has_direct_vfd_support())

        return cls(hdf5=hdf5, flags=flags.as_complete())

    @staticmethod
    def _find_hdf5_compiler_settings(mpi: bool = False) -> CompilerSettings:
        """Get compiler settings from environment or pkgconfig."""
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
                "Specify at most one of: HDF5 lib/include dirs, HDF5 prefix dir, "
                "or HDF5 pkgconfig name. Received:\n"
                f"HDF5={hdf5}\n"
                f"HDF5_INCLUDEDIR={hdf5_includedir}\n"
                f"HDF5_LIBDIR={hdf5_libdir}\n"
                f"HDF5_PKGCONFIG_NAME={hdf5_pkgconfig_name}\n"
            )

        if hdf5_includedir or hdf5_libdir:
            inc_dirs = [hdf5_includedir] if hdf5_includedir else []
            lib_dirs = [hdf5_libdir] if hdf5_libdir else []
            return CompilerSettings(
                include_dirs=inc_dirs,
                lib_dirs=lib_dirs,
            )

        # Specified a prefix dir (e.g. '/usr/local')
        if hdf5:
            inc_dirs = [op.join(hdf5, 'include')]
            for subdir in ["lib64", "lib32", "lib"]:
                p = Path(hdf5, subdir)
                if p.is_dir() and list(p.glob("libhdf5.*")):
                    break
            else:
                raise FileNotFoundError("couldn't locate HDF5's lib directory")

            lib_dirs = [str(p)]
            if sys.platform.startswith('win'):
                lib_dirs.append(op.join(hdf5, 'bin'))
            return CompilerSettings(
                include_dirs=inc_dirs,
                lib_dirs=lib_dirs,
            )

        # Specified a name to be looked up in pkgconfig
        if hdf5_pkgconfig_name:
            import pkgconfig
            if not pkgconfig.exists(hdf5_pkgconfig_name):
                raise ValueError(
                    f"No pkgconfig information for {hdf5_pkgconfig_name}"
                )
            pc = pkgconfig.parse(hdf5_pkgconfig_name)
            return CompilerSettings(
                include_dirs=pc['include_dirs'],
                lib_dirs=pc['library_dirs'],
                define_macros=pc['define_macros'],
            )

        # Fallback: query pkgconfig for default hdf5 names
        import pkgconfig
        pc_name = 'hdf5-openmpi' if mpi else 'hdf5'
        pc = {}
        try:
            if pkgconfig.exists(pc_name):
                pc = pkgconfig.parse(pc_name)
        except OSError:
            if os.name != 'nt':
                print(
                    "Building h5py requires pkg-config unless the HDF5 path "
                    "is explicitly specified using the environment variable HDF5_DIR. "
                    "For more information and details, "
                    "see https://docs.h5py.org/en/stable/build.html#custom-installation", file=sys.stderr
                )
                raise

        return CompilerSettings(
            include_dirs=pc.get('include_dirs', []),
            lib_dirs=pc.get('library_dirs', []),
            define_macros=pc.get('define_macros', []),
        )

    def as_dict(self):
        return {
            'hdf5_includedirs': self.hdf5.settings.include_dirs,
            'hdf5_libdirs': self.hdf5.settings.lib_dirs,
            'hdf5_define_macros': self.hdf5.settings.define_macros,
            'hdf5_version': list(self.hdf5.version),  # list() to match the JSON
            'mpi': self.flags.mpi,
            'ros3': self.flags.ros3,
            'direct_vfd': self.flags.direct_vfd,
            'msmpi': self.msmpi.is_enabled,
            'msmpi_inc_dirs': self.msmpi.settings.include_dirs,
            'msmpi_lib_dirs': self.msmpi.settings.lib_dirs,
        }

    def changed(self) -> bool:
        """Has the config changed since the last build?"""
        return self.as_dict() != load_stashed_config()

    def record_built(self) -> None:
        """Record config after a successful build"""
        stash_config(self.as_dict())

    def summarise(self) -> None:
        def fmt_dirs(l):
            return '\n'.join((['['] + [f'  {d!r}' for d in l] + [']'])) if l else '[]'

        print('*' * 80)
        print(' ' * 23 + "Summary of the h5py configuration")
        print('')
        print("  HDF5 include dirs:", fmt_dirs(self.hdf5.settings.include_dirs))
        print("  HDF5 library dirs:", fmt_dirs(self.hdf5.settings.lib_dirs))
        print("       HDF5 Version:", self.hdf5.version.as_version_string())
        print("        MPI Enabled:", self.flags.mpi)
        print("   ROS3 VFD Enabled:", self.flags.ros3)
        print(" DIRECT VFD Enabled:", self.flags.direct_vfd)
        print("   Rebuild Required:", self.changed())
        print("     MS-MPI Enabled:", self.msmpi.is_enabled)
        print("MS-MPI include dirs:", self.msmpi.settings.include_dirs)
        print("MS-MPI library dirs:", self.msmpi.settings.lib_dirs)
        print('')
        print('*' * 80)


class HDF5LibWrapper:

    def __init__(self, libdirs):
        self._load_hdf5_lib(libdirs)

    def _load_hdf5_lib(self, libdirs):
        """
        Detect and load the HDF5 library.

        Raises an exception if anything goes wrong.

        libdirs: the library paths to search for the library
        """
        import ctypes

        # extra keyword args to pass to LoadLibrary
        load_kw = {}
        if sys.platform.startswith('darwin'):
            default_path = 'libhdf5.dylib'
            regexp = re.compile(r'^libhdf5.dylib')
        elif sys.platform.startswith('win'):
            if 'MSC' in sys.version:
                default_path = 'hdf5.dll'
                regexp = re.compile(r'^hdf5.dll')
            else:
                default_path = 'libhdf5-0.dll'
                regexp = re.compile(r'^libhdf5-[0-9].dll')
            # To overcome "difficulty" loading the library on windows
            # https://bugs.python.org/issue42114
            load_kw['winmode'] = 0
        elif sys.platform.startswith('cygwin'):
            default_path = 'cyghdf5-200.dll'
            regexp = re.compile(r'^cyghdf5-\d+.dll$')
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

        print("Loading library to get build settings and version:", path)

        self._lib_path = path

        if op.isabs(path) and not op.exists(path):
            raise FileNotFoundError(f"{path} is missing")

        try:
            lib = ctypes.CDLL(path, **load_kw)
        except Exception:
            print("error: Unable to load dependency HDF5, make sure HDF5 is installed properly")
            print(f"on {sys.platform=} with {platform.machine()=}")
            print("Library dirs checked:", libdirs)
            raise

        self._lib = lib

    def autodetect_version(self) -> VersionTuple:
        """
        Detect the current version of HDF5, and return a (X, Y, Z) version tuple.

        Raises an exception if anything goes wrong.
        """
        import ctypes
        from ctypes import byref

        major = ctypes.c_uint()
        minor = ctypes.c_uint()
        micro = ctypes.c_uint()

        try:
            self._lib.H5get_libversion(byref(major), byref(minor), byref(micro))
        except Exception:
            print("error: Unable to find HDF5 version")
            raise

        return VersionTuple(
            major=int(major.value),
            minor=int(minor.value),
            micro=int(micro.value),
        )

    def load_function(self, func_name):
        try:
            return getattr(self._lib, func_name)
        except AttributeError:
            # No such function
            return None

    def has_functions(self, *func_names) -> bool:
        for func_name in func_names:
            if self.load_function(func_name) is None:
                return False
        return True

    def has_mpi_support(self) -> bool:
        return self.has_functions("H5Pget_fapl_mpio", "H5Pset_fapl_mpio")

    def has_ros3_support(self) -> bool:
        return self.has_functions("H5Pget_fapl_ros3", "H5Pset_fapl_ros3")

    def has_direct_vfd_support(self) -> bool:
        return self.has_functions("H5Pget_fapl_direct", "H5Pset_fapl_direct")
