#!/usr/bin/env python3
import argparse
import json
import os
import os.path as op
import platform
import re
import sys
import sysconfig
from pathlib import Path

import numpy as np
from Cython import __version__ as cython_version
from packaging.version import Version


def load_stashed_config():
    """Load settings dict from the pickle file"""
    try:
        with open("h5config.json", "r") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise TypeError
    except Exception:
        return {}
    return cfg


def validate_version(s):
    """Ensure that s contains an X.Y.Z format version string, or ValueError."""
    # HDF5 tags can have a patch version, which we'll ignore for now.
    m = re.match(r"(\d+)\.(\d+)\.(\d+)(?:\.\d+)?$", s)
    if m:
        return tuple(int(x) for x in m.groups())
    raise ValueError(f"HDF5 version string {s!r} not in X.Y.Z[.P] format")


def mpi_enabled():
    return os.environ.get("HDF5_MPI") == "ON"


class BuildConfig:
    def __init__(
        self,
        hdf5_includedirs,
        hdf5_libdirs,
        hdf5_define_macros,
        hdf5_version,
        mpi,
        ros3,
        direct_vfd,
    ):
        self.hdf5_includedirs = hdf5_includedirs
        self.hdf5_libdirs = hdf5_libdirs
        self.hdf5_define_macros = hdf5_define_macros
        self.hdf5_version = hdf5_version
        self.mpi = mpi
        self.ros3 = ros3
        self.direct_vfd = direct_vfd

        if self.mpi and os.environ.get("H5PY_MSMPI") == "ON":
            self.msmpi = True
            self.msmpi_inc_dirs = os.environ.get("MSMPI_INC").split(";")
            import platform

            bitness, _ = platform.architecture()
            if bitness == "64bit":
                mpi_lib_envvar = "MSMPI_LIB64"
            else:
                mpi_lib_envvar = "MSMPI_LIB32"
            self.msmpi_lib_dirs = os.environ.get(mpi_lib_envvar).split(";")
        else:
            self.msmpi = False
            self.msmpi_inc_dirs = []
            self.msmpi_lib_dirs = []

    @classmethod
    def from_env(cls):
        mpi = mpi_enabled()
        h5_inc, h5_lib, h5_macros = cls._find_hdf5_compiler_settings(mpi)

        h5_version_s = os.environ.get("HDF5_VERSION")
        h5py_ros3 = os.environ.get("H5PY_ROS3")
        h5py_direct_vfd = os.environ.get("H5PY_DIRECT_VFD")

        if h5_version_s and not mpi and h5py_ros3 and h5py_direct_vfd:
            # if we know config, don't use wrapper, it may not be supported
            return cls(
                h5_inc,
                h5_lib,
                h5_macros,
                validate_version(h5_version_s),
                mpi,
                h5py_ros3 == "1",
                h5py_direct_vfd == "1",
            )

        h5_wrapper = HDF5LibWrapper(h5_lib)
        if h5_version_s:
            h5_version = validate_version(h5_version_s)
        else:
            h5_version = h5_wrapper.autodetect_version()
            if mpi and not h5_wrapper.has_mpi_support():
                raise RuntimeError("MPI support not detected")

        if h5py_ros3:
            ros3 = h5py_ros3 == "1"
        else:
            ros3 = h5_wrapper.has_ros3_support()

        if h5py_direct_vfd:
            direct_vfd = h5py_direct_vfd == "1"
        else:
            direct_vfd = h5_wrapper.has_direct_vfd_support()

        return cls(h5_inc, h5_lib, h5_macros, h5_version, mpi, ros3, direct_vfd)

    @staticmethod
    def _find_hdf5_compiler_settings(mpi=False):
        """Get compiler settings from environment or pkgconfig.

        Returns (include_dirs, lib_dirs, define_macros)
        """
        hdf5 = os.environ.get("HDF5_DIR")
        hdf5_includedir = os.environ.get("HDF5_INCLUDEDIR")
        hdf5_libdir = os.environ.get("HDF5_LIBDIR")
        hdf5_pkgconfig_name = os.environ.get("HDF5_PKGCONFIG_NAME")

        if (
            sum(
                [
                    bool(hdf5_includedir or hdf5_libdir),
                    bool(hdf5),
                    bool(hdf5_pkgconfig_name),
                ]
            )
            > 1
        ):
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
            inc_dirs = [op.join(hdf5, "include")]
            lib_dirs = [op.join(hdf5, "lib")]
            if sys.platform.startswith("win"):
                lib_dirs.append(op.join(hdf5, "bin"))
            return (inc_dirs, lib_dirs, [])

        # Specified a name to be looked up in pkgconfig
        if hdf5_pkgconfig_name:
            import pkgconfig

            if not pkgconfig.exists(hdf5_pkgconfig_name):
                raise ValueError(f"No pkgconfig information for {hdf5_pkgconfig_name}")
            pc = pkgconfig.parse(hdf5_pkgconfig_name)
            return (pc["include_dirs"], pc["library_dirs"], pc["define_macros"])

        # Fallback: query pkgconfig for default hdf5 names
        import pkgconfig

        pc_name = "hdf5-openmpi" if mpi else "hdf5"
        pc = {}
        try:
            if pkgconfig.exists(pc_name):
                pc = pkgconfig.parse(pc_name)
        except OSError:
            if os.name != "nt":
                print(
                    "Building h5py requires pkg-config unless the HDF5 path "
                    "is explicitly specified using the environment variable HDF5_DIR. "
                    "For more information and details, "
                    "see https://docs.h5py.org/en/stable/build.html#custom-installation",
                    file=sys.stderr,
                )
                raise

        return (
            pc.get("include_dirs", []),
            pc.get("library_dirs", []),
            pc.get("define_macros", []),
        )

    def as_dict(self):
        return {
            "hdf5_includedirs": self.hdf5_includedirs,
            "hdf5_libdirs": self.hdf5_libdirs,
            "hdf5_define_macros": self.hdf5_define_macros,
            "hdf5_version": list(self.hdf5_version),  # list() to match the JSON
            "mpi": self.mpi,
            "ros3": self.ros3,
            "direct_vfd": self.direct_vfd,
            "msmpi": self.msmpi,
            "msmpi_inc_dirs": self.msmpi_inc_dirs,
            "msmpi_lib_dirs": self.msmpi_lib_dirs,
        }

    def changed(self):
        """Has the config changed since the last build?"""
        return self.as_dict() != load_stashed_config()

    def summarise(self):
        def fmt_dirs(l):
            return "\n".join((["["] + [f"  {d!r}" for d in l] + ["]"])) if l else "[]"

        print("*" * 80)
        print(" " * 23 + "Summary of the h5py configuration")
        print("")
        print("  HDF5 include dirs:", fmt_dirs(self.hdf5_includedirs))
        print("  HDF5 library dirs:", fmt_dirs(self.hdf5_libdirs))
        print("       HDF5 Version:", repr(self.hdf5_version))
        print("        MPI Enabled:", self.mpi)
        print("   ROS3 VFD Enabled:", self.ros3)
        print(" DIRECT VFD Enabled:", self.direct_vfd)
        print("   Rebuild Required:", self.changed())
        print("     MS-MPI Enabled:", self.msmpi)
        print("MS-MPI include dirs:", self.msmpi_inc_dirs)
        print("MS-MPI library dirs:", self.msmpi_lib_dirs)
        print("")
        print("*" * 80)


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
        if sys.platform.startswith("darwin"):
            default_path = "libhdf5.dylib"
            regexp = re.compile(r"^libhdf5.dylib")
        elif sys.platform.startswith("win"):
            if "MSC" in sys.version:
                default_path = "hdf5.dll"
                regexp = re.compile(r"^hdf5.dll")
            else:
                default_path = "libhdf5-0.dll"
                regexp = re.compile(r"^libhdf5-[0-9].dll")
            # To overcome "difficulty" loading the library on windows
            # https://bugs.python.org/issue42114
            load_kw["winmode"] = 0
        elif sys.platform.startswith("cygwin"):
            default_path = "cyghdf5-200.dll"
            regexp = re.compile(r"^cyghdf5-\d+.dll$")
        else:
            default_path = "libhdf5.so"
            regexp = re.compile(r"^libhdf5.so")

        path = None
        for d in libdirs:
            try:
                candidates = [x for x in os.listdir(d) if regexp.match(x)]
            except Exception:
                continue  # Skip invalid entries

            if len(candidates) != 0:
                candidates.sort(
                    key=lambda x: len(x)
                )  # Prefer libfoo.so to libfoo.so.X.Y.Z
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
            print(
                "error: Unable to load dependency HDF5, make sure HDF5 is installed properly"
            )
            print(f"on {sys.platform=} with {platform.machine()=}")
            print("Library dirs checked:", libdirs)
            raise

        self._lib = lib

    def autodetect_version(self):
        """
        Detect the current version of HDF5, and return X.Y.Z version string.

        Raises an exception if anything goes wrong.
        """
        import ctypes
        from ctypes import byref

        major = ctypes.c_uint()
        minor = ctypes.c_uint()
        release = ctypes.c_uint()

        try:
            self._lib.H5get_libversion(byref(major), byref(minor), byref(release))
        except Exception:
            print("error: Unable to find HDF5 version")
            raise

        return int(major.value), int(minor.value), int(release.value)

    def load_function(self, func_name):
        try:
            return getattr(self._lib, func_name)
        except AttributeError:
            # No such function
            return None

    def has_functions(self, *func_names):
        for func_name in func_names:
            if self.load_function(func_name) is None:
                return False
        return True

    def has_mpi_support(self):
        return self.has_functions("H5Pget_fapl_mpio", "H5Pset_fapl_mpio")

    def has_ros3_support(self):
        return self.has_functions("H5Pget_fapl_ros3", "H5Pset_fapl_ros3")

    def has_direct_vfd_support(self):
        return self.has_functions("H5Pget_fapl_direct", "H5Pset_fapl_direct")


def version_tuple(dunder_version: str) -> tuple[int, int, int]:
    v = Version(dunder_version)
    return (v.major, v.minor, v.micro)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=Path,
        help="path to the output directory",
    )
    args = parser.parse_args(argv)

    # Get configuration from environment variables
    config = BuildConfig.from_env()
    config.summarise()

    if config.hdf5_version < (1, 10, 7) or config.hdf5_version == (1, 12, 0):
        raise Exception(
            f"This version of h5py requires HDF5 >= 1.10.7 and != 1.12.0 (got version "
            f"{config.hdf5_version} from environment variable or library)"
        )

    # Record the configuration we built
    target = args.output_dir / "h5config.json"
    with target.open("w") as f:
        json.dump(config.as_dict(), f)
    print(f"wrote {target}")

    templ_config = {
        "MPI": bool(config.mpi),
        "ROS3": bool(config.ros3),
        "HDF5_VERSION": config.hdf5_version,
        "DIRECT_VFD": bool(config.direct_vfd),
        "VOL_MIN_HDF5_VERSION": (1, 11, 5),
        "COMPLEX256_SUPPORT": hasattr(np, "complex256"),
        "NUMPY_BUILD_VERSION": np.__version__,
        "NUMPY_BUILD_VERSION_TUPLE": version_tuple(np.__version__),
        "CYTHON_BUILD_VERSION": cython_version,
        "PLATFORM_SYSTEM": platform.system(),
        "OBJECTS_USE_LOCKING": True,
        "OBJECTS_DEBUG_ID": False,
        "FREE_THREADING": sysconfig.get_config_var("Py_GIL_DISABLED") == 1,
    }

    subs = args.output_dir / "template_substitutions.json"
    with subs.open("w") as f:
        json.dump(templ_config, f)
    print(f"wrote {subs}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
