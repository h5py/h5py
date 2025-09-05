# -*- coding: utf-8 -*-
"""
Script for downloading and building HDF5 on Windows
This does not support MPI, nor non-Windows OSes

This script may not completely clean up after itself, it is designed to run in a
CI environment which thrown away each time
"""

from os import environ, makedirs, walk, getcwd, chdir
from os.path import join as pjoin, exists, basename, dirname, abspath
from tempfile import TemporaryFile, TemporaryDirectory
from sys import stderr
from shutil import copy
from glob import glob
from subprocess import run
from zipfile import ZipFile
import requests
import platform

HDF5_URL = "https://github.com/HDFGroup/hdf5/archive/refs/tags/{zip_file}"
ZLIB_ROOT = environ.get('ZLIB_ROOT')
arch = platform.machine().lower()

CI_DIR = dirname(abspath(__file__))

CMAKE_CONFIGURE_CMD = [
    "cmake", "-DBUILD_SHARED_LIBS:BOOL=ON", "-DCMAKE_BUILD_TYPE:STRING=RELEASE",
    "-DHDF5_BUILD_CPP_LIB=OFF", "-DHDF5_BUILD_HL_LIB=ON",
    "-DHDF5_BUILD_TOOLS:BOOL=OFF", "-DBUILD_TESTING:BOOL=OFF",
]
if ZLIB_ROOT:
    if arch in ("arm64", "aarch64"):
        # ZLIB includes based on vcpkg layout
        CMAKE_CONFIGURE_CMD += [
            "-DHDF5_ENABLE_Z_LIB_SUPPORT=ON",
            f"-DZLIB_INCLUDE_DIR={ZLIB_ROOT}\\include",
            f"-DZLIB_LIBRARY_RELEASE={ZLIB_ROOT}\\lib\\zlib.lib",
            f"-DZLIB_LIBRARY_DEBUG={ZLIB_ROOT}\\debug\\lib\\zlibd.lib",
        ]
    elif arch in ("amd64", "x86_64"):
        ## ZLIB includes based on nuget layout
        CMAKE_CONFIGURE_CMD += [
            "-DHDF5_ENABLE_Z_LIB_SUPPORT=ON",
            f"-DZLIB_INCLUDE_DIR={ZLIB_ROOT}\\include",
            f"-DZLIB_LIBRARY_RELEASE={ZLIB_ROOT}\\lib_release\\zlib.lib",
            f"-DZLIB_LIBRARY_DEBUG={ZLIB_ROOT}\\lib_debug\\zlibd.lib",
        ]
    else:
        raise RuntimeError(f"Unexpected architecture detected: {platform.machine()=}")

CMAKE_BUILD_CMD = ["cmake", "--build"]
CMAKE_INSTALL_ARG = ["--target", "install", '--config', 'Release']
CMAKE_INSTALL_PATH_ARG = "-DCMAKE_INSTALL_PREFIX={install_path}"
CMAKE_HDF5_LIBRARY_PREFIX = ["-DHDF5_EXTERNAL_LIB_PREFIX=h5py_"]
REL_PATH_TO_CMAKE_CFG = "hdf5-{dir_suffix}"
DEFAULT_VERSION = '1.12.2'
VSVERSION_TO_GENERATOR = {
    "9": "Visual Studio 9 2008",
    "10": "Visual Studio 10 2010",
    "14": "Visual Studio 14 2015",
    "15": "Visual Studio 15 2017",
    "16": "Visual Studio 16 2019",
    "9-64": "Visual Studio 9 2008 Win64",
    "10-64": "Visual Studio 10 2010 Win64",
    "14-64": "Visual Studio 14 2015 Win64",
    "15-64": "Visual Studio 15 2017 Win64",
    "16-64": "Visual Studio 16 2019",
    "17-64": "Visual Studio 17 2022",
    "17-arm64": "Visual Studio 17 2022",
}


def download_hdf5(version, outfile):
    zip_fmt1 = "hdf5-" + version.replace(".", "_") + ".zip"
    zip_fmt2 = "hdf5_" + version.replace("-", ".") + ".zip"
    files = [HDF5_URL.format(zip_file=zip_fmt1),
             HDF5_URL.format(zip_file=zip_fmt2),
             ]

    for file in files:
        print(f"Downloading hdf5 from {file} ...", file=stderr)
        r = requests.get(file, stream=True)
        try:
            r.raise_for_status()
        except requests.HTTPError:
            print(f"Failed to download hdf5 from {file}", file=stderr)
            continue
        else:
            for chunk in r.iter_content(chunk_size=None):
                outfile.write(chunk)
            print(f"Successfully downloaded hdf5 from {file}", file=stderr)
            return file

    msg = (f"Cannot download HDF5 source ({version}) from any of the "
           f"following URLs: {[f for f in files]}")
    raise RuntimeError(msg)


def build_hdf5(version, hdf5_file, install_path, cmake_generator, use_prefix,
               dl_zip):
    try:
        run(["cmake", "--version"])  # Show what version of cmake we'll use
        with TemporaryDirectory() as hdf5_extract_path:
            generator_args = (
                ["-G", cmake_generator]
                if cmake_generator is not None
                else []
            )
            prefix_args = CMAKE_HDF5_LIBRARY_PREFIX if use_prefix else []

            with ZipFile(hdf5_file) as z:
                z.extractall(hdf5_extract_path)

            old_dir = getcwd()

            with TemporaryDirectory() as new_dir:
                chdir(new_dir)
                cfg_cmd = CMAKE_CONFIGURE_CMD + [
                    get_cmake_install_path(install_path),
                    get_cmake_config_path(hdf5_extract_path, dl_zip),
                ] + generator_args + prefix_args
                print("Configuring HDF5 version {version}...".format(version=version))
                print(' '.join(cfg_cmd), file=stderr)
                run(cfg_cmd, check=True)

                build_cmd = CMAKE_BUILD_CMD + [
                    '.',
                ] + CMAKE_INSTALL_ARG
                print("Building HDF5 version {version}...".format(version=version))
                print(' '.join(build_cmd), file=stderr)
                run(build_cmd, check=True)

                print("Installed HDF5 version {version} to {install_path}".format(
                    version=version, install_path=install_path,
                ), file=stderr)
                chdir(old_dir)
    except OSError as e:
        if e.winerror == 145:
            print("Hit the rmtree race condition, continuing anyway...", file=stderr)
        else:
            raise
    for f in glob(pjoin(install_path, 'bin/*.dll')):
        copy(f, pjoin(install_path, 'lib'))


def get_cmake_config_path(extract_point, zip_file):
    dir_suffix = basename(zip_file).removesuffix(".zip")
    return pjoin(extract_point, REL_PATH_TO_CMAKE_CFG.format(dir_suffix=dir_suffix))


def get_cmake_install_path(install_path):
    if install_path is not None:
        return CMAKE_INSTALL_PATH_ARG.format(install_path=install_path)
    return ' '


def hdf5_install_cached(install_path):
    if exists(pjoin(install_path, "lib", "hdf5.dll")):
        return True
    return False


def main():
    install_path = environ.get("HDF5_DIR")
    version = environ.get("HDF5_VERSION", DEFAULT_VERSION)
    vs_version = environ.get("HDF5_VSVERSION")
    use_prefix = True if environ.get("H5PY_USE_PREFIX") is not None else False

    if install_path is not None:
        if not exists(install_path):
            makedirs(install_path)
    if vs_version is not None:
        cmake_generator = VSVERSION_TO_GENERATOR[vs_version]
        if vs_version == '9-64':
            # Needed for
            # http://help.appveyor.com/discussions/kb/38-visual-studio-2008-64-bit-builds
            run("ci\\appveyor\\vs2008_patch\\setup_x64.bat")
    else:
        cmake_generator = None

    if not hdf5_install_cached(install_path):
        with TemporaryFile() as f:
            dl_zip = download_hdf5(version, f)
            build_hdf5(version, f, install_path, cmake_generator, use_prefix,
                       dl_zip)
    else:
        print("using cached hdf5", file=stderr)
    if install_path is not None:
        print("hdf5 files: ", file=stderr)
        for dirpath, dirnames, filenames in walk(install_path):
            for file in filenames:
                print(" * " + pjoin(dirpath, file))


if __name__ == '__main__':
    main()
