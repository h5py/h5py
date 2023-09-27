# -*- coding: utf-8 -*-
"""
Script for downloading and building HDF5 on Windows
This does not support MPI, nor non-Windows OSes

This script may not completely clean up after itself, it is designed to run in a
CI environment which thrown away each time
"""

from os import environ, makedirs, walk, getcwd, chdir
from os.path import join as pjoin, exists
from tempfile import TemporaryFile, TemporaryDirectory
from sys import exit, stderr
from shutil import copy
from glob import glob
from subprocess import run
from zipfile import ZipFile
import requests

HDF5_URL = "https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-{series}/hdf5-{version}/src/hdf5-{version}.zip"
ZLIB_ROOT = environ.get('ZLIB_ROOT')

CMAKE_CONFIGURE_CMD = [
    "cmake", "-DBUILD_SHARED_LIBS:BOOL=ON", "-DCMAKE_BUILD_TYPE:STRING=RELEASE",
    "-DHDF5_BUILD_CPP_LIB=OFF", "-DHDF5_BUILD_HL_LIB=ON",
    "-DHDF5_BUILD_TOOLS:BOOL=OFF", "-DBUILD_TESTING:BOOL=OFF",
]
if ZLIB_ROOT:
    CMAKE_CONFIGURE_CMD += [
        "-DHDF5_ENABLE_Z_LIB_SUPPORT=ON",
        f"-DZLIB_INCLUDE_DIR={ZLIB_ROOT}\\include",
        f"-DZLIB_LIBRARY_RELEASE={ZLIB_ROOT}\\lib_release\\zlib.lib",
        f"-DZLIB_LIBRARY_DEBUG={ZLIB_ROOT}\\lib_debug\\zlibd.lib",
    ]
CMAKE_BUILD_CMD = ["cmake", "--build"]
CMAKE_INSTALL_ARG = ["--target", "install", '--config', 'Release']
CMAKE_INSTALL_PATH_ARG = "-DCMAKE_INSTALL_PREFIX={install_path}"
CMAKE_HDF5_LIBRARY_PREFIX = ["-DHDF5_EXTERNAL_LIB_PREFIX=h5py_"]
REL_PATH_TO_CMAKE_CFG = "hdf5-{version}"
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
}


def download_hdf5(version, outfile):
    series = '.'.join(version.split(".")[:2])
    file = HDF5_URL.format(version=version, series=series)

    print("Downloading " + file, file=stderr)
    r = requests.get(file, stream=True)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        print("Failed to download hdf5 version {version}, exiting".format(
            version=version
        ), file=stderr)
        exit(1)
    else:
        for chunk in r.iter_content(chunk_size=None):
            outfile.write(chunk)


def build_hdf5(version, hdf5_file, install_path, cmake_generator, use_prefix):
    try:
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
                    get_cmake_config_path(version, hdf5_extract_path),
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


def get_cmake_config_path(version, extract_point):
    return pjoin(extract_point, REL_PATH_TO_CMAKE_CFG.format(version=version))


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
            download_hdf5(version, f)
            build_hdf5(version, f, install_path, cmake_generator, use_prefix)
    else:
        print("using cached hdf5", file=stderr)
    if install_path is not None:
        print("hdf5 files: ", file=stderr)
        for dirpath, dirnames, filenames in walk(install_path):
            for file in filenames:
                print(" * " + pjoin(dirpath, file))


if __name__ == '__main__':
    main()
