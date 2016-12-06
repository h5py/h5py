
from os import environ, mkdir, walk, listdir, getcwd, chdir
from os.path import join as pjoin, exists
from tempfile import TemporaryFile, TemporaryDirectory
from sys import exit, stderr, platform
from shutil import copyfileobj, copy
from glob import glob
from subprocess import run, PIPE, STDOUT
from zipfile import ZipFile

import requests

HDF5_URL = "https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-{version}/src/"
HDF5_FILE = HDF5_URL + "hdf5-{version}.zip"
CMAKE_CONFIGURE_CMD = [
    "cmake", "-DBUILD_SHARED_LIBS:BOOL=ON", "-DCMAKE_BUILD_TYPE:STRING=RELEASE",
    "-DHDF5_BUILD_CPP_LIB=OFF", "-DHDF5_BUILD_HL_LIB=ON",
    "-DHDF5_BUILD_TOOLS:BOOL=ON",
]
CMAKE_BUILD_CMD = ["cmake", "--build"]
CMAKE_INSTALL_ARG = ["--target", "install", '--config', 'Release']
CMAKE_INSTALL_PATH_ARG = "-DCMAKE_INSTALL_PREFIX={install_path}"
REL_PATH_TO_CMAKE_CFG = "hdf5-{version}"
DEFAULT_VERSION = '1.8.17'

def download_hdf5(version, outfile):
    r = requests.get(HDF5_FILE.format(version=version), stream=True)
    try:
        r.raise_for_status()
        copyfileobj(r.raw, outfile)
    except requests.HTTPError:
        print("Failed to download hdf5 version {version}, exiting".format(
            version=version
        ), file=stderr)
        exit(1)


def build_hdf5(version, hdf5_file, install_path, vs_version):
    with TemporaryDirectory() as hdf5_extract_path:
        with ZipFile(hdf5_file) as z:
            z.extractall(hdf5_extract_path)
        old_dir = getcwd()
        with TemporaryDirectory() as new_dir:
            chdir(new_dir)
            cfg_cmd = CMAKE_CONFIGURE_CMD + [
                get_cmake_install_path(install_path),
                get_cmake_config_path(version, hdf5_extract_path),
            ] + ["-G", vs_version]
            build_cmd = CMAKE_BUILD_CMD + [
                '.',
            ] + CMAKE_INSTALL_ARG
            print("Configuring HDF5 version {version}...".format(version=version), file=stderr)
            print(' '.join(cfg_cmd), file=stderr)
            p = run(cfg_cmd, stdout=PIPE, stderr=STDOUT, universal_newlines=True)
            print(p.stdout)
            print("Building HDF5 version {version}...".format(version=version), file=stderr)
            print(' '.join(build_cmd), file=stderr)
            p = run(build_cmd, stdout=PIPE, stderr=STDOUT, universal_newlines=True)
            print(p.stdout)
            print("Installed HDF5 version {version} to {install_path}".format(
                version=version, install_path=install_path,
            ), file=stderr)
            chdir(old_dir)
    if platform.startswith('win'):
        for f in glob(pjoin(install_path, 'bin/*.dll')):
            copy(f, pjoin(install_path, 'lib'))


def get_cmake_config_path(version, extract_point):
    return pjoin(extract_point, REL_PATH_TO_CMAKE_CFG.format(version=version))


def get_cmake_install_path(install_path):
    if install_path is not None:
        return CMAKE_INSTALL_PATH_ARG.format(install_path=install_path)
    return ' '


def main():
    install_path = environ.get("HDF5_DIR")
    version = environ.get("HDF5_VERSION", DEFAULT_VERSION)
    vs_version = environ.get("HDF5_VISUAL_STUDIO_VERSION")
    if install_path is not None:
        if not exists(install_path):
            mkdir(install_path)
    if vs_version is None:
        raise RuntimeError("Visual Studio version not defined")
    with TemporaryFile() as f:
        download_hdf5(version, f)
        build_hdf5(version, f, install_path, vs_version)
    if install_path is not None:
        print("hdf5 files: ", file=stderr)
        for dirpath, dirnames, filenames in walk(install_path):
            for file in filenames:
                print(" * " + pjoin(dirpath, file))


if __name__ == '__main__':
    main()
