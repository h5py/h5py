import shutil, zipfile, tempfile, glob, urllib
import os
import os.path as op

from paver.easy import *
from paver.path import pushd

# Directory containing pavement.py
ROOTPATH = op.dirname(op.abspath(__file__))

def archive(fname):
    """ Currently just copies a single file to the current directory """
    print "Archiving %s" % str(fname)
    if op.exists(op.join(ROOTPATH, fname)):
        os.remove(op.join(ROOTPATH, fname))
    shutil.copy(fname, ROOTPATH)


# --- Tasks to download and build HDF5 ----------------------------------------

CACHENAME = op.join(ROOTPATH, "cacheinit.cmake")

ZIPDIR = "HDF5-1.8.13"
ZIPFILE_NAME = "hdf5-1.8.13.zip"
ZIPFILE_URL = "http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.13/src/hdf5-1.8.13.zip"

def build(vs_version):
    """ Build HDF5.
    
    Intended to be called with os.cwd inside the root of the HDF5 checkout.
    This means a place containing a directory called "src".
    
    vs_version: Integer giving Visual Studio version (2008 or 2010).
    """
    build_dir = {2008: "build_2008", 2010: "build_2010"}[vs_version]
    vs_name = {2008: "Visual Studio 9 2008", 2010: "Visual Studio 10"}[vs_version]
    
    if not op.exists(build_dir):
        os.mkdir(build_dir)
        
    os.chdir(build_dir)
    
    sh('cmake -C "{}" -G "{}" ..'.format(CACHENAME, vs_name))
    sh('cmake --build . --config Release', ignore_error=True)
    sh('copy bin\Release\* bin /Y')
    sh('cmake --build . --config Release')
    sh('cpack -G ZIP')
    
    fname = glob.glob("HDF5-*.zip")[0]
    new_fname = 'hdf5-h5py-vs%d.zip' % vs_version
    if op.exists(new_fname):
        os.remove(new_fname)
    os.rename(fname, new_fname)
    
    archive(new_fname)
    

def check_zip():
    if not op.exists(ZIPFILE_NAME):
        print "Downloading HDF5 source code..."
        urllib.urlretrieve(ZIPFILE_URL, ZIPFILE_NAME)
    try:
        sh('cmake --version')
    except Exception:
        raise ValueError("CMake must be installed to build HDF5")
        
@task
def build_2008():
    """ Build HDF5 using Visual Studio 2008 """
    check_zip()
    if not op.exists(ZIPDIR):
        with zipfile.ZipFile(ZIPFILE_NAME) as z:
            z.extractall('.')
    with pushd(ZIPDIR):
        build(2008)
        
@task
def build_2010():
    """ Build HDF5 using Visual Studio 2010 """
    check_zip()
    if not op.exists(ZIPDIR):
        with zipfile.ZipFile(ZIPFILE_NAME) as z:
            z.extractall('.')
    with pushd(ZIPDIR):
        build(2010)
        
