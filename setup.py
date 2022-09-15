#!/usr/bin/env python

"""
    This is the main setup script for h5py (http://www.h5py.org).

    Most of the functionality is provided in two separate modules:
    setup_configure, which manages compile-time/Cython-time build options
    for h5py, and setup_build, which handles the actual compilation process.
"""

from setuptools import Extension, setup
import sys
import os

# Newer packaging standards may recommend removing the current dir from the
# path, add it back if needed.
if '' not in sys.path:
    sys.path.insert(0, '')

import setup_build, setup_configure


VERSION = '3.7.0'


# these are required to use h5py
RUN_REQUIRES = [
    # We only really aim to support NumPy & Python combinations for which
    # there are wheels on PyPI (e.g. NumPy >=1.17.5 for Python 3.8).
    # But we don't want to duplicate the information in oldest-supported-numpy
    # here, and if you can build an older NumPy on a newer Python, h5py probably
    # works (assuming you build it from source too).
    # NumPy 1.14.5 is the first with wheels for Python 3.7, our minimum Python.
    "numpy >=1.14.5",
]

# Packages needed to build h5py (in addition to static list in pyproject.toml)
# For packages we link to (numpy, mpi4py), we build against the oldest
# supported version; h5py wheels should then work with newer versions of these.
# Downstream packagers - e.g. Linux distros - can safely build with newer
# versions.
# TODO: setup_requires is deprecated in setuptools.
SETUP_REQUIRES = []

if setup_configure.mpi_enabled():
    RUN_REQUIRES.append('mpi4py >=3.0.2')
    SETUP_REQUIRES.append("mpi4py ==3.0.2; python_version<'3.8'")
    SETUP_REQUIRES.append("mpi4py ==3.0.3; python_version=='3.8.*'")
    SETUP_REQUIRES.append("mpi4py ==3.1.0; python_version>='3.9'")

# Set the environment variable H5PY_SETUP_REQUIRES=0 if we need to skip
# setup_requires for any reason.
if os.environ.get('H5PY_SETUP_REQUIRES', '1') == '0':
    SETUP_REQUIRES = []

# --- Custom Distutils commands -----------------------------------------------

CMDCLASS = {'build_ext': setup_build.h5py_build_ext}


# --- Distutils setup and metadata --------------------------------------------

cls_txt = \
"""
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Cython
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering
Topic :: Database
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: Unix
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
"""

short_desc = "Read and write HDF5 files from Python"

long_desc = \
"""
The h5py package provides both a high- and low-level interface to the HDF5
library from Python. The low-level interface is intended to be a complete
wrapping of the HDF5 API, while the high-level component supports  access to
HDF5 files, datasets and groups using established Python and NumPy concepts.

A strong emphasis on automatic conversion between Python (Numpy) datatypes and
data structures and their HDF5 equivalents vastly simplifies the process of
reading and writing data from Python.

Wheels are provided for several popular platforms, with an included copy of
the HDF5 library (usually the latest version when h5py is released).

You can also `build h5py from source
<https://docs.h5py.org/en/stable/build.html#source-installation>`_
with any HDF5 stable release from version 1.8.4 onwards, although naturally new
HDF5 versions released after this version of h5py may not work.
Odd-numbered minor versions of HDF5 (e.g. 1.13) are experimental, and may not
be supported.
"""

package_data = {'h5py': [], "h5py.tests.data_files": ["*.h5"]}
if os.name == 'nt':
    package_data['h5py'].append('*.dll')

setup(
  name = 'h5py',
  version = VERSION,
  description = short_desc,
  long_description = long_desc,
  classifiers = [x for x in cls_txt.split("\n") if x],
  author = 'Andrew Collette',
  author_email = 'andrew.collette@gmail.com',
  maintainer = 'Andrew Collette',
  maintainer_email = 'andrew.collette@gmail.com',
  license = 'BSD',
  url = 'https://www.h5py.org',
  project_urls = {
      'Source': 'https://github.com/h5py/h5py',
      'Documentation': 'https://docs.h5py.org/en/stable/',
      'Release notes': 'https://docs.h5py.org/en/stable/whatsnew/index.html'
  },
  packages = [
      'h5py',
      'h5py._hl',
      'h5py.tests',
      'h5py.tests.data_files',
      'h5py.tests.test_vds',
  ],
  package_data = package_data,
  ext_modules = [Extension('h5py.x',['x.c'])],  # To trick build into running build_ext
  install_requires = RUN_REQUIRES,
  setup_requires = SETUP_REQUIRES,
  python_requires='>=3.7',
  cmdclass = CMDCLASS,
)
