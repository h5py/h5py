#!/usr/bin/env python

"""
    This is the main setup script for h5py (http://www.h5py.org).

    Most of the functionality is provided in two separate modules:
    setup_configure, which manages compile-time/Cython-time build options
    for h5py, and setup_build, which handles the actual compilation process.
"""

from setuptools import Extension, setup
from distutils.cmd import Command
import sys
import os
import os.path as op

# Newer packaging standards may recommend removing the current dir from the
# path, add it back if needed.
if '' not in sys.path:
    sys.path.insert(0, '')

import setup_build, setup_configure


VERSION = '3.4.0'

# Minimum supported versions of Numpy & Cython depend on the Python version
NUMPY_MIN_VERSIONS = [
    # Numpy    Python
    ('1.14.5', "=='3.7'"),
    ('1.17.5', "=='3.8'"),
    ('1.19.3', ">='3.9'"),
]

# these are required to use h5py
RUN_REQUIRES = ["cached-property; python_version<'3.8'"] + [
    f"numpy >={np_min}; python_version{py_condition}"
    for np_min, py_condition in NUMPY_MIN_VERSIONS
]

# these are required to build h5py
# For packages we link to (numpy, mpi4py), we build against the oldest
# supported version; h5py wheels should then work with newer versions of these.
# Downstream packagers - e.g. Linux distros - can safely build with newer
# versions.
SETUP_REQUIRES = [
    'pkgconfig',
    "Cython >=0.29; python_version<'3.8'",
    "Cython >=0.29.14; python_version=='3.8'",
    "Cython >=0.29.15; python_version>='3.9'",
] + [
    f"numpy =={np_min}; python_version{py_condition}"
    for np_min, py_condition in NUMPY_MIN_VERSIONS
]

if setup_configure.mpi_enabled():
    RUN_REQUIRES.append('mpi4py >=3.0.2')
    SETUP_REQUIRES.append("mpi4py ==3.0.2; python_version<'3.8'")
    SETUP_REQUIRES.append("mpi4py ==3.0.3; python_version>='3.8'")

# Set the environment variable H5PY_SETUP_REQUIRES=0 if we need to skip
# setup_requires for any reason.
if os.environ.get('H5PY_SETUP_REQUIRES', '1') == '0':
    SETUP_REQUIRES = []

# --- Custom Distutils commands -----------------------------------------------

class test(Command):

    """
        Custom Distutils command to run the h5py test suite.

        This command will invoke build/build_ext if the project has not
        already been built.  It then patches in the build directory to
        sys.path and runs the test suite directly.
    """

    description = "Run the test suite"

    user_options = [('detail', 'd', 'Display additional test information')]

    def initialize_options(self):
        self.detail = False

    def finalize_options(self):
        self.detail = bool(self.detail)

    def run(self):
        """ Called by Distutils when this command is run """
        import sys

        buildobj = self.distribution.get_command_obj('build')
        buildobj.run()

        oldpath = sys.path
        oldcwd = os.getcwd()
        build_lib_dir = op.abspath(buildobj.build_lib)
        try:
            sys.path = [build_lib_dir] + oldpath
            os.chdir(build_lib_dir)

            import h5py
            sys.exit(h5py.run_tests())
        finally:
            sys.path = oldpath
            os.chdir(oldcwd)


CMDCLASS = {'build_ext': setup_build.h5py_build_ext,
            'test': test, }


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

Supports HDF5 versions 1.8.4 and higher.  On Windows, HDF5 is included with
the installer.
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
  url = 'http://www.h5py.org',
  download_url = 'https://pypi.python.org/pypi/h5py',
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
