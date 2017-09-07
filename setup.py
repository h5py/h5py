#!/usr/bin/env python

"""
    This is the main setup script for h5py (http://www.h5py.org).

    Most of the functionality is provided in two separate modules:
    setup_configure, which manages compile-time/Cython-time build options
    for h5py, and setup_build, which handles the actual compilation process.
"""

try:
    from setuptools import Extension, setup
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from distutils.cmd import Command
from distutils.dist import Distribution
import sys
import os
import os.path as op

import setup_build, setup_configure


VERSION = '2.7.1.post0'

NUMPY_DEP = 'numpy>=1.7'

# these are required to use h5py
RUN_REQUIRES = [NUMPY_DEP, 'six']

# these are required to build h5py
# RUN_REQUIRES is included as setup.py test needs RUN_REQUIRES for testing
# RUN_REQUIRES can be removed when setup.py test is removed
SETUP_REQUIRES = RUN_REQUIRES + [NUMPY_DEP, 'Cython>=0.19', 'pkgconfig']

# Needed to avoid trying to install numpy/cython on pythons which the latest
# versions don't support
if ("sdist" in sys.argv and "bdist_wheel" not in sys.argv and
    "install" not in sys.argv) or "check" in sys.argv:
    use_setup_requires = False
else:
    use_setup_requires = True


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
        py_version = sys.version_info[:2]
        if py_version != (2, 6):
            import unittest
        else:
            try:
                import unittest2 as unittest
            except ImportError:
                raise ImportError( "unittest2 is required to run tests with Python 2.6")

        buildobj = self.distribution.get_command_obj('build')
        buildobj.run()

        oldpath = sys.path
        try:
            sys.path = [op.abspath(buildobj.build_lib)] + oldpath
            import h5py
            result = h5py.run_tests(verbose=self.detail)
            if not result.wasSuccessful():
                sys.exit(1)
        finally:
            sys.path = oldpath


CMDCLASS = {'build_ext': setup_build.h5py_build_ext,
            'configure': setup_configure.configure,
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
Programming Language :: Python :: 2
Programming Language :: Python :: 2.6
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.3
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
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

if os.name == 'nt':
    package_data = {'h5py': ['*.dll']}
else:
    package_data = {'h5py': []}

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
  url = 'http://www.h5py.org',
  download_url = 'https://pypi.python.org/pypi/h5py',
  packages = ['h5py', 'h5py._hl', 'h5py.tests', 'h5py.tests.old', 'h5py.tests.hl'],
  package_data = package_data,
  ext_modules = [Extension('h5py.x',['x.c'])],  # To trick build into running build_ext
  install_requires = RUN_REQUIRES,
  setup_requires = SETUP_REQUIRES if use_setup_requires else [],
  cmdclass = CMDCLASS,
)
