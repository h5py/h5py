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


VERSION = '2.4.0'


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
Programming Language :: Python
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
  author_email = 'andrew dot collette at gmail dot com',
  maintainer = 'Andrew Collette',
  maintainer_email = 'andrew dot collette at gmail dot com',
  url = 'http://www.h5py.org',
  download_url = 'https://pypi.python.org/pypi/h5py',
  packages = ['h5py', 'h5py._hl', 'h5py.tests', 'h5py.tests.old', 'h5py.tests.hl'],
  package_data = package_data,
  ext_modules = [Extension('h5py.x',['x.c'])],  # To trick build into running build_ext
  requires = ['numpy (>=1.6.1)', 'Cython (>=0.17)'],
  install_requires = ['numpy>=1.6.1', 'Cython>=0.17', 'six'],
  setup_requires = ['pkgconfig', 'six'],
  cmdclass = CMDCLASS,
)
