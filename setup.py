#!/usr/bin/env python

try:
    # If possible, use setuptools.Extension so we get setup_requires
    from setuptools import Extension, setup
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from distutils.cmd import Command
from distutils.dist import Distribution
from distutils.version import LooseVersion
import warnings
import sys, os
import os.path as op
from functools import reduce
import shutil

import configure   # Sticky-options configuration and version auto-detect
import setup_build

VERSION = '2.4.0a0'


if sys.version_info[0] >= 3:
    # Shamelessly stolen from Cython 0.14
    import lib2to3.refactor
    from distutils.command.build_py \
         import build_py_2to3 as build_py
else:
    from distutils.command.build_py import build_py


# --- Support functions and "super-option" configuring ------------------------

def localpath(*args):
    return op.abspath(reduce(op.join, (op.dirname(__file__),)+args))


# --- Determine configuration settings ----------------------------------------

settings = configure.scrape_eargs()          # lowest priority
settings.update(configure.scrape_cargs())    # highest priority

HDF5 = settings.get('hdf5')
HDF5_VERSION = settings.get('hdf5_version')

MPI = settings.setdefault('mpi', False)
if MPI:
    if not HAVE_CYTHON:
        raise ValueError("Cython is required to compile h5py in MPI mode")
    try:
        import mpi4py
    except ImportError:
        raise ImportError("mpi4py is required to compile h5py in MPI mode")


# --- Configure Cython and create extensions ----------------------------------




# --- Custom distutils commands -----------------------------------------------

class test(Command):

    """Run the test suite."""

    description = "Run the test suite"

    user_options = [('verbosity=', 'V', 'set test report verbosity')]

    def initialize_options(self):
        self.verbosity = 0

    def finalize_options(self):
        try:
            self.verbosity = int(self.verbosity)
        except ValueError:
            raise ValueError('verbosity must be an integer.')

    def run(self):
        import sys
        py_version = sys.version_info[:2]
        if py_version == (2,7) or py_version >= (3,2):
            import unittest
        else:
            try:
                import unittest2 as unittest
            except ImportError:
                raise ImportError(
                    "unittest2 is required to run tests with python-%d.%d"
                    % py_version
                    )
        buildobj = self.distribution.get_command_obj('build')
        buildobj.run()
        oldpath = sys.path
        try:
            sys.path = [op.abspath(buildobj.build_lib)] + oldpath
            suite = unittest.TestLoader().discover(op.join(buildobj.build_lib,'h5py'))
            result = unittest.TextTestRunner(verbosity=self.verbosity+1).run(suite)
            if not result.wasSuccessful():
                sys.exit(1)
        finally:
            sys.path = oldpath



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

Supports HDF5 versions 1.8.3 and higher.  On Windows, HDF5 is included with
the installer.
"""

if os.name == 'nt':
    package_data = {'h5py': ['*.dll']}
else:
    package_data = {'h5py': []}

# Avoid going off and installing NumPy if the user only queries for information
if any('--' + opt in sys.argv for opt in Distribution.display_option_names + ['help']):
    setup_requires = []
else:
    setup_requires = ['numpy >=1.0.1']

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
  download_url = 'http://code.google.com/p/h5py/downloads/list',
  packages = ['h5py', 'h5py._hl', 'h5py.tests'],
  package_data = package_data,
  ext_modules = [Extension('h5py.x',['x.c'])],  # To trick build into running build_ext
  requires = ['numpy (>=1.0.1)'],
  setup_requires = setup_requires,
  cmdclass = {'build_ext': setup_build.h5py_build_ext, 'test': test, 'build_py':build_py}
)
