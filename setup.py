from distutils.core import setup
from distutils.extension import Extension
from distutils.cmd import Command
import sys, os
import os.path as op
from functools import reduce

try:
    import Cython.Compiler.Version
    vers = tuple(int(x.rstrip('+')) for
                 x in Cython.Compiler.Version.version.split('.'))
    if vers < (0,13):
        raise ImportError
    from Cython.Distutils import build_ext
    SUFFIX = '.pyx'
except ImportError:
    from distutils.command.build_ext import build_ext
    SUFFIX = '.c'

import numpy

import configure

VERSION = '2.2.0a1'

def localpath(*args):
    return op.abspath(reduce(op.join, (op.dirname(__file__),)+args))

if sys.version_info[0] >= 3:
    # Shamelessly stolen from Cython 0.14
    import lib2to3.refactor
    from distutils.command.build_py \
         import build_py_2to3 as build_py
else:
    from distutils.command.build_py import build_py

# --- Determine HDF5 location -------------------------------------------------

settings = configure.scrape_eargs()          # lowest priority
settings.update(configure.scrape_cargs())    # highest priority

HDF5 = settings.get('hdf5')


# --- Create extensions -------------------------------------------------------

if sys.platform.startswith('win'):
    COMPILER_SETTINGS = {
        'libraries'     : ['hdf5dll18','hdf5_hldll'],
        'include_dirs'  : [numpy.get_include(),  localpath('lzf'),
                           localpath('win_include')],
        'library_dirs'  : [],
        'define_macros' : [('H5_USE_16_API', None), ('_HDF5USEDLL_', None)]
    }
    if HDF5 is not None:
        COMPILER_SETTINGS['include_dirs'] += [op.join(HDF5, 'include')]
        COMPILER_SETTINGS['library_dirs'] += [op.join(HDF5, 'dll')]
else:
    COMPILER_SETTINGS = {
       'libraries'      : ['hdf5', 'hdf5_hl'],
       'include_dirs'   : [numpy.get_include(), localpath('lzf')],
       'library_dirs'   : [],
       'define_macros'  : [('H5_USE_16_API', None)]
    }
    if HDF5 is not None:
        COMPILER_SETTINGS['include_dirs'] += [op.join(HDF5, 'include')]
        COMPILER_SETTINGS['library_dirs'] += [op.join(HDF5, 'lib')]
    elif sys.platform == 'darwin':
        COMPILER_SETTINGS['include_dirs'] += ['/opt/local/include']
        COMPILER_SETTINGS['library_dirs'] += ['/opt/local/lib']
    COMPILER_SETTINGS['runtime_library_dirs'] = [op.abspath(x) for x in COMPILER_SETTINGS['library_dirs']]

MODULES =  ['defs','_errors','_objects','_proxy', 'h5fd', 'h5z',
            'h5','h5i','h5r','utils',
            '_conv', 'h5t','h5s',
            'h5p',
            'h5d', 'h5a', 'h5f', 'h5g',
            'h5l', 'h5o',
            'h5ds']

EXTRA_SRC = {'h5z': [ localpath("lzf/lzf_filter.c"),
                      localpath("lzf/lzf/lzf_c.c"),
                      localpath("lzf/lzf/lzf_d.c")]}

def make_extension(module):
    sources = [op.join('h5py', module+SUFFIX)] + EXTRA_SRC.get(module, [])
    return Extension('h5py.'+module, sources, **COMPILER_SETTINGS)

EXTENSIONS = [make_extension(m) for m in MODULES]

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

Supports HDF5 versions 1.8.2 and higher.  On Windows, HDF5 is included with
the installer.
"""

if os.name == 'nt':
    package_data = {'h5py': ['*.pyx', '*.dll']}
else:
    package_data = {'h5py': ['*.pyx']}

setup(
  name = 'h5py',
  version = VERSION,
  description = short_desc,
  long_description = long_desc,
  classifiers = [x for x in cls_txt.split("\n") if x],
  author = 'Andrew Collette',
  author_email = '"h5py" at the domain "alfven.org"',
  maintainer = 'Andrew Collette',
  maintainer_email = '"h5py" at the domain "alfven.org"',
  url = 'http://h5py.alfven.org',
  download_url = 'http://code.google.com/p/h5py/downloads/list',
  packages = ['h5py', 'h5py._hl', 'h5py._hl.tests', 'h5py.lowtest'],
  package_data = package_data,
  ext_modules = EXTENSIONS,
  requires = ['numpy (>=1.0.1)'],
  cmdclass = {'build_ext': build_ext, 'test': test, 'build_py':build_py}
)


