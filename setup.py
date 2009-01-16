#!/usr/bin/env python

#+
# 
# This file is part of h5py, a low-level Python interface to the HDF5 library.
# 
# Copyright (C) 2008 Andrew Collette
# http://h5py.alfven.org
# License: BSD  (See LICENSE.txt for full license)
# 
# $Date$
# 
#-

"""
    Setup script for the h5py package.  

    * Quick install:

      python setup.py build [--api=<16|18>] [--hdf5=/path/to/hdf5]
      [sudo] python setup.py install

    * Full rebuild (i.e. if checked out from trunk):

      python setup.py cython build [--api=<16|18>] [--hdf5=/path/to/hdf5]
      [sudo] python setup.py install

    New commands:

    * cython [--api16] [--api18] [--diag]
"""

import os
import sys
import shutil
import commands
import os.path as op

# Basic package options
NAME = 'h5py'               # Software title
VERSION = '1.1.0'
MIN_NUMPY = '1.0.3'
MIN_CYTHON = '0.9.8.1.1'
SRC_PATH = 'h5py'           # Name of directory with .pyx files

USE_DISTUTILS = False

MODULES = ['h5', 'h5f', 'h5g', 'h5s', 'h5t', 'h5d', 'h5a', 'h5p', 'h5z',
                 'h5i', 'h5r', 'h5fd', 'utils', 'h5o', 'h5l']
EXTRA_SRC = {'h5': ["lzf_filter.c", "lzf/lzf_c.c", "lzf/lzf_d.c"]}

def version_check(vers, required):
    """ Compare versions between two "."-separated strings. """

    def tpl(istr):
        return tuple(int(x) for x in istr.split('.'))

    return tpl(vers) >= tpl(required)

def fatal(instring, code=1):
    print >> sys.stderr, "Fatal: "+instring
    exit(code)

def warn(instring):
    print >> sys.stderr, "Warning: "+instring

def debug(instring):
    if DEBUG:
        print " DEBUG: "+instring

def localpath(*args):
    return op.abspath(reduce(op.join, (op.dirname(__file__),)+args))


# --- Imports -----------------------------------------------------------------

# Evil test options for setup.py
DEBUG = False
for arg in sys.argv[:]:
    if arg.find('--disable-numpy') == 0:
        sys.argv.remove(arg)
        sys.modules['numpy'] = None
    if arg.find('--disable-cython') == 0:
        sys.argv.remove(arg)
        sys.modules['Cython'] = None
    if arg.find('--use-distutils') == 0:
        sys.argv.remove(arg)
        USE_DISTUTILS = True
    if arg.find('--setup-debug') == 0:
        sys.argv.remove(arg)
        DEBUG = True

# Check Python version (2.5 or greater required)
if not (sys.version_info[0:2] >= (2,5)):
    fatal("At least Python 2.5 is required to install h5py")

# Check for Numpy (required)
try:
    import numpy
    if numpy.version.version < MIN_NUMPY:
        fatal("Numpy version %s is out of date (>= %s needed)" % (numpy.version.version, MIN_NUMPY))
except ImportError:
    fatal("Numpy not installed (version >= %s required)" % MIN_NUMPY)

if not USE_DISTUTILS:
    try:
        import ez_setup
        ez_setup.use_setuptools(download_delay=0)
        USE_DISTUTILS = False
    except Exception, e:
        debug("Setuptools import FAILED: %s" % str(e))
        USE_DISTUTILS = True
    else:
        debug("Using setuptools")
        from setuptools import setup
        HAVE_SETUPTOOLS = True

if USE_DISTUTILS:
    debug("Using distutils")
    from distutils.core import setup
    HAVE_SETUPTOOLS = False

from distutils.errors import DistutilsError
from distutils.extension import Extension
from distutils.command.build import build 
from distutils.command.build_ext import build_ext
from distutils.cmd import Command


# --- Compiler and library config ---------------------------------------------

class GlobalOpts:

    def __init__(self):
        self.hdf5 = None
        self.api = None

    def parse_argv(self):

        for arg in sys.argv[:]:
            if arg.find('--hdf5=') == 0:
                self.hdf5 = arg.partition('=')[2]
                if not op.isdir(self.hdf5):
                    fatal('Invalid HDF5 directory "%s"' % self.hdf5)
                sys.argv.remove(arg)
            if arg.find('--api=') == 0:
                self.api = arg.partition('=')[2]
                try:
                    self.api = int(self.api)
                    if self.api not in (16,18):
                        raise Exception
                except Exception:
                    fatal('Illegal option %s to --api= (legal values are 16,18)' % self.api)
                sys.argv.remove(arg)

    def get_api_version(self):
        """ Get the active HDF5 version, from the command line or by
            trying to run showconfig.
        """
        if self.api is not None:
            return self.api

        if self.hdf5 is not None:
            cmd = reduce(op.join, (self.hdf5, 'bin', 'h5cc'))+" -showconfig"
        else:
            cmd = "h5cc -showconfig"
        output = commands.getoutput(cmd)
        l = output.find("HDF5 Version")

        if l > 0:
            if output[l:l+30].find('1.8') > 0:
                debug("Autodetected HDF5 1.8")
                return 18
            elif output[l:l+30].find('1.6') > 0:
                debug("Autodetected HDF5 1.6")
                return 16

        debug("Autodetect FAILED")
        warn("Can't determine HDF5 version, assuming 1.6 (use --api= to override)")
        return 16

class ExtensionCreator(object):

    """ Figures out what include/library dirs are appropriate, and
        serves as a factory for Extension instances.

        Note this is a purely C-oriented process; it doesn't know or
        care about Cython.
    """

    def __init__(self, hdf5_loc=None):
        if os.name == 'nt':
            if hdf5_loc is None:
                fatal("On Windows, HDF5 directory must be specified.")

            self.libraries = ['hdf5dll18']
            self.include_dirs = [numpy.get_include(), op.join(hdf5_loc, 'include'), op.abspath('win_include')]
            self.library_dirs = [op.join(hdf5_loc, 'dll')]
            self.runtime_dirs = []
            self.extra_compile_args = ['/DH5_USE_16_API', '/D_HDF5USEDLL_']
            self.extra_link_args = []

        else:
            self.libraries = ['hdf5']
            if hdf5_loc is None:
                self.include_dirs = [numpy.get_include(), '/usr/include', '/usr/local/include']
                self.library_dirs = ['/usr/lib/', '/usr/local/lib']
            else:
                self.include_dirs = [numpy.get_include(), op.join(hdf5_loc, 'include')]
                self.library_dirs = [op.join(hdf5_loc, 'lib')]
            self.runtime_dirs = self.library_dirs
            self.extra_compile_args = ['-DH5_USE_16_API', '-Wno-unused', '-Wno-uninitialized']
            self.extra_link_args = []

    
    def create_extension(self, name, extra_src=None):
        """ Create a distutils Extension object for the given module.  A list
            of C source files to be included in the compilation can also be
            provided.
        """
        if extra_src is None:
            extra_src = []
        sources = [op.join(SRC_PATH, name+'.c')]+[op.join(SRC_PATH,x) for x in extra_src]
        return Extension(NAME+'.'+name,
                            sources, 
                            include_dirs = self.include_dirs, 
                            libraries = self.libraries,
                            library_dirs = self.library_dirs,
                            runtime_library_dirs = self.runtime_dirs,
                            extra_compile_args = self.extra_compile_args,
                            extra_link_args = self.extra_link_args)

GLOBALOPTS = GlobalOpts()
GLOBALOPTS.parse_argv()

creator = ExtensionCreator(GLOBALOPTS.hdf5)
EXTENSIONS = [creator.create_extension(x, EXTRA_SRC.get(x, None)) for x in MODULES]


# --- Custom extensions for distutils -----------------------------------------

class cython(Command):

    """ Cython pre-builder """

    user_options = [('diag', 'd', 'Enable library debug logging'),
                    ('api16', '6', 'Only build version 1.6'),
                    ('api18', '8', 'Only build version 1.8'),
                    ('force', 'f', 'Bypass timestamp checking'),
                    ('clean', 'c', 'Clean up Cython files')]

    boolean_options = ['diag', 'force', 'clean']

    def initialize_options(self):
        self.diag = None
        self.api16 = None
        self.api18 = None
        self.force = False
        self.clean = False

    def finalize_options(self):
        if not (self.api16 or self.api18):
            self.api16 = self.api18 = True

    def checkdir(self, path):
        if not op.isdir(path):
            os.mkdir(path)

    def run(self):
        
        if self.clean:
            for path in [localpath(x) for x in ('api16','api18')]:
                shutil.rmtree(path)
            return

        try:
            from Cython.Compiler.Main import Version, compile, compile_multiple, CompilationOptions
            if not version_check(Version.version, MIN_CYTHON):
                fatal("Old Cython %s version detected; at least %s required" % (Version.version, MIN_CYTHON))
        except ImportError:
            fatal("Cython (http://cython.org) is required to rebuild h5py")

        print "Rebuilding Cython files (this may take a few minutes)..."

        def cythonize(api, diag):

            outpath = localpath('api%d' % api)
            self.checkdir(outpath)

            pxi_str = \
"""# This file is automatically generated.  Do not edit.

DEF H5PY_VERSION = "%(VERSION)s"

DEF H5PY_API = %(API_MAX)d     # Highest API level (i.e. 18 or 16)
DEF H5PY_16API = %(API_16)d    # 1.6.X API available (always true, for now)
DEF H5PY_18API = %(API_18)d    # 1.8.X API available

DEF H5PY_DEBUG = %(DEBUG)d    # Logging-level number, or 0 to disable
"""
            pxi_str %= {"VERSION": VERSION, "API_MAX": api,
                        "API_16": True, "API_18": api == 18,
                        "DEBUG": 10 if diag else 0}

            f = open(op.join(outpath, 'config.pxi'),'w')
            f.write(pxi_str)
            f.close()

            debug("  Cython: %s" % Version.version)
            debug("  API level: %d" % api)
            debug("  Diagnostic mode: %s" % ('yes' if diag else 'no'))

            for module in MODULES:

                pyx_path = localpath(SRC_PATH, module+'.pyx')
                c_path = localpath(outpath, module+'.c')

                if self.force or \
                not op.exists(c_path) or \
                os.stat(pyx_path).st_mtime > os.stat(c_path).st_mtime:

                    debug("Cythoning %s" % pyx_path)
                    result = compile(pyx_path, verbose=False,
                                     include_path=[outpath], output_file=c_path)
                    if result.num_errors != 0:
                        fatal("Cython error; aborting.")

        # end "def cythonize(...)"

        if self.api16:
            cythonize(16, self.diag)
        if self.api18:
            cythonize(18, self.diag)

class hbuild(build):

    user_options = build.user_options + \
                    [('hdf5=', '5', 'Custom location for HDF5'),
                     ('api=', 'a', 'Set API levels (--api=16 or --api=18)')]

    boolean_options = build.boolean_options 

    def initialize_options(self):
        build.initialize_options(self)
        self.hdf5 = None
        self.api = None

    def finalize_options(self):
        build.finalize_options(self)

class hbuild_ext(build_ext):

    user_options = build_ext.user_options + \
                    [('hdf5=', '5', 'Custom location for HDF5'),
                     ('api=', 'a', 'Set API levels (--api=16 or --api=18)')]

    boolean_options = build_ext.boolean_options 

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.hdf5 = None
        self.api = None

    def finalize_options(self):
        build_ext.finalize_options(self)

    def run(self):

        api = GLOBALOPTS.get_api_version()

        c_path = localpath('api%d' % api)
        
        if not all(op.exists(op.join(c_path, x+'.c')) for x in MODULES):
            fatal("Cython rebuild required ('python setup.py cython')")
        
        for x in MODULES:
            src = op.join(c_path, x+'.c')
            dst = localpath(SRC_PATH)
            debug("Copying %s -> %s" % (src, dst))
            shutil.copy(src, dst)

        build_ext.run(self)

class test_stub(Command):

    user_options = []
    boolean_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass

    def run(self):
        fatal("Testing only available with setuptools")

CMD_CLASS = {'build': hbuild, 'cython': cython, 'build_ext': hbuild_ext}

if not HAVE_SETUPTOOLS:
    CMD_CLASS.update({'test': test_stub})


# --- Setup parameters --------------------------------------------------------

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

Supports HDF5 versions 1.6.5 through 1.8.2.  On Windows, HDF5 is included in
the installer.
"""

if os.name == 'nt':
    package_data = {'h5py': ['*.pyx', '*.dll'],
                       'h5py.tests': ['data/*.hdf5']}
else:
    package_data = {'h5py': ['*.pyx'],
                   'h5py.tests': ['data/*.hdf5']}

setup(
  name = NAME,
  version = VERSION,
  description = short_desc,
  long_description = long_desc,
  classifiers = [x for x in cls_txt.split("\n") if x],
  author = 'Andrew Collette',
  author_email = '"h5py" at the domain "alfven.org"',
  maintainer = 'Andrew Collette',
  maintainer_email = '"h5py" at the domain "alfven.org"',
  url = 'http://h5py.alfven.org',
  packages = ['h5py','h5py.tests'],
  package_data = package_data,
  ext_modules = EXTENSIONS,
  requires = ['numpy (>=%s)' % MIN_NUMPY],
  cmdclass = CMD_CLASS
)



