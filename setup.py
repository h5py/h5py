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

from __future__ import with_statement

"""
    Setup script for the h5py package.  

    All commands take the usual distutils options, like --home, etc.  Cython is
    not required for installation, but will be invoked if one of the --cython
    options is used, or if non-default options are specified for the build.

    To build:
    python setup.py build [--help for additional options]

    To install:
    sudo python setup.py install

    To run the test suite locally (won't install anything):
    python setup.py test
"""

import os
import sys
import shutil
import commands
import pickle
import os.path as op

from distutils.errors import DistutilsError
from distutils.core import setup
from distutils.extension import Extension
from distutils.command.build import build 
from distutils.command.clean import clean
from distutils.command.sdist import sdist
from distutils.cmd import Command

# Basic package options
NAME = 'h5py'               # Software title
VERSION = '1.1.0'
MIN_NUMPY = '1.0.3'
MIN_CYTHON = '0.9.8.1.1'
SRC_PATH = 'h5py'           # Name of directory with .pyx files
HAVE_CYTHON = False         # Flag (updated below in "Import Checks")
CMD_CLASS = {}              # Custom command classes for setup()
PICKLE_FILE = 'buildconf.pickle'

# The list of modules depends on max API version
MODULES = {16:  ['h5', 'h5f', 'h5g', 'h5s', 'h5t', 'h5d', 'h5a', 'h5p', 'h5z',
                 'h5i', 'h5r', 'h5fd', 'utils'],
           18:  ['h5', 'h5f', 'h5g', 'h5s', 'h5t', 'h5d', 'h5a', 'h5p', 'h5z',
                 'h5i', 'h5r', 'h5fd', 'utils', 'h5o', 'h5l']}
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


# === Import Checks ===========================================================

# Evil test options for setup.py
for arg in sys.argv[:]:
    if arg.find('--disable-numpy') == 0:
        sys.argv.remove(arg)
        sys.modules['numpy'] = None
    if arg.find('--disable-cython') == 0:
        sys.argv.remove(arg)
        sys.modules['Cython'] = None

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

try:
    from Cython.Compiler.Main import Version, compile, compile_multiple, CompilationOptions
    if version_check(Version.version, MIN_CYTHON):
        HAVE_CYTHON = True
    else:
        HAVE_CYTHON = False
        warn("Old Cython %s version ignored; at least %s required" % (Version.version, MIN_CYTHON))
except ImportError:
    HAVE_CYTHON = False
    warn("Cython (http://cython.org) is not available; only default build possible")

# === Platform-dependent compiler config ======================================


class ExtensionCreator(object):

    """ Figures out what include/library dirs are appropriate, and
        serves as a factory for Extension instances.  This is in a
        class as opposed to module code since the HDF5 location
        isn't known until runtime.

        Note this is a purely C-oriented process; it doesn't know or
        care about Cython.
    """

    def __init__(self, hdf5_loc=None):
        if os.name == 'nt':
            if hdf5_loc is None:
                fatal("On Windows, HDF5 directory must be specified.")

            self.libraries = ['hdf5dll']
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


# === Custom extensions for distutils =========================================

class cybuild(build):

    """ Cython-aware builder.
    """

    user_options = build.user_options + \
                    [('hdf5=', '5', 'Custom location for HDF5'),
                     ('api=', 'a', 'Set API levels (--api=16 or --api=18)'),
                     ('cython','y','Run Cython'),
                     ('cython-only','Y', 'Run Cython and stop'),
                     ('diag', 'd','Enable library debug logging'),
                     ('no-threads', 't', 'Build without thread support')]

    boolean_options = build.boolean_options + ['cython', 'cython-only', 'no-threads','diag']

    def initialize_options(self):
        build.initialize_options(self)

        # Build options
        self.hdf5 = None
        self.api = None

        # Cython (config) options
        self.cython = False
        self.cython_only = False
        self.diag = False
        self.no_threads = False


    def finalize_options(self):

        build.finalize_options(self)

        if self.no_threads:
            warn("Option --no-threads will disappear soon")

        if self.hdf5 is not None:
            self.hdf5 = op.abspath(self.hdf5)
            if not op.exists(self.hdf5):
                fatal('Specified HDF5 directory "%s" does not exist' % self.hdf5)

        if self.api is not None:
            try:
                self.api = int(self.api)
                if self.api not in (16,18):
                    raise Exception
            except Exception:
                fatal('Illegal option %s to --api= (legal values are 16,18)' % self.api)

    def run(self):

        # Hack to prevent "install" command from running Cython
        if not 'build' in sys.argv:
            build.run(self)
            return

        if self.api is None:
            self.api = self.get_hdf5_version()  # either 16 or 18

        modules = MODULES[self.api]
        creator = ExtensionCreator(self.hdf5)
        extensions = [creator.create_extension(x, EXTRA_SRC.get(x, None)) for x in modules]

        self.distribution.ext_modules = extensions

        # Cython must be run if any of the following are true:
        # 1. Some of the .c files don't exist
        # 2. The current config options don't match the last used options
        # 3. Either option --cython or --cython-only is provided

        src_missing = not all(op.exists(op.join(SRC_PATH, x+'.c')) for x in modules)
        pxi = self.generate_pxi()
        stale_pxi = pxi != self.read_pxi()

        if any((src_missing, stale_pxi, self.cython, self.cython_only)):
            if not HAVE_CYTHON:
                fatal("Cython recompilation required, but Cython is unavailable or out of date")
            if stale_pxi:
                self.write_pxi(pxi)  # Do this AFTER the Cython check
            self.compile_cython(sorted(modules), stale_pxi)
            if self.cython_only:
                exit(0)

        # Hand over control to distutils
        build.run(self)

        # For commands test and doc, which need to know about this build
        with open(PICKLE_FILE,'w') as f:
            pickle.dump((modules, extensions, self.build_lib), f)

    def get_hdf5_version(self):
        """ Try to determine the installed HDF5 version.

            Returns either 16 or 18.  Defaults to 16 (and prints a warning)
            if the installed version can't be identified.
        """
        if self.hdf5 is not None:
            cmd = reduce(op.join, (self.hdf5, 'bin', 'h5cc'))+" -showconfig"
        else:
            cmd = "h5cc -showconfig"
        output = commands.getoutput(cmd)
        l = output.find("HDF5 Version")

        if l > 0:
            if output[l:l+30].find('1.8') > 0:
                return 18
            elif output[l:l+30].find('1.6') > 0:
                return 16

        warn("Can't determine HDF5 version, assuming 1.6 (use --api= to override)")
        return 16

    def generate_pxi(self):
        """ Generate a Cython .pxi file reflecting the current options. """

        pxi_str = \
"""# This file is automatically generated.  Do not edit.

DEF H5PY_VERSION = "%(VERSION)s"

DEF H5PY_API = %(API_MAX)d     # Highest API level (i.e. 18 or 16)
DEF H5PY_16API = %(API_16)d    # 1.6.X API available (always true, for now)
DEF H5PY_18API = %(API_18)d    # 1.8.X API available

DEF H5PY_DEBUG = %(DEBUG)d    # Logging-level number, or 0 to disable

DEF H5PY_THREADS = %(THREADS)d  # Enable thread-safety and non-blocking reads
"""
        return pxi_str % {"VERSION": VERSION, "API_MAX": self.api,
                    "API_16": True, "API_18": self.api == 18,
                    "DEBUG": 10 if self.diag else 0, "THREADS": not self.no_threads}

    def read_pxi(self):
        """ Returns the current config.pxi file, or an empty string. """

        pxi_path = op.join(SRC_PATH, 'config.pxi')
        if not op.exists(pxi_path):
            return ""

        try:
            f = open(pxi_path, 'r')
            return f.read()
        except (IOError, OSError):
            fatal("Can't read file %s" % pxi_path)
        else:
            f.close()
    
    def write_pxi(self, pxi):
        """ Unconditionally overwrite the config.pxi file """

        pxi_path = op.join(SRC_PATH, 'config.pxi')
        try:
            f = open(pxi_path, 'w')
            f.write(pxi)
            f.close()
        except IOError:
            fatal('Failed write to "%s"' % pxi_path)

    def compile_cython(self, modules, recompile_all=False):
        """ Regenerate the C source files for the build process.
        """

        print "Running Cython (%s)..." % Version.version
        print "  API level: %d" % self.api
        print "  Thread-aware: %s" % ('yes' if not self.no_threads else 'no')
        print "  Diagnostic mode: %s" % ('yes' if self.diag else 'no')
        print "  HDF5: %s" % ('default' if self.hdf5 is None else self.hdf5)

        # Build each extension
        # This should be a single call to compile_multiple, but it's
        # broken in Cython 0.9.8.1.1
        if 1:
            cyopts = CompilationOptions(verbose=False)
            for module in modules:
                pyx_path = op.join(SRC_PATH,module+'.pyx')
                c_path = op.join(SRC_PATH,module+'.c')
                if not op.exists(c_path) or \
                   os.stat(pyx_path).st_mtime > os.stat(c_path).st_mtime or \
                   recompile_all or\
                   self.force:
                    print "Cythoning %s" % pyx_path
                    result = compile(pyx_path, cyopts)
                    if result.num_errors != 0:
                        fatal("Cython error; aborting.")
        else:
            cyopts = CompilationOptions(verbose=True, timestamps=True)
            modpaths = [op.join(SRC_PATH, x+'.pyx') for x in modules]
            result = compile_multiple(modpaths, cyopts)
            if result.num_errors != 0:
                fatal("%d Cython errors; aborting" % result.num_errors)

class test(Command):

    """ Run unit tests """

    description = "Run unit tests in-place"
    user_options = [('sections=','s','Comma separated list of tests ("-" prefix to NOT run)'),
                    ('detail=', 'd', 'Level of output detail (0-3, default 1)')]

    def initialize_options(self):
        self.sections = None
        self.output = False
        self.detail = 1

    def finalize_options(self):
        self.detail = int(self.detail)

    def run(self):

        try:
            with open(PICKLE_FILE, 'r') as f:
                modules, extensions, build_path = pickle.load(f)
        except (IOError, OSError):
            fatal("Project must be built before tests can be run")

        oldpath = sys.path
        try:
            sys.path = [op.abspath(build_path)] + oldpath
            import h5py.tests
            if not h5py.tests.runtests(None if self.sections is None else tuple(self.sections.split(',')), self.detail):
                raise DistutilsError("Unit tests failed.")
        finally:
            sys.path = oldpath

class doc(Command):

    """ Regenerate documentation.  Unix only, requires epydoc/sphinx. """

    description = "Rebuild documentation"

    user_options = [('rebuild', 'r', "Rebuild from scratch")]
    boolean_options = ['rebuild']

    def initialize_options(self):
        self.rebuild = False

    def finalize_options(self):
        pass

    def run(self):

        try:
            with open(PICKLE_FILE, 'r') as f:
                modules, extensions, pth = pickle.load(f)
        except (IOError, OSError):
            fatal("Project must be built before docs can be compiled")

        if self.rebuild and op.exists('docs/build'):
            shutil.rmtree('docs/build')

        cmd = "export H5PY_PATH=%s; cd docs; make html" % pth

        retval = os.system(cmd)
        if retval != 0:
            fatal("Can't build documentation")

        if op.exists('docs/html'):
            shutil.rmtree('docs/html')

        shutil.copytree('docs/build/html', 'docs/html')


class cyclean(Command):

    """ Clean up Cython-generated files and build cache"""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        
        allmodules = set()
        for x in MODULES.values():
            allmodules.update(x)

        dirs = ['build']

        for x in dirs:
            try:
                shutil.rmtree(x)
            except OSError:
                pass

        fnames = [ op.join(SRC_PATH, x+'.dep') for x in allmodules ] + \
                 [ op.join(SRC_PATH, x+'.c') for x in allmodules ] + \
                 [ op.join(SRC_PATH, 'config.pxi'), PICKLE_FILE]

        for name in fnames:
            try:
                os.remove(name)
            except OSError:
                pass


class new_sdist(sdist):

    """ Version of sdist that doesn't cache the MANIFEST file """

    def run(self):
        if os.path.exists('MANIFEST'):
            os.remove('MANIFEST')

        ipath = reduce(op.join, ('docs', 'source', 'build.rst'))
        if op.exists(ipath):
            shutil.copyfile(ipath, 'INSTALL.txt')

        sdist.run(self)

# New commands for setup (e.g. "python setup.py test")
if os.name == 'nt':
    CMD_CLASS.update({'build': cybuild, 'test': test, 'sdist': new_sdist,
                       'clean': cyclean})
else:
    CMD_CLASS.update({'build': cybuild, 'test': test, 'sdist': new_sdist,
                      'doc': doc, 'clean': cyclean, })


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
  ext_modules = [],
  requires = ['numpy (>=%s)' % MIN_NUMPY],
  cmdclass = CMD_CLASS
)



