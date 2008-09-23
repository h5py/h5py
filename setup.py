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

    All commands take the usual distutils options, like --home, etc.  Cython is
    not required for installation, but will be invoked if the .c files are
    missing, one of the --pyrex options is used, or if a non-default API 
    version or debug level is requested.

    To build:
    python setup.py build

    To install:
    sudo python setup.py install

    To run the test suite locally (won't install anything):
    python setup.py test

    See INSTALL.txt or the h5py manual for additional build options.
"""

import os
import sys
import shutil
import os.path as op

from distutils.cmd import Command
from distutils.errors import DistutilsError, DistutilsExecError
from distutils.core import setup
from distutils.extension import Extension
from distutils.command.build import build 
from distutils.command.clean import clean
from distutils.command.sdist import sdist

# Basic package options
NAME = 'h5py'
VERSION = '0.4.0'
MIN_NUMPY = '1.0.3'
MIN_CYTHON = '0.9.8.1'
KNOWN_API = (16,18)    # Legal API levels (1.8.X or 1.6.X)
SRC_PATH = 'h5py'      # Name of directory with .pyx files
CMD_CLASS = {}         # Custom command classes for setup()
HDF5 = None            # Custom HDF5 directory.

# The list of modules depends on max API version
MODULES = {16:  ['h5', 'h5f', 'h5g', 'h5s', 'h5t', 'h5d', 'h5a', 'h5p', 'h5z',
                 'h5i', 'h5r', 'h5fd', 'utils'],
           18:  ['h5', 'h5f', 'h5g', 'h5s', 'h5t', 'h5d', 'h5a', 'h5p', 'h5z',
                 'h5i', 'h5r', 'h5fd', 'utils', 'h5o', 'h5l']}

# C source files required for all modules
EXTRA_SRC = ['utils_low.c']

def fatal(instring, code=1):
    print >> sys.stderr, "Fatal: "+instring
    exit(code)

def warn(instring):
    print >> sys.stderr, "Warning: "+instring

# === Required imports ========================================================

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

for arg in sys.argv[:]:
    if arg.find('--hdf5=') == 0:
        splitarg = arg.split('=',1)
        if len(splitarg) != 2:
            fatal("HDF5 directory not understood (wants --hdf5=/path/to/hdf5)")
        path = op.abspath(splitarg[1])
        if not op.exists(path):
            fatal("HDF5 path is illegal: %s" % path)
        HDF5 = path
        sys.argv.remove(arg)
        
# === Platform-dependent compiler config ======================================

if os.name == 'nt':
    if HDF5 is None:
        fatal("On Windows, HDF5 directory must be specified.")

    libraries = ['hdf5dll']
    include_dirs = [numpy.get_include(), op.join(HDF5, 'include')]
    library_dirs = [op.join(HDF5, 'dll2')]  # Must contain only "hdf5dll.dll.a"
    runtime_dirs = []
    extra_compile_args = ['-DH5_USE_16_API', '-D_HDF5USEDLL_', '-DH5_SIZEOF_SSIZE_T=4']
    extra_link_args = []
    package_data = {'h5py': ['*.pyx', '*.dll', 
                                    'Microsoft.VC90.CRT/*.manifest',
                                    'Microsoft.VC90.CRT/*.dll'],
                       'h5py.tests': ['data/*.hdf5']}

else:   # Assume Unix-like

    libraries = ['hdf5']
    if HDF5 is None:
        include_dirs = [numpy.get_include(), '/usr/include', '/usr/local/include']
        library_dirs = ['/usr/lib/', '/usr/local/lib']
    else:
        include_dirs = [numpy.get_include(), op.join(HDF5, 'include')]
        library_dirs = [op.join(HDF5, 'lib')]
    runtime_dirs = library_dirs
    extra_compile_args = ['-DH5_USE_16_API', '-Wno-unused', '-Wno-uninitialized']
    extra_link_args = []

    package_data = {'h5py': ['*.pyx'],
                   'h5py.tests': ['data/*.hdf5']}

# The actual extensions themselves are created at runtime, as the list of
# modules depends on command-line options.
def create_extension(name):
    """ Create a distutils Extension object for the given module.  Uses the
        globals in this file for things like the include directory.
    """
    sources = [op.join(SRC_PATH, name+'.c')]+[op.join(SRC_PATH,x) for x in EXTRA_SRC]
    ext = Extension(NAME+'.'+name,
                        sources, 
                        include_dirs = include_dirs, 
                        libraries = libraries,
                        library_dirs = library_dirs,
                        runtime_library_dirs = runtime_dirs,
                        extra_compile_args = extra_compile_args,
                        extra_link_args = extra_link_args)
    return ext


# === Custom extensions for distutils =========================================

class cybuild(build):

    """ Cython-aware subclass of the distutils build command

        It handles the h5py-centric configuration for the build processing,
        generating the compile-time definitions file "config.pxi" and
        running Cython before handing control over to the native distutils
        build.  The advantage is that Cython is run on all files before
        building begins, and is only run when the distribution is actually
        being built (and not, for example, when "sdist" is called for).
    """

    user_options = build.user_options + \
                    [('cython','y','Run Cython'),
                     ('cython-only','Y', 'Run Cython and stop'),
                     ('diag', 'd','Enable library logging'),
                     ('api=', 'a', 'Set API levels (--api=16,18)'),
                     ('threads', 't', 'Thread-aware')]
    boolean_options = build.boolean_options + ['cython', 'threads','diag']

    def initialize_options(self):
        self.cython = False
        self.cython_only = False
        self.threads = False
        self.api = (16,)
        self.diag = False
        self._explicit_only = False     # Hack for test subclass
        build.initialize_options(self)

    def finalize_options(self):

        if self.cython_only or  \
           self.api != (16,) or \
           self.threads or \
           self.diag:
            self.cython = True

        # Validate API levels
        if self.api != (16,):
            try:
                self.api = tuple(int(x) for x in self.api.split(',') if len(x) > 0)
                if len(self.api) == 0 or not all(x in KNOWN_API for x in self.api):
                    raise Exception
            except Exception:
                fatal('Illegal option %s to --api= (legal values are %s)' % (self.api, ','.join(str(x) for x in KNOWN_API)))

        build.finalize_options(self)

    def _get_pxi(self):
        """ Get the configuration .pxi for the current options. """

        pxi_str = \
"""# This file is automatically generated.  Do not edit.
# HDF5: %(HDF5)s

DEF H5PY_VERSION = "%(VERSION)s"

DEF H5PY_API = %(API_MAX)d     # Highest API level (i.e. 18 or 16)
DEF H5PY_16API = %(API_16)d    # 1.6.X API available
DEF H5PY_18API = %(API_18)d    # 1.8.X API available

DEF H5PY_DEBUG = %(DEBUG)d    # Logging-level number, or 0 to disable

DEF H5PY_THREADS = %(THREADS)d  # Enable thread-safety and non-blocking reads
"""
        pxi_str %= {"VERSION": VERSION, "API_MAX": max(self.api),
                    "API_16": 16 in self.api, "API_18": 18 in self.api,
                    "DEBUG": 10 if self.diag else 0, "THREADS": self.threads,
                    "HDF5": "Default" if HDF5 is None else HDF5}

        return pxi_str

    def run(self, *args, **kwds):

        modules = MODULES[max(self.api)]
        self.distribution.ext_modules = [create_extension(x) for x in modules]

        # Necessary because Cython doesn't detect changes to the .pxi
        recompile_all = False

        # Check if the config.pxi file needs to be updated for the given
        # command-line options.
        if self.cython or not self._explicit_only:
            pxi_path = op.join(SRC_PATH, 'config.pxi')
            pxi = self._get_pxi()
            if not op.exists(pxi_path):
                try:
                    f = open(pxi_path, 'w')
                    f.write(pxi)
                    f.close()
                except IOError:
                    fatal('Failed write to "%s"' % pxi_path)
                recompile_all = True
            else:
                try:
                    f = open(pxi_path, 'r+')
                except IOError:
                    fatal("Can't read file %s" % pxi_path)
                if f.read() != pxi:
                    f.close()
                    f = open(pxi_path, 'w')
                    f.write(pxi)
                    recompile_all = True
                f.close()

        if self.cython or recompile_all:
            print "Running Cython..."
            try:
                from Cython.Compiler.Main import Version, compile, compile_multiple, CompilationOptions
                from Cython.Distutils import build_ext
            except ImportError:
                fatal("Cython recompilation required, but Cython not installed.")

            if Version.version < MIN_CYTHON:
                fatal("Old Cython version detected; at least %s required" % MIN_CYTHON)

            print "  API levels: %s" % ','.join(str(x) for x in self.api)
            print "  Thread-aware: %s" % ('yes' if self.threads else 'no')
            print "  Diagnostic mode: %s" % ('yes' if self.diag else 'no')
            print "  HDF5: %s" % ('default' if HDF5 is None else HDF5)


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

        if self.cython_only:
            exit(0)

        build.run(self, *args, **kwds)

class test(cybuild):
    description = "Build and run unit tests"
    user_options = cybuild.user_options + \
                   [('sections=','s','Comma separated list of tests ("-" prefix to NOT run)')]

    def initialize_options(self):
        self.sections = None
        cybuild.initialize_options(self)

    def finalize_options(self):
        pass
        cybuild.finalize_options(self)

    def run(self):
        self._explicit_only = True  # Ignore config.pxi disagreement unless --cython
        cybuild.run(self)
        oldpath = sys.path
        try:
            sys.path = [op.abspath(self.build_lib)] + oldpath
            import h5py.tests
            if not h5py.tests.runtests(None if self.sections is None else tuple(self.sections.split(','))):
                raise DistutilsError("Unit tests failed.")
        finally:
            sys.path = oldpath

class doc(cybuild):

    """ Regenerate documentation.  Unix only, requires epydoc/sphinx. """

    description = "Rebuild documentation"

    def run(self):

        cybuild.run(self)

        for x in ('docs', 'docs/api-html'):
            if not op.exists(x):
                os.mkdir(x, 0755)

        retval = os.spawnlp(os.P_WAIT, 'epydoc', '-q', '--html',
                    '-o', 'docs/api-html', '--config', 'docs.cfg', 
                    os.path.join(self.build_lib, NAME) )
        if retval != 0:
            warn("Could not run epydoc to build documentation.")


        retval = os.system("cd docs; make html")
        if retval != 0:
            warn("Could not run Sphinx doc generator")
        else:
            if op.exists('docs/manual-html'):
                shutil.rmtree('docs/manual-html')
            shutil.copytree('docs/build/html', 'docs/manual-html')

class cyclean(clean):

    def run(self):
        
        allmodules = set()
        for x in MODULES.values():
            allmodules.update(x)

        for x in ('build','docs/api-html', 'docs/manual-html'):
            try:
                shutil.rmtree(x)
            except OSError:
                pass
        fnames = [ op.join(SRC_PATH, x+'.dep') for x in allmodules ] + \
                 [ op.join(SRC_PATH, x+'.c') for x in allmodules ] + \
                 [ 'MANIFEST']

        for name in fnames:
            try:
                os.remove(name)
            except OSError:
                pass

        clean.run(self)

class new_sdist(sdist):

    def run(self):
        if os.path.exists('MANIFEST'):
            os.remove('MANIFEST')
        shutil.copyfile(reduce(op.join, ('docs', 'source', 'build.rst')), 'INSTALL.txt')

        sdist.run(self)

# New commands for setup (e.g. "python setup.py test")
if os.name == 'nt':
    CMD_CLASS.update({'build': cybuild, 'test': test, 'sdist': new_sdist})
else:
    CMD_CLASS.update({'build': cybuild, 'test': test, 'sdist': new_sdist,
                      'doc': doc, 'clean': cyclean, })


cls_txt = \
"""
Development Status :: 4 - Beta
Intended Audience :: Developers
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: Unix
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
"""

short_desc = "General-purpose Python bindings for the HDF5 library"

long_desc = \
"""
The h5py package provides both a high- and low-level interface to the HDF5
library from Python. The low-level interface is intended to be a complete
wrapping of the HDF5 API, while the high-level component supports Python-style
object-oriented access to HDF5 files, datasets and groups.

A strong emphasis on automatic conversion between Python (Numpy) datatypes and
data structures and their HDF5 equivalents vastly simplifies the process of
reading and writing data from Python. 
"""

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
  url = 'h5py.alfven.org',
  packages = ['h5py','h5py.tests'],
  package_data = package_data,
  ext_modules = [],
  requires = ['numpy (>=%s)' % MIN_NUMPY],
  cmdclass = CMD_CLASS
)



