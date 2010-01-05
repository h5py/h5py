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

    Read INSTALL.txt for instructions.
"""

import os
import sys
import shutil
import commands
import os.path as op
import pickle

NAME = 'h5py'
VERSION = '1.3.0'
MIN_NUMPY = '1.0.3'
MIN_CYTHON = '0.12'
SRC_PATH = 'h5py'           # Name of directory with .pyx files

USE_DISTUTILS = False

# --- Helper functions --------------------------------------------------------

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

from contextlib import contextmanager
@contextmanager
def tempdir(*args):
    """ Create a temp dir and clean it up afterwards. """
    path = localpath(*args)
    try:
        shutil.rmtree(path)
    except Exception:
        pass
    os.mkdir(path)
    try:
        yield
    finally:
        try:
            shutil.rmtree(path)
        except Exception:
            pass

MODULES = ['h5', 'h5e', 'h5f', 'h5g', 'h5s', 'h5t', 'h5d', 'h5a', 'h5p', 'h5z',
                 'h5i', 'h5r', 'h5fd', 'utils', 'h5o', 'h5l', '_conv', '_proxy']

EXTRA_SRC = {'h5': [ localpath("lzf/lzf_filter.c"), 
                     localpath("lzf/lzf/lzf_c.c"),
                     localpath("lzf/lzf/lzf_d.c")]}

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

if USE_DISTUTILS:
    debug("Using distutils")
    from distutils.core import setup

from distutils.errors import DistutilsError
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
from distutils.command.clean import clean
from distutils.cmd import Command


# --- Compiler and library config ---------------------------------------------

class GlobalSettings(object):

    """
        Repository for all settings which are fixed when the script is run.
        This includes the following:

        * Any custom path to HDF5
        * Any custom API level
        * Compiler settings for extension modules
    """

    api_string = {'16': (1,6), '18': (1,8)}

    def get_environment_args(self):
        """ Look for options in environment vars """
        hdf5 = os.environ.get("HDF5_DIR", '')
        if hdf5 != '':
            debug("Found environ var HDF5_DIR=%s" % hdf5)
        else:
            hdf5 = None

        api = os.environ.get("HDF5_API", '')
        if api != '':
            debug("Found environ var HDF5_API=%s" % api)
            try:
                api = self.api_string[api]
            except KeyError:
                fatal("API level must be one of %s" % ", ".join(self.api_string))
        else:
            api = None

        return hdf5, api

    def get_commandline_args(self):
        """ Look for global options in the command line """
        hdf5 = api = None
        for arg in sys.argv[:]:
            if arg.find('--hdf5=') == 0:
                hdf5 = arg.split('=')[-1]
                if hdf5.lower() == 'default':
                    hdf5 = False    # This means "explicitly forget"
                sys.argv.remove(arg)
            if arg.find('--api=') == 0:
                api = arg.split('=')[-1]
                if api.lower() == 'default':
                    api = False
                else:
                    try:
                        api = self.api_string[api]
                    except KeyError:
                        fatal("API level must be 16 or 18")
                sys.argv.remove(arg)

        # We save command line args to a pickle file so that the user doesn't
        # have to keep specifying them for the different distutils commands.
        if (hdf5 or api) or (hdf5 is False or api is False):
            self.save_pickle_args(hdf5 if hdf5 is not False else None,
                                  api if api is not False else None)
        hdf5 = hdf5 if hdf5 else None
        api = api if api else None

        return hdf5, api
 
    def get_pickle_args(self):
        """ Look for options stored in the pickle file """
        import pickle
        hdf5 = api = None
        try:
            f = open(localpath('buildconf.pickle'),'r')
            hdf5, api = pickle.load(f)
            f.close()
        except Exception:
            pass
        return hdf5, api

    def save_pickle_args(self, hdf5, api):
        """ Save options to the pickle file """
        import pickle
        f = open(localpath('buildconf.pickle'),'w')
        pickle.dump((hdf5, api), f)
        f.close()
    
    def __init__(self):

        # --- Handle custom dirs and API levels for HDF5 ----------------------

        eargs = self.get_environment_args()
        cargs = self.get_commandline_args()
        pargs = self.get_pickle_args()

        # Commandline args have first priority, followed by pickle args and
        # finally by environment args
        hdf5 = cargs[0]
        if hdf5 is None: hdf5 = pargs[0]
        if hdf5 is None: hdf5 = eargs[0]

        api = cargs[1]
        if api is None: api = pargs[1]
        if api is None: api = eargs[1]

        if hdf5 is not None and not op.isdir(hdf5):
            fatal('Invalid HDF5 path "%s"' % hdf5)

        self.hdf5 = hdf5
        self.api = api

        # --- Extension settings ----------------------------------------------

        if sys.platform == 'win32':
            if hdf5 is None:
                warn("On Windows, HDF5 directory must be specified.")
                hdf5 = '.'
                
            self.libraries = ['hdf5dll18']
            self.include_dirs = [numpy.get_include(),
                                 op.join(hdf5, 'include'),
                                 localpath('lzf'),
                                 localpath('win_include')]
            self.library_dirs = [op.join(hdf5, 'dll')]
            self.runtime_dirs = []
            self.extra_compile_args = ['/DH5_USE_16_API', '/D_HDF5USEDLL_']

        else:
            self.libraries = ['hdf5']
            if hdf5 is None:
                self.include_dirs = [numpy.get_include()]
                self.library_dirs = []
                if sys.platform == 'darwin':
                    self.include_dirs += ['/opt/local/include']
                    self.library_dirs += ['/opt/local/lib']
            else:
                self.include_dirs = [numpy.get_include(), op.join(hdf5, 'include')]
                self.library_dirs = [op.join(hdf5, 'lib')]
            self.include_dirs += [localpath('lzf')]
            self.runtime_dirs = self.library_dirs
            self.extra_compile_args = ['-DH5_USE_16_API', '-Wno-unused', '-Wno-uninitialized']
    
    def check_hdf5(self):
        
        if hasattr(self, '_vers_cache'):
            return self._vers_cache

        from distutils import ccompiler
        from distutils.core import CompileError, LinkError
        import subprocess
        
        cc = ccompiler.new_compiler()
        cc.libraries = self.libraries
        cc.include_dirs = self.include_dirs
        cc.library_dirs = self.library_dirs
        cc.runtime_library_dirs = self.runtime_dirs

        with tempdir('detect'):

            f = open(localpath('detect', 'h5vers.c'),'w')
            f.write(
r"""\
#include <stdio.h>
#include "hdf5.h"

int main(){
    unsigned int main, minor, release;
    H5get_libversion(&main, &minor, &release);
    fprintf(stdout, "%d.%d.%d\n", main, minor, release);
    return 0;
}
""")
            f.close()
            try:
                objs = cc.compile([localpath('detect','h5vers.c')], extra_preargs=self.extra_compile_args)
            except CompileError:
                fatal("Can't find your installation of HDF5.  Use the --hdf5 option to manually specify the path.")
            try:
                cc.link_executable(objs, localpath('detect','h5vers.exe'))
            except LinkError:
                fatal("Can't link against HDF5.")
            if sys.platform == 'win32':
                shutil.copy(os.path.join(self.hdf5, 'dll', 'hdf5dll18.dll'), localpath('detect', 'hdf5dll18.dll'))
                shutil.copy(os.path.join(self.hdf5, 'dll', 'zlib1.dll'), localpath('detect', 'zlib1.dll'))
            result = subprocess.Popen(localpath('detect', 'h5vers.exe'),
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            so, se = result.communicate()
            if result.returncode:
                fatal("Error running HDF5 version detection script:\n%s\n%s" % (so,se))
            vmaj, vmin, vrel = (int(v) for v in so.split('.'))
            self._vers_cache = (vmaj, vmin, vrel)
            return (vmaj, vmin, vrel)

    def print_config(self):
        """ Print a summary of the configuration to stdout """

        vers = self.check_hdf5()

        print "\nSummary of the h5py configuration"
        print   "---------------------------------"
        print       "Installed HDF5 version:      %d.%d.%d" % vers
        print       "Path to HDF5 installation:   %s" % \
               (self.hdf5 if self.hdf5 is not None else "default")
        if self.api is None:
            print   "API compatibility level:     default (%d.%d)" % vers[0:2]
        else:
            print   "API compatibility level:     %d.%d (manually set)" % self.api


    def create_extension(self, name, extra_src=None):
        """ Create a distutils Extension object for the given module.  A list
            of C source files to be included in the compilation can also be
            provided.
        """
        if extra_src is None:
            extra_src = []
        sources = [op.join(SRC_PATH, name+'.c')]+[x for x in extra_src]
        return Extension(NAME+'.'+name,
                            sources, 
                            include_dirs = self.include_dirs, 
                            libraries = self.libraries,
                            library_dirs = self.library_dirs,
                            runtime_library_dirs = self.runtime_dirs,
                            extra_compile_args = self.extra_compile_args)

SETTINGS = GlobalSettings()
EXTENSIONS = [SETTINGS.create_extension(x, EXTRA_SRC.get(x, None)) for x in MODULES]


# --- Custom extensions for distutils -----------------------------------------

class configure(Command):

    description = "Display and check ssettings for h5py build"

    user_options = [('hdf5=', '5', 'Path to HDF5'),
                    ('api=', 'a', 'API version ("16" or "18")')]

    boolean_options = ['show']

    def initialize_options(self):
        self.hdf5 = None
        self.api = None
        self.show = False

    def finalize_options(self):
        pass

    def run(self):

        SETTINGS.print_config()

class cython(Command):

    """ Cython pre-builder """

    description = "Rebuild Cython-generated C files"

    user_options = [('api16', '6', 'Only build version 1.6'),
                    ('api18', '8', 'Only build version 1.8'),
                    ('force', 'f', 'Bypass timestamp checking'),
                    ('clean', 'c', 'Clean up Cython files'),
                    ('profile', 'p', 'Enable Cython profiling')]

    boolean_options = ['force', 'clean', 'profile']

    def initialize_options(self):
        self.api16 = None
        self.api18 = None
        self.force = False
        self.clean = False
        self.profile = False

    def finalize_options(self):
        if not (self.api16 or self.api18):
            self.api16 = self.api18 = True

    def checkdir(self, path):
        if not op.isdir(path):
            os.mkdir(path)

    def run(self):
        
        if self.clean:
            for path in [localpath(x) for x in ('api16','api18')]:
                try:
                    shutil.rmtree(path)
                except Exception:
                    debug("Failed to remove file %s" % path)
                else:
                    debug("Cleaned up %s" % path)
            return

        try:
            from Cython.Compiler.Main import Version, compile, compile_multiple, CompilationOptions
            if not version_check(Version.version, MIN_CYTHON):
                fatal("Old Cython %s version detected; at least %s required" % (Version.version, MIN_CYTHON))
        except ImportError:
            fatal("Cython (http://cython.org) is required to rebuild h5py")

        print "Rebuilding Cython files (this may take a few minutes)..."

        def cythonize(api):

            outpath = localpath('api%d' % api)
            self.checkdir(outpath)

            pxi_str = \
"""# This file is automatically generated.  Do not edit.

DEF H5PY_VERSION = "%(VERSION)s"

DEF H5PY_API = %(API_MAX)d     # Highest API level (i.e. 18 or 16)
DEF H5PY_16API = %(API_16)d    # 1.6.X API available (always true, for now)
DEF H5PY_18API = %(API_18)d    # 1.8.X API available
"""
            pxi_str %= {"VERSION": VERSION, "API_MAX": api,
                        "API_16": True, "API_18": api == 18}

            f = open(op.join(outpath, 'config.pxi'),'w')
            f.write(pxi_str)
            f.close()

            debug("  Cython: %s" % Version.version)
            debug("  API level: %d" % api)

            for module in MODULES:

                pyx_path = localpath(SRC_PATH, module+'.pyx')
                c_path = localpath(outpath, module+'.c')

                if self.force or \
                not op.exists(c_path) or \
                os.stat(pyx_path).st_mtime > os.stat(c_path).st_mtime:

                    debug("Cythoning %s" % pyx_path)
                    result = compile(pyx_path, verbose=False,
                                     compiler_directives = {'profile': self.profile},
                                     include_path=[outpath], output_file=c_path)
                    if result.num_errors != 0:
                        fatal("Cython error; aborting.")

        # end "def cythonize(...)"

        if self.api16:
            cythonize(16)
        if self.api18:
            cythonize(18)

class hbuild_ext(build_ext):
        
    def run(self):

        # First check if we can find HDF5
        vers =  SETTINGS.check_hdf5()

        # Used as a part of the path to the correct Cython build
        if SETTINGS.api is not None:
            api = 10*SETTINGS.api[0] + SETTINGS.api[1]
        else:
            api = 10*vers[0] + vers[1]

        if SETTINGS.hdf5 is None:
            autostr = "(path not specified)"
        else:
            autostr = "(located at %s)" % SETTINGS.hdf5
        
        print "Building for HDF5 %s.%s %s" % (divmod(api,10) + (autostr,))

        def identical(src, dst):
            if not op.isfile(src) or not op.isfile(dst):
                return False
            ident = False
            src_f = open(src,'r')
            dst_f = open(dst,'r')
            if src_f.read() == dst_f.read():
                ident = True
            src_f.close()
            dst_f.close()
            return ident

        src_files = [localpath('api%d'%api, x+'.c') for x in MODULES]
        dst_files = [localpath(SRC_PATH, x+'.c') for x in MODULES]

        if not all(op.exists(x) for x in src_files):
            fatal("Cython rebuild required ('python setup.py cython')")
        
        for src, dst in zip(src_files, dst_files):

            if identical(src, dst):
                debug("Skipping %s == %s" % (src, dst))
            else:
                debug("Copying %s -> %s" % (src, dst))
                shutil.copy(src, dst)
                #self.force = True   # If any files are out of date, we need to
                                    # recompile the whole thing for consistency

        build_ext.run(self)

        SETTINGS.print_config()

class cleaner(clean):

    def run(self):
        c_files = [localpath(SRC_PATH, x+'.c') for x in MODULES]
        so_files = [localpath(SRC_PATH, x+'.so') for x in MODULES]
        ext_files = [localpath('buildconf.pickle')]

        for path in c_files+so_files+ext_files:
            try:
                os.remove(path)
            except Exception:
                debug("Failed to clean up file %s" % path)
            else:
                debug("Cleaning up %s" % path)
        clean.run(self)

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


        buildobj = self.distribution.get_command_obj('build')
        buildobj.run()
        pth = op.abspath(buildobj.build_lib)

        if self.rebuild and op.exists('docs/build'):
            shutil.rmtree('docs/build')

        cmd = "export H5PY_PATH=%s; cd docs; make html" % pth

        retval = os.system(cmd)
        if retval != 0:
            fatal("Can't build documentation")

        if op.exists('docs/html'):
            shutil.rmtree('docs/html')

        shutil.copytree('docs/build/html', 'docs/html')

class nose_stub(Command):

    description = "UNSUPPORTED"

    user_options = []
    boolean_options = []

    def initialize_options(self):
        pass
    def finalize_options(self):
        pass

    def run(self):
        fatal("h5py is not compatible with nosetests command")

CMD_CLASS = {'cython': cython, 'build_ext': hbuild_ext, 'clean': cleaner,
             'configure': configure, 'doc': doc, 'nosetests': nose_stub}

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

Supports HDF5 versions 1.6.5 through 1.8.3.  On Windows, HDF5 is included in
the installer.
"""

if os.name == 'nt':
    package_data = {'h5py': ['*.pyx', '*.dll'],
                       'h5py.tests': ['data/*.hdf5', 'data/*.h5']}
else:
    package_data = {'h5py': ['*.pyx'],
                   'h5py.tests': ['data/*.hdf5', 'data/*.h5']}

setup(
  name = NAME,
  version = VERSION if sys.platform != 'win32' else VERSION.replace('-beta',''),
  description = short_desc,
  long_description = long_desc,
  classifiers = [x for x in cls_txt.split("\n") if x],
  author = 'Andrew Collette',
  author_email = '"h5py" at the domain "alfven.org"',
  maintainer = 'Andrew Collette',
  maintainer_email = '"h5py" at the domain "alfven.org"',
  url = 'http://h5py.alfven.org',
  download_url = 'http://code.google.com/p/h5py/downloads/list',
  packages = ['h5py','h5py.tests'],
  package_data = package_data,
  ext_modules = EXTENSIONS,
  requires = ['numpy (>=%s)' % MIN_NUMPY],
  cmdclass = CMD_CLASS,
  test_suite = 'h5py.tests'
)




