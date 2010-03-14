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
    Setup script for h5py, rewritten for 1.3.

    Universal options (all commands):

    --hdf5=<path>   Path to your HDF5 installation (containing lib, include)
    --api=<16|18>   Which API level to use (HDF5 1.6 or HDF5 1.8)

    Custom commands:

    configure:  Compiles & links a test program to check HDF5.
"""

import sys, os
import os.path as op

VERSION = '1.3.0'

try:
    from setuptools import setup
    from setuptools.extension import Extension
except ImportError:
    from distutils.core import setup, Extension
    
from distutils.errors import DistutilsError
from distutils.command.build_ext import build_ext
from distutils.command.clean import clean
from distutils.cmd import Command
from distutils.command.build_ext import build_ext
import detect

try:
    import numpy
except ImportError:
    fatal("NumPy 1.0.3 or higher is required to use h5py (http://numpy.scipy.org).")

# --- Convenience functions ---------------------------------------------------

def debug(what):
    pass

def fatal(instring, code=1):
    print >> sys.stderr, "Fatal: "+instring
    exit(code)

def warn(instring):
    print >> sys.stderr, "Warning: "+instring

def localpath(*args):
    return op.abspath(reduce(op.join, (op.dirname(__file__),)+args))

def loadpickle(name):
    """ Load object from pickle file, or None if it can't be opened """
    import pickle
    try:
        f = open(name,'r')
    except IOError:
        return None
    try:
        return pickle.load(f)
    except Exception:
        return None
    finally:
        f.close()

def savepickle(name, data):
    """ Save to pickle file, exiting if it can't be written """
    import pickle
    try:
        f = open(name, 'w')
    except IOError:
        fatal("Can't open pickle file \"%s\" for writing" % name)
    try:
        pickle.dump(data, f)
    finally:
        f.close()

# --- Try to recover path, api options ---

def discover_settings():
    """ Discover custom settings for HDF5 path and API level """

    api_string = {'16': (1,6), '18': (1,8)}

    def get_eargs():
        """ Look for options in environment vars """

        settings = {}

        hdf5 = os.environ.get("HDF5_DIR", '')
        if hdf5 != '':
            debug("Found environ var HDF5_DIR=%s" % hdf5)
            settings['hdf5'] = hdf5

        api = os.environ.get("HDF5_API", '')
        if api != '':
            debug("Found environ var HDF5_API=%s" % api)
            try:
                api = api_string[api]
            except KeyError:
                fatal("API level must be one of %s" % ", ".join(api_string))
            settings['api'] = api

        return settings

    def get_cargs():
        """ Look for global options in the command line """
        settings = loadpickle('buildconf.pickle')
        if settings is None:  settings = {}
        for arg in sys.argv[:]:
            if arg.find('--hdf5=') == 0:
                hdf5 = arg.split('=')[-1]
                if hdf5.lower() == 'default':
                    settings.pop('hdf5', None)
                else:
                    settings['hdf5'] = hdf5
                sys.argv.remove(arg)
            if arg.find('--api=') == 0:
                api = arg.split('=')[-1]
                if api.lower() == 'default':
                    api = False
                else:
                    try:
                        settings['api'] = api_string[api]
                    except KeyError:
                        fatal("API level must be 16 or 18")
                sys.argv.remove(arg)
        savepickle('buildconf.pickle', settings)
        return settings

    settings = get_eargs()          # lowest priority
    settings.update(get_cargs())    # highest priority

    return settings.get('hdf5'), settings.get('api')

HDF5, API = discover_settings()

if HDF5 is not None and not op.exists(HDF5):
    warn("HDF5 directory \"%s\" does not appear to exist" % HDF5)

# --- Create extensions -------------------------------------------------------

if sys.platform.startswith('win'):
    COMPILER_SETTINGS = {
        'libraries'     : ['hdf5dll18'],
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
       'libraries'      : ['hdf5'],
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

MODULES = ['h5', 'h5e', 'h5f', 'h5g', 'h5s', 'h5t', 'h5d', 'h5a', 'h5p', 'h5z',
                 'h5i', 'h5r', 'h5fd', 'utils', 'h5o', 'h5l', '_conv', '_proxy']

EXTRA_SRC = {'h5': [ localpath("lzf/lzf_filter.c"), 
                     localpath("lzf/lzf/lzf_c.c"),
                     localpath("lzf/lzf/lzf_d.c")]}

def make_extension(module):
    sources = [op.join('h5py', module+'.c')] + EXTRA_SRC.get(module, [])
    return Extension('h5py.'+module, sources, **COMPILER_SETTINGS)

EXTENSIONS = [make_extension(m) for m in MODULES]


# --- Custom distutils commands -----------------------------------------------
    
class configure(Command):

    description = "Discover HDF5 version and features"

    # DON'T REMOVE: distutils demands these be here even if they do nothing.
    user_options = []
    boolean_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass

    tempdir = 'detect'

    def create_tempdir(self):
        import shutil
        self.erase_tempdir()
        os.mkdir(self.tempdir)
        if sys.platform.startswith('win'):
            shutil.copy(op.join(HDF5, 'dll', 'hdf5dll18.dll'), op.join(self.tempdir, 'hdf5dll18.dll'))
            shutil.copy(op.join(HDF5, 'dll', 'zlib1.dll'), op.join(self.tempdir, 'zlib1.dll'))
            shutil.copy(op.join(HDF5, 'dll', 'szlibdll.dll'), op.join(self.tempdir, 'szlibdll.dll'))

    def erase_tempdir(self):
        import shutil
        try:
            shutil.rmtree(self.tempdir)
        except Exception:
            pass

    def getcached(self):
        return loadpickle('configure.pickle')

    def run(self):
        self.create_tempdir()
        try:
            print "*"*42
            print "Configure: Autodetecting HDF5 settings..."
            print "    Custom HDF5 dir:       %s" % (HDF5,)
            print "    Custom API level:      %s" % (API,)
            config = detect.detect_hdf5(self.tempdir, **COMPILER_SETTINGS)
            savepickle('configure.pickle', config)
        except Exception:
            print """
    Failed to compile HDF5 test program.  Please check to make sure:

    * You have a C compiler installed
    * A development version of Python is installed (including header files)
    * A development version of HDF5 is installed (including header files)
    * If HDF5 is not in a default location, supply the argument --hdf5=<path>"""
            raise
        else:
            print "    HDF5 version detected: %s" % ".".join(str(x) for x in config['vers'])
        finally:
            print "*"*42
            self.erase_tempdir()
        self.config = config

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
        import shutil

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
            from Cython.Compiler.Main import compile, Version, compile_multiple, CompilationOptions
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

            self.debug_print("  Cython: %s" % Version.version)
            self.debug_print("  API level: %d" % api)

            for module in MODULES:

                pyx_path = localpath('h5py', module+'.pyx')
                c_path = localpath(outpath, module+'.c')

                if self.force or \
                not op.exists(c_path) or \
                os.stat(pyx_path).st_mtime > os.stat(c_path).st_mtime:

                    self.debug_print("Cythoning %s" % pyx_path)
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

        import shutil

        hdf5 = HDF5
        if hdf5 is not None and not op.isdir(hdf5):
            fatal("Custom HDF5 directory \"%s\" does not exist" % hdf5)

        configure = self.distribution.get_command_obj('configure')
        config = configure.getcached()
        if config is None:
            configure.run()
            config = configure.config

        api = API if API is not None else config['vers'][0:2]

        if api == (1,8) and config['vers'] in [(1,8,0), (1,8,1)]:
            warn('!'*42)
            warn('HDF5 1.8 features require HDF5 1.8.2 or later')
            warn('Forcing API to emulate HDF5 1.6')
            warn('!'*42)
            api = (1,6)

        if hdf5 is None:
            autostr = "(path not specified)"
        else:
            autostr = "(located at %s)" % hdf5
        
        if sys.platform.startswith('win'):
            if hdf5 is None:
                fatal("HDF5 directory must be specified on Windows")
            shutil.copy(op.join(hdf5, 'dll', 'hdf5dll18.dll'), localpath('h5py','hdf5dll18.dll'))
            shutil.copy(op.join(hdf5, 'dll', 'zlib1.dll'), localpath('h5py', 'zlib1.dll'))
            shutil.copy(op.join(hdf5, 'dll', 'szlibdll.dll'), localpath('h5py', 'szlibdll.dll'))

        print "*"*49
        print "Build: Building for HDF5 %s.%s %s" % (api[0], api[1], autostr)
        print "*"*49

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

        src_files = [localpath('api%d%d' % api, x+'.c') for x in MODULES]
        dst_files = [localpath('h5py', x+'.c') for x in MODULES]

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

class cleaner(clean):

    def run(self):
        c_files = [localpath('h5py', x+'.c') for x in MODULES]
        so_files = [localpath('h5py', x+'.so') for x in MODULES]
        ext_files = [localpath('buildconf.pickle'), localpath('configure.pickle')]

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
        import shutil
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


CMD_CLASS = {'cython': cython, 'build_ext': hbuild_ext,
             'configure': configure, 'clean': cleaner, 'doc': doc}

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

Supports HDF5 versions 1.6.5 through 1.8.4.  On Windows, HDF5 is included in
the installer.
"""

if os.name == 'nt':
    package_data = {'h5py': ['*.pyx', '*.dll'],
                       'h5py.tests': ['data/*.hdf5', 'data/*.h5']}
else:
    package_data = {'h5py': ['*.pyx'],
                   'h5py.tests': ['data/*.hdf5', 'data/*.h5']}

setup(
  name = 'h5py',
  version = VERSION if not sys.platform.startswith('win') else VERSION.partition('-')[0],
  description = short_desc,
  long_description = long_desc,
  classifiers = [x for x in cls_txt.split("\n") if x],
  author = 'Andrew Collette',
  author_email = '"h5py" at the domain "alfven.org"',
  maintainer = 'Andrew Collette',
  maintainer_email = '"h5py" at the domain "alfven.org"',
  url = 'http://h5py.alfven.org',
  download_url = 'http://code.google.com/p/h5py/downloads/list',
  packages = ['h5py','h5py.tests', 'h5py.tests.low', 'h5py.tests.high'],
  package_data = package_data,
  ext_modules = EXTENSIONS,
  requires = ['numpy (>=1.0.1)'],
  cmdclass = CMD_CLASS,
  test_suite = 'h5py.tests'
)








