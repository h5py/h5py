#!/usr/bin/env python3
"""
    Implements a custom Distutils build_ext replacement, which handles the
    full extension module build process, from api_gen to C compilation and
    linking.
"""

try:
    from setuptools import Extension
except ImportError:
    from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import sys
import os
import os.path as op
import subprocess
import api_gen


def localpath(*args):
    return op.abspath(op.join(op.dirname(__file__), *args))


MODULES = ['defs', '_errors', '_objects', '_proxy', 'h5fd', 'h5z',
            'h5', 'h5i', 'h5r', 'utils', '_selector',
            '_conv', 'h5t', 'h5s',
            'h5p',
            'h5d', 'h5a', 'h5f', 'h5g',
            'h5l', 'h5o',
            'h5ds', 'h5ac',
            'h5pl']

EXTRA_SRC = {'h5z': [ localpath("lzf/lzf_filter.c"),
              localpath("lzf/lzf/lzf_c.c"),
              localpath("lzf/lzf/lzf_d.c")]}

COMPILER_SETTINGS = {
   'libraries'      : ['hdf5', 'hdf5_hl'],
   'include_dirs'   : [localpath('lzf')],
   'library_dirs'   : [],
   'define_macros'  : [('H5_USE_18_API', None),
                       ('NPY_NO_DEPRECATED_API', 0),
                      ]
}

CYTHON_SETTINGS = {
    'compiler_directives': {}
}

if sys.platform.startswith('win'):
    COMPILER_SETTINGS['include_dirs'].append(localpath('windows'))
    COMPILER_SETTINGS['define_macros'].extend([
        ('_HDF5USEDLL_', None),
        ('H5_BUILT_AS_DYNAMIC_LIB', None)
    ])

if os.environ.get('CYTHON_COVERAGE'):
    CYTHON_SETTINGS['compiler_directives'].update(**{'linetrace': True})
    COMPILER_SETTINGS['define_macros'].extend([
        ('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')
    ])

def cython_settings_to_args(cython_settings):
    """
    Converts cython compiler directives to cython cli arguments.

    NOTE: ONLY SUPPORTS COMPILER DIRECTIVES, SUPPORT FOR ADDITIONAL KEYWORDS
    DOES NOT EXIST AT THIS TIME
    """
    arguments = []
    for setting_name, setting in cython_settings.items():
        if setting_name != "compiler_directives":
            raise Exception("Unable to produce matching cli argument to {}".format(setting_name))
        for directive, dir_value in setting:
            arguments.extend(['-X', directive + '=' + str(dir_value)])
    return arguments


class h5py_build_ext(build_ext):

    """
        Custom distutils command which encapsulates api_gen pre-building,
        Cython building, and C compilation.

        Also handles making the Extension modules, since we can't rely on
        NumPy being present in the main body of the setup script.
    """

    @staticmethod
    def _make_extensions(config):
        """ Produce a list of Extension instances which can be passed to
        cythonize().

        This is the point at which custom directories, MPI options, etc.
        enter the build process.
        """
        import numpy

        settings = COMPILER_SETTINGS.copy()

        settings['include_dirs'][:0] = config.hdf5_includedirs
        settings['library_dirs'][:0] = config.hdf5_libdirs
        settings['define_macros'].extend(config.hdf5_define_macros)

        try:
            numpy_includes = numpy.get_include()
        except AttributeError:
            # if numpy is not installed get the headers from the .egg directory
            import numpy.core
            numpy_includes = os.path.join(os.path.dirname(numpy.core.__file__), 'include')

        settings['include_dirs'] += [numpy_includes]
        if config.mpi:
            import mpi4py
            settings['include_dirs'] += [mpi4py.get_include()]

        # TODO: should this only be done on UNIX?
        if os.name != 'nt':
            settings['runtime_library_dirs'] = settings['library_dirs']

        def make_extension(module):
            sources = [localpath('h5py', module + '.pyx')] + EXTRA_SRC.get(module, [])
            return Extension('h5py.' + module, sources, **settings)

        return [make_extension(m) for m in MODULES]

    @staticmethod
    def run_system_cython(pyx_files):
        try:
            retcode = subprocess.call(
                ['cython', '--fast-fail', '--verbose'] +
                cython_settings_to_args(CYTHON_SETTINGS) + pyx_files
            )
            if not retcode == 0:
                raise Exception('ERROR: Cython failed')
        except OSError as e:
            print("ERROR: cython exec failed. Is cython not in the path? ", str(e))
            raise
        except Exception as e:
            print("ERROR: cython exec failed", str(e))
            raise

    def check_rerun_cythonize(self):
        """ Check whether the cythonize() call produced the expected .c files.
        If the expected .c files are not found then cython from the system path will
        be executed in order to produce the missing files. """

        missing_c_src_files = []
        for c_src_file in [ext.sources[0] for ext in self.extensions]:
            if not op.isfile(c_src_file):
                missing_c_src_files.append(c_src_file)
        if missing_c_src_files:
            print("WARNING: cythonize() failed to create all .c files (setuptools too old?)")
            pyx_files = [os.path.splitext(fname)[0] + ".pyx" for fname in missing_c_src_files]
            print("         Executing system cython on pyx files: ", str(pyx_files))
            self.run_system_cython(pyx_files)

    def run(self):
        """ Distutils calls this method to run the command """

        from Cython import __version__ as cython_version
        from Cython.Build import cythonize
        import numpy

        # This allows ccache to recognise the files when pip builds in a temp
        # directory. It speeds up repeatedly running tests through tox with
        # ccache configured (CC="ccache gcc"). It should have no effect if
        # ccache is not in use.
        os.environ['CCACHE_BASEDIR'] = op.dirname(op.abspath(__file__))
        os.environ['CCACHE_NOHASHDIR'] = '1'

        # Provides all of our build options
        config = self.get_finalized_command('configure')
        config.run()

        defs_file = localpath('h5py', 'defs.pyx')
        func_file = localpath('h5py', 'api_functions.txt')
        config_file = localpath('h5py', 'config.pxi')

        # Rebuild low-level defs if missing or stale
        if not op.isfile(defs_file) or os.stat(func_file).st_mtime > os.stat(defs_file).st_mtime:
            print("Executing api_gen rebuild of defs")
            api_gen.run()

        # Rewrite config.pxi file if needed
        if not op.isfile(config_file) or config.rebuild_required:
            with open(config_file, 'wb') as f:
                s = """\
# This file is automatically generated by the h5py setup script.  Don't modify.

DEF MPI = %(mpi)s
DEF HDF5_VERSION = %(version)s
DEF SWMR_MIN_HDF5_VERSION = (1,9,178)
DEF VDS_MIN_HDF5_VERSION = (1,9,233)
DEF VOL_MIN_HDF5_VERSION = (1,11,5)
DEF COMPLEX256_SUPPORT = %(complex256_support)s
DEF NUMPY_BUILD_VERSION = '%(numpy_version)s'
DEF CYTHON_BUILD_VERSION = '%(cython_version)s'
"""
                s %= {
                    'mpi': bool(config.mpi),
                    'version': tuple(int(x) for x in config.hdf5_version.split('.')),
                    'complex256_support': hasattr(numpy, 'complex256'),
                    'numpy_version': numpy.__version__,
                    'cython_version': cython_version,
                }
                s = s.encode('utf-8')
                f.write(s)

        # Run Cython
        print("Executing cythonize()")
        self.extensions = cythonize(
            self._make_extensions(config),
            force = config.rebuild_required or self.force,
            language_level=3,
            **CYTHON_SETTINGS
        )
        self.check_rerun_cythonize()

        # Perform the build
        build_ext.run(self)

        # Mark the configuration as built
        config.reset_rebuild()
