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
import copy
import sys
import os
import os.path as op
from pathlib import Path

import api_gen
from setup_configure import BuildConfig


def localpath(*args):
    return op.abspath(op.join(op.dirname(__file__), *args))


MODULES_NUMPY2 = ['_npystrings']
MODULES = ['defs', '_errors', '_objects', '_proxy', 'h5fd', 'h5z',
            'h5', 'h5i', 'h5r', 'utils', '_selector',
            '_conv', 'h5t', 'h5s',
            'h5p',
            'h5d', 'h5a', 'h5f', 'h5g',
            'h5l', 'h5o',
            'h5ds', 'h5ac',
            'h5pl'] + MODULES_NUMPY2

COMPILER_SETTINGS = {
   'libraries'      : ['hdf5', 'hdf5_hl'],
   'include_dirs'   : [localpath('lzf')],
   'library_dirs'   : [],
   'define_macros'  : [('H5_USE_110_API', None),
                       # The definition should imply the one below, but CI on
                       # Ubuntu 20.04 still gets H5Rdereference1 for some reason
                       ('H5Rdereference_vers', 2),
                       ('NPY_NO_DEPRECATED_API', 0),
                      ]
}

EXTRA_SRC = {'h5z': [ localpath("lzf/lzf_filter.c") ]}

# Set the environment variable H5PY_SYSTEM_LZF=1 if we want to
# use the system lzf library
if os.environ.get('H5PY_SYSTEM_LZF', '0') == '1':
    EXTRA_LIBRARIES = {
       'h5z': [ 'lzf' ]
    }
else:
    COMPILER_SETTINGS['include_dirs'] += [localpath('lzf/lzf')]

    EXTRA_SRC['h5z'] += [localpath("lzf/lzf/lzf_c.c"),
                  localpath("lzf/lzf/lzf_d.c")]

    EXTRA_LIBRARIES = {}

if sys.platform.startswith('win'):
    COMPILER_SETTINGS['include_dirs'].append(localpath('windows'))
    COMPILER_SETTINGS['define_macros'].extend([
        ('_HDF5USEDLL_', None),
        ('H5_BUILT_AS_DYNAMIC_LIB', None)
    ])


class h5py_build_ext(build_ext):

    """
        Custom distutils command which encapsulates api_gen pre-building,
        Cython building, and C compilation.

        Also handles making the Extension modules, since we can't rely on
        NumPy being present in the main body of the setup script.
    """

    @classmethod
    def _make_extensions(cls, config):
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

        if config.msmpi:
            settings['include_dirs'].extend(config.msmpi_inc_dirs)
            settings['library_dirs'].extend(config.msmpi_lib_dirs)
            settings['libraries'].append('msmpi')

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

        return [cls._make_extension(m, settings) for m in MODULES]

    @staticmethod
    def _make_extension(module, settings):
        import numpy

        sources = [localpath('h5py', module + '.pyx')] + EXTRA_SRC.get(module, [])
        settings = copy.deepcopy(settings)
        settings['libraries'] += EXTRA_LIBRARIES.get(module, [])

        assert int(numpy.__version__.split('.')[0]) >= 2  # See build dependencies in pyproject.toml
        if module in MODULES_NUMPY2:
            # Enable NumPy 2.0 C API for modules that require it.
            # These modules will not be importable when NumPy 1.x is installed.
            settings['define_macros'].append(('NPY_TARGET_VERSION', 0x00000012))

        return Extension('h5py.' + module, sources, **settings)

    def run(self):
        """ Distutils calls this method to run the command """

        from Cython import __version__ as cython_version
        from Cython.Build import cythonize
        import numpy

        complex256_support = hasattr(numpy, 'complex256')

        # This allows ccache to recognise the files when pip builds in a temp
        # directory. It speeds up repeatedly running tests through tox with
        # ccache configured (CC="ccache gcc"). It should have no effect if
        # ccache is not in use.
        os.environ['CCACHE_BASEDIR'] = op.dirname(op.abspath(__file__))
        os.environ['CCACHE_NOHASHDIR'] = '1'

        # Get configuration from environment variables
        config = BuildConfig.from_env()
        config.summarise()

        if config.hdf5_version < (1, 10, 7) or config.hdf5_version == (1, 12, 0):
            raise Exception(
                f"This version of h5py requires HDF5 >= 1.10.7 and != 1.12.0 (got version "
                f"{config.hdf5_version} from environment variable or library)"
            )

        config_file = localpath('h5py', 'config.pxi')

        # Refresh low-level defs if missing or stale
        print("Executing api_gen rebuild of defs")
        api_gen.run()

        # Rewrite config.pxi file if needed
        s = f"""\
# This file is automatically generated by the h5py setup script.  Don't modify.

DEF MPI = {bool(config.mpi)}
DEF ROS3 = {bool(config.ros3)}
DEF HDF5_VERSION = {config.hdf5_version}
DEF DIRECT_VFD = {bool(config.direct_vfd)}
DEF VOL_MIN_HDF5_VERSION = (1,11,5)
DEF COMPLEX256_SUPPORT = {complex256_support}
DEF NUMPY_BUILD_VERSION = '{numpy.__version__}'
DEF CYTHON_BUILD_VERSION = '{cython_version}'
"""
        write_if_changed(config_file, s)

        # Run Cython
        print("Executing cythonize()")
        self.extensions = cythonize(self._make_extensions(config),
                                    force=config.changed() or self.force,
                                    language_level=3)

        # Perform the build
        build_ext.run(self)

        # Record the configuration we built
        config.record_built()


def write_if_changed(target_path, s: str):
    """Overwrite target_path unless the contents already match s

    Avoids changing the mtime when we're just writing the same data.
    """
    p = Path(target_path)
    b = s.encode('utf-8')
    try:
        if p.read_bytes() == b:
            return
    except FileNotFoundError:
        pass

    p.write_bytes(b)
    print(f'Updated {p}')
