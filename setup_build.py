
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
from functools import reduce
import api_gen


def localpath(*args):
    return op.abspath(reduce(op.join, (op.dirname(__file__),)+args))


MODULES =  ['defs','_errors','_objects','_proxy', 'h5fd', 'h5z',
            'h5','h5i','h5r','utils',
            '_conv', 'h5t','h5s',
            'h5p',
            'h5d', 'h5a', 'h5f', 'h5g',
            'h5l', 'h5o',
            'h5ds', 'h5ac']


EXTRA_SRC = {'h5z': [ localpath("lzf/lzf_filter.c"),
              localpath("lzf/lzf/lzf_c.c"),
              localpath("lzf/lzf/lzf_d.c")]}

FALLBACK_PATHS = {
    'include_dirs': [],
    'library_dirs': []
}

COMPILER_SETTINGS = {
   'libraries'      : ['hdf5', 'hdf5_hl'],
   'include_dirs'   : [localpath('lzf')],
   'library_dirs'   : [],
   'define_macros'  : [('H5_USE_16_API', None)]
}

if sys.platform.startswith('win'):
    COMPILER_SETTINGS['include_dirs'].append(localpath('windows'))
    COMPILER_SETTINGS['define_macros'].extend([
        ('_HDF5USEDLL_', None),
        ('H5_BUILT_AS_DYNAMIC_LIB', None)
    ])
else:
    FALLBACK_PATHS['include_dirs'].extend(['/opt/local/include', '/usr/local/include'])
    FALLBACK_PATHS['library_dirs'].extend(['/opt/local/lib', '/usr/local/lib'])


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
        import pkgconfig

        settings = COMPILER_SETTINGS.copy()

        # Ensure that if a custom HDF5 location is specified, prevent
        # pkg-config and fallback locations from appearing in the settings
        if config.hdf5 is not None:
            settings['include_dirs'].insert(0, op.join(config.hdf5, 'include'))
            settings['library_dirs'].insert(0, op.join(config.hdf5, 'lib'))
        else:
            try:
                if pkgconfig.exists('hdf5'):
                    pkgcfg = pkgconfig.parse("hdf5")
                    settings['include_dirs'].extend(pkgcfg['include_dirs'])
                    settings['library_dirs'].extend(pkgcfg['library_dirs'])
                    settings['define_macros'].extend(pkgcfg['define_macros'])
            except EnvironmentError:
                pass
            settings['include_dirs'].extend(FALLBACK_PATHS['include_dirs'])
            settings['library_dirs'].extend(FALLBACK_PATHS['library_dirs'])

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
            sources = [localpath('h5py', module+'.pyx')] + EXTRA_SRC.get(module, [])
            return Extension('h5py.'+module, sources, **settings)

        return [make_extension(m) for m in MODULES]


    @staticmethod
    def run_system_cython(pyx_files):
        try:
            retcode = subprocess.call(['cython', '--fast-fail', '--verbose'] + pyx_files)
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
                missing_c_src_files.append( c_src_file )
        if missing_c_src_files:
            print("WARNING: cythonize() failed to create all .c files (setuptools too old?)")
            pyx_files = [os.path.splitext(fname)[0] + ".pyx" for fname in missing_c_src_files]
            print("         Executing system cython on pyx files: ", str(pyx_files))
            self.run_system_cython(pyx_files)


    def run(self):
        """ Distutils calls this method to run the command """

        from Cython.Build import cythonize
        import numpy

        # Provides all of our build options
        config = self.distribution.get_command_obj('configure')
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
                if config.mpi:
                    import mpi4py
                    from distutils.version import StrictVersion
                    v2 = StrictVersion(mpi4py.__version__) > StrictVersion("1.3.1")
                else:
                    v2 = False
                s = """\
# This file is automatically generated by the h5py setup script.  Don't modify.

DEF MPI = %(mpi)s
DEF MPI4PY_V2 = %(mpi4py_v2)s
DEF HDF5_VERSION = %(version)s
DEF SWMR_MIN_HDF5_VERSION = (1,9,178)
DEF VDS_MIN_HDF5_VERSION = (1,9,233)
DEF COMPLEX256_SUPPORT = %(complex256_support)s
"""
                s %= {
                    'mpi': bool(config.mpi),
                    'mpi4py_v2': bool(v2),
                    'version': tuple(int(x) for x in config.hdf5_version.split('.')),
                    'complex256_support': hasattr(numpy, 'complex256')
                }
                s = s.encode('utf-8')
                f.write(s)

        # Run Cython
        print("Executing cythonize()")
        self.extensions = cythonize(self._make_extensions(config),
                            force=config.rebuild_required or self.force)
        self.check_rerun_cythonize()

        # Perform the build
        build_ext.run(self)

        # Mark the configuration as built
        config.reset_rebuild()
