from distutils.core import setup
from distutils.extension import Extension
from distutils.cmd import Command
from Cython.Distutils import build_ext
import numpy
import os.path as op

VERSION = '1.4.0'

def localpath(*args):
    return op.abspath(reduce(op.join, (op.dirname(__file__),)+args))

modules = ['defs','_errors','_objects','_proxy', 'h5fd', 'h5z',
            'h5','h5i','h5r','utils',
            '_conv', 'h5t','h5s',
            'h5p',
            'h5d', 'h5a', 'h5f', 'h5g',
            'h5l', 'h5o']

EXTRA_SRC = {'h5z': [ localpath("lzf/lzf_filter.c"), 
                     localpath("lzf/lzf/lzf_c.c"),
                     localpath("lzf/lzf/lzf_d.c")]}

ext_modules = [ Extension('h5py.%s' %x,
                ['h5py/%s.pyx' % x] + EXTRA_SRC.get(x,[]),
                include_dirs = [localpath('lzf'), numpy.get_include()],
                libraries = ['hdf5'],
                define_macros = [('H5_USE_16_API',None)],
) for x in modules]

with open('h5py/config.pxi','w') as config_file:
    config_file.write('DEF H5PY_16API=1\n')
    config_file.write('DEF H5PY_18API=1\n')
    config_file.write('DEF H5PY_API=18\n')
    config_file.write('DEF H5PY_VERSION="%s"\n'%VERSION)

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

        suite = unittest.TestLoader().discover('.')
        unittest.TextTestRunner(verbosity=self.verbosity+1).run(suite)

setup(
    cmdclass = {'build_ext': build_ext, 'test': test},
    ext_modules = ext_modules,
    packages = ['h5py', 'h5py._hl', 'h5py._hl.tests', 'h5py.lowtest']
)

