#!/usr/bin/env python

"""
    This is the main setup script for h5py (http://www.h5py.org).

    Most of the functionality is provided in two separate modules:
    setup_configure, which manages compile-time/Cython-time build options
    for h5py, and setup_build, which handles the actual compilation process.
"""

from setuptools import Extension, setup
import sys
import os

# Newer packaging standards may recommend removing the current dir from the
# path, add it back if needed.
if '' not in sys.path:
    sys.path.insert(0, '')

import setup_build, setup_configure



# these are required to use h5py
RUN_REQUIRES = [
    # We only really aim to support NumPy & Python combinations for which
    # there are wheels on PyPI (e.g. NumPy >=1.23.2 for Python 3.11).
    # But we don't want to duplicate the information in oldest-supported-numpy
    # here, and if you can build an older NumPy on a newer Python, h5py probably
    # works (assuming you build it from source too).
    # NumPy 1.21.2 is the first release with any wheels for Python 3.10, our minimum Python.
    "numpy >=1.21.2",
]

if setup_configure.mpi_enabled():
    # 3.1.2 is the first release with any wheels for Python 3.10
    RUN_REQUIRES.append('mpi4py >=3.1.2')

# --- Custom Distutils commands -----------------------------------------------

CMDCLASS = {'build_ext': setup_build.h5py_build_ext}


# --- Dynamic metadata for setuptools -----------------------------------------

package_data = {'h5py': [], "h5py.tests.data_files": ["*.h5"]}
if os.name == 'nt':
    package_data['h5py'].append('*.dll')

setup(
  package_data = package_data,
  ext_modules = [Extension('h5py.x',['x.c'])],  # To trick build into running build_ext
  install_requires = RUN_REQUIRES,
  cmdclass = CMDCLASS,
)

# see pyproject.toml for static metadata
