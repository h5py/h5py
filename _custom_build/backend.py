# wrap setuptools.build_meta with an in-tree build backend
# This is the recommended way to implement dynamic build requirements that
# cannot be expressed via environment markers
# https://setuptools.pypa.io/en/latest/build_meta.html#dynamic-build-dependencies-and-other-build-meta-tweaks

import os
from setuptools import build_meta as _orig
from setuptools.build_meta import *


def get_requires_for_build_wheel(config_settings=None):
    # return is a list of packages needed to build h5py
    # (in addition to static list in pyproject.toml)
    # For mpi4py, we build against the oldest supported version;
    # h5py wheels should then work with newer versions of these.
    # Downstream packagers - e.g. Linux distros - can safely build with newer
    # versions.
    requires = _orig.get_requires_for_build_wheel(config_settings)

    # Set the environment variable H5PY_SETUP_REQUIRES=0 if we need to skip
    # setup_requires for any reason.
    if os.getenv('HDF5_MPI') == 'ON' and os.getenv('H5PY_SETUP_REQUIRES') != '0':
        requires.extend([
            "mpi4py ==3.1.2; python_version=='3.10.*'",
            "mpi4py ==3.1.4; python_version=='3.11.*'",
            "mpi4py ==3.1.6; python_version=='3.12.*'",
            "mpi4py ==4.0.1; python_version=='3.13.*'",
            "mpi4py ==4.1.0; python_version=='3.14.*'",
            # leave dependency unpinned for unstable Python versions
            "mpi4py",
        ])

    return requires
