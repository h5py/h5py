# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.
import sys
import shlex
from importlib.util import find_spec
from subprocess import call

def run_tests(args=''):
    if find_spec("pytest") is None:
        print("Tests require pytest, pytest not installed")
        return 1

    cli = [sys.executable, "-m", "pytest", "--pyargs", "h5py"]
    cli.extend(shlex.split(args))
    return call(cli)
