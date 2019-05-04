#+
#
# This file is part of h5py, a low-level Python interface to the HDF5 library.
#
# Contributed by Anthony Wertz
#
# http://h5py.alfven.org
# License: BSD  (See LICENSE.txt for full license)
#
#-

# pylint: disable=eval-used,protected-access

"""
    This provides an interface for enabling IPython extensions.
"""

def enable_ipython_completer():
    """ Call this from an interactive IPython session to enable tab-completion
    of group and attribute names.
    """
    from .utils import is_ipython_initialized
    if is_ipython_initialized():
        from . import completer
        return completer.load_ipython_extension()

    raise RuntimeError('Completer must be enabled in active ipython session')


def enable_ipython_formatter():
    """ Call this from an interactive IPython session to enable prettier printing
    of h5py objects.
    """
    from .utils import is_ipython_initialized
    if is_ipython_initialized():
        from . import formatter
        return formatter.load_ipython_extension()

    raise RuntimeError('Formatter must be enabled in active ipython session')


def enable_ipython(completer=True, formatter=True):
    """ Call this from an interactive IPython session to enable multiple
    convenience extensions for IPython.
    """
    if completer: enable_ipython_completer()
    if formatter: enable_ipython_formatter()
