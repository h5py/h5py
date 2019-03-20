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
    This provides a common set of functionality for interacting with IPython.
"""

from __future__ import absolute_import

__get_ipython = None

def get_ipython():
    global __get_ipython
    if __get_ipython is None:
        try:
            # >=ipython-1.0
            from IPython import get_ipython as get_ipy
        except ImportError:
            try:
                # support >=ipython-0.11, <ipython-1.0
                from IPython.core.ipapi import get as get_ipy
            except ImportError:
                # support <ipython-0.11
                from IPython.ipapi import get as get_ipy
        __get_ipython = get_ipy
    return __get_ipython()


def is_ipython_initialized():
    """ Determine whether or not an IPython interactive shell is initialized.
    """
    import sys
    ip_running = False
    if 'IPython' in sys.modules:
        try:
            from IPython.core.interactiveshell import InteractiveShell
            ip_running = InteractiveShell.initialized()
        except ImportError:
            # support <ipython-0.11
            from IPython import ipapi as _ipapi
            ip_running = _ipapi.get() is not None
        except Exception:
            pass

    return ip_running
