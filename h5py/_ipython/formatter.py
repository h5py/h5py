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
    This is the h5py formatter extension for ipython.  It is loaded by
    calling the function h5py.enable_ipython_formatter() from within an
    interactive IPython session.
"""

from __future__ import absolute_import

import six
import os
from .utils import get_ipython


def __pprint_KeysViewHDF5(obj, p, cycle):
    p.text('\n'.join(["<KeysViewHDF5>:"] + ["  + " + key for key in list(obj)]))


def __pprint_Group(obj, p, cycle):
    if not obj:
        r = u"<Closed HDF5 group>"
    else:
        namestr = (
            u'"%s"' % obj.name
        ) if obj.name is not None else u"(anonymous)"
        r = u'<HDF5 group %s (%d members)>\n' % (namestr, len(obj))

    if six.PY2:
        r = r.encode('utf8')

    p.text(r)
    __pprint_KeysViewHDF5(obj.keys(), p, cycle)


def __pprint_File(obj, p, cycle):
    if not obj.id:
        r = u'<Closed HDF5 file>'
    else:
        # Filename has to be forced to Unicode if it comes back bytes
        # Mode is always a "native" string
        filename = obj.filename
        if isinstance(filename, bytes):  # Can't decode fname
            filename = filename.decode('utf8', 'replace')
        r = u'<HDF5 file "%s" (mode %s)>\n' % (os.path.basename(filename),
                                                obj.mode)

    if six.PY2:
        r = r.encode('utf8')

    p.text(r)
    __pprint_Group(obj, p, cycle)


def load_ipython_extension(ip=None):
    """ Load formatter function into IPython """
    if ip is None:
        ip = get_ipython()

    from .._hl.base import KeysViewHDF5
    from .._hl.group import Group
    from .._hl.files import File

    text_formatter = ip.display_formatter.formatters['text/plain']
    text_formatter.for_type(KeysViewHDF5, __pprint_KeysViewHDF5)
    text_formatter.for_type(Group, __pprint_Group)
    text_formatter.for_type(File, __pprint_File)
