#+
#
# This file is part of h5py, a low-level Python interface to the HDF5 library.
#
# Contributed by Darren Dale
#
# Copyright (C) 2009 Darren Dale
#
# http://h5py.alfven.org
# License: BSD  (See LICENSE.txt for full license)
#
#-

"""
h5py completer extension for ipython. This completer is automatically loaded
when h5py is imported within ipython. It will let you do things like::

  f=File('foo.h5')
  f['<tab>
  # or:
  f['ite<tab>

which will do tab completion based on the subgroups of `f`. Also::

  f['item1'].at<tab>

will perform tab completion for the attributes in the usual way. This should
also work::

  a = b = f['item1'].attrs.<tab>

as should::

  f['item1/item2/it<tab>

"""

import posixpath
import re

try:
    from IPython.utils import generics
    from IPython.core.error import TryNext
    from IPython.core.ipapi import get as ipget
except ImportError:
    # support <ipython-0.11
    from IPython import generics
    from IPython.ipapi import TryNext, get as ipget

import readline

from h5py.highlevel import AttributeManager, HLObject

re_attr_match = re.compile(r"(?:.*\=)?(.+\[.*\].*)\.(\w*)$")
re_item_match = re.compile(r"""(?:.*\=)?(.*)\[(?P<s>['|"])(?!.*(?P=s))(.*)$""")
re_object_match = re.compile(r"(?:.*\=)?(.+?)(?:\[)")


def h5py_item_completer(context, command):
    """Compute possible item matches for dict-like objects"""

    base, item = re_item_match.split(command)[1:4:2]

    # we don't want to call any functions, but I couldn't find a robust regex
    # that filtered them without unintended side effects. So keys containing
    # "(" will not complete.
    try:
        assert '(' not in base
    except AssertionError:
        raise ValueError()

    try:
        obj = eval(base, context.shell.user_ns)
    except:
        return []

    path, target = posixpath.split(item)
    if path:
        items = (posixpath.join(path, name) for name in obj[path].iterkeys())
    else:
        items = obj.iterkeys()
    items = list(items)

    readline.set_completer_delims(' \t\n`!@#$^&*()=+[{]}\\|;:\'",<>?')

    return [i for i in items if i[:len(item)] == item]


def h5py_attr_completer(context, command):
    """Compute possible attr matches for nested dict-like objects"""

    base, attr = re_attr_match.split(command)[1:3]
    base = base.strip()

    try:
        assert '(' not in base
    except AssertionError:
        raise ValueError()

    try:
        obj = eval(base, context.shell.user_ns)
    except:
        return []

    attrs = dir(obj)
    try:
        attrs = generics.complete_object(obj, attrs)
    except TryNext:
        pass

    omit__names = None
    try:
        # support >=ipython-0.12
        omit__names = ipget().Completer.omit__names
    except AttributeError:
        pass
    if omit__names is None:
        try:
            # support ipython-0.11
            omit__names = ipget().readline_omit__names
        except AttributeError:
            pass
    if omit__names is None:
        try:
            # support <ipython-0.11
            omit__names = ipget().options.readline_omit__names
        except AttributeError:
            omit__names = 0
    if omit__names == 1:
        attrs = [a for a in attrs if not a.startswith('__')]
    elif omit__names == 2:
        attrs = [a for a in attrs if not a.startswith('_')]

    readline.set_completer_delims(' =')

    return ["%s.%s" % (base, a) for a in attrs if a[:len(attr)] == attr]


def h5py_completer(self, event):
    base = re_object_match.split(event.line)[1]

    if not isinstance(self._ofind(base)['obj'], (AttributeManager, HLObject)):
        raise TryNext

    try:
        return h5py_attr_completer(self, event.line)
    except ValueError:
        pass

    try:
        return h5py_item_completer(self, event.line)
    except ValueError:
        pass

    return []


def load_ipython_extension(ip=None):
    if ip is None:
        ip = ipget()
    ip.set_hook('complete_command', h5py_completer, re_key=r"(?:.*\=)?(.+?)\[")
