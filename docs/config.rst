Configuring h5py
================

Library configuration
---------------------

A few library options are available to change the behavior of the library.
You can get a reference to the global library configuration object via the
function :func:`h5py.get_config`.  This object supports the following
attributes:

    **complex_names**
        Set to a 2-tuple of strings (real, imag) to control how complex numbers
        are saved.  The default is ('r','i').

    **bool_names**
        Booleans are saved as HDF5 enums.  Set this to a 2-tuple of strings
        (false, true) to control the names used in the enum.  The default
        is ("FALSE", "TRUE").

    **track_order**
        Whether to track dataset/group/attribute creation order.  If
        container creation order is tracked, its links and attributes
        are iterated in ascending creation order (consistent with
        ``dict`` in Python 3.7+); otherwise in ascending alphanumeric
        order.  Global configuration value can be overridden for
        particular container by specifying ``track_order`` argument to
        :class:`h5py.File`, :meth:`h5py.Group.create_group`,
        :meth:`h5py.Group.create_dataset`.  The default is ``False``.


IPython
-------

H5py ships with a custom ipython completer, which provides object introspection
and tab completion for h5py objects in an ipython session. For example, if a
file contains 3 groups, "foo", "bar", and "baz"::

   In [4]: f['b<TAB>
   bar   baz

   In [4]: f['f<TAB>
   # Completes to:
   In [4]: f['foo'

   In [4]: f['foo'].<TAB>
   f['foo'].attrs            f['foo'].items            f['foo'].ref
   f['foo'].copy             f['foo'].iteritems        f['foo'].require_dataset
   f['foo'].create_dataset   f['foo'].iterkeys         f['foo'].require_group
   f['foo'].create_group     f['foo'].itervalues       f['foo'].values
   f['foo'].file             f['foo'].keys             f['foo'].visit
   f['foo'].get              f['foo'].name             f['foo'].visititems
   f['foo'].id               f['foo'].parent

The easiest way to enable the custom completer is to do the following in an
IPython session::

   In  [1]: import h5py

   In [2]: h5py.enable_ipython_completer()

It is also possible to configure IPython to enable the completer every time you
start a new session. For >=ipython-0.11, "h5py.ipy_completer" just needs to be
added to the list of extensions in your ipython config file, for example
:file:`~/.config/ipython/profile_default/ipython_config.py` (if this file does
not exist, you can create it by invoking `ipython profile create`)::

   c = get_config()
   c.InteractiveShellApp.extensions = ['h5py.ipy_completer']

For <ipython-0.11, the completer can be enabled by adding the following lines
to the :func:`main` in :file:`.ipython/ipy_user_conf.py`::

   def main():
       ip.ex('from h5py import ipy_completer')
       ip.ex('ipy_completer.load_ipython_extension()')
