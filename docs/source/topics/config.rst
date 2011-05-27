Configuring h5py
================


IPython
-------

h5py ships with a custom ipython completer, which provides object introspection
and tab completion for h5py objects in an ipython session. For example, if a
file contains 3 groups, "foo", "bar", and "baz"::

   In [4]: f['b<TAB>
   bar   baz

   In [4]: f['f<TAB>
   # Completes to:
   In [4]: f['foo'

   In [4]: f['foo'].<TAB>
   f['bye'].attrs            f['bye'].items            f['bye'].ref
   f['bye'].copy             f['bye'].iteritems        f['bye'].require_dataset
   f['bye'].create_dataset   f['bye'].iterkeys         f['bye'].require_group
   f['bye'].create_group     f['bye'].itervalues       f['bye'].values
   f['bye'].file             f['bye'].keys             f['bye'].visit
   f['bye'].get              f['bye'].name             f['bye'].visititems
   f['bye'].id               f['bye'].parent

The easiest way to enable the custom completer is to do the following in an
IPython session::

   In  [1]: import h5py

   In [2]: h5py.enable_ipython_completer()

It is also possible to configure IPython to enable the completer every time you
start a new session. For >=ipython-0.11, "h5py.ipy_completer" just needs to be
added to the list of extensions in
:file:`~/.config/ipython/ipython_config.py`. Here is the simplest possible
config file, in its entirety::

   c = get_config()
   c.Global.extensions = ['h5py.ipy_completer']

For <ipython-0.11, the completer can be enabled by adding the following lines
to the :func:`main` in :file:`.ipython/ipy_user_conf.py`::

   def main():
       ip.ex('from h5py import ipy_completer')
       ip.ex('ipy_completer.load_ipython_extension()')
