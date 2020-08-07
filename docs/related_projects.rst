.. _related_projects:

Tools and Related Projects
==========================

There are a number of projects which build upon h5py, or who build upon HDF5,
which will likely be of interest to users of h5py. This page is non-exhaustive,
but if you think there should be a project added, feel free to create an issue
or  pull request at https://github.com/h5py/h5py/.

.. contents::
   :local:

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

Exploring and Visualising HDF5 files
------------------------------------

h5py does not contain a tool for exploring or visualising HDF5 files, but tools
built on h5py that can display the structure of h5py include:

 * `h5glance <https://github.com/European-XFEL/h5glance>`_


Additional Filters
------------------

Some projects providing additional HDF5 filter with integration into h5py
include:

 * `hdf5plugin <https://github.com/silx-kit/hdf5plugin>`_: this provides several
   plugins (currently blosc, bitshuffle, lz4, FCIDECOMP and ZFP), and newer
   plugins should look to supporting h5py via inclusion into hdf5plugin.


Field/Format Specific Libraries
-------------------------------

The are projects which build upon h5py to read and write to HDF5 files with a
specific schema, users of these formats may be interested in using these
higher-level wrappers of h5py:

 *


Other projects/tools
--------------------

The HDF5 lists a number of other tools on their
`website <https://portal.hdfgroup.org/display/support/Other+Tools>`_.
