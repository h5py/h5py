********
Overview
********

.. note::

   This document assumes a basic familiarity with HDF5, including groups,
   datasets, and the file hierarchy.  For more information, the `user guide
   distributed by the HDF Group`__ is an excellent introduction.  It also
   assumes you have used `Numpy`_ before.

__ http://hdf.ncsa.uiuc.edu/HDF5/doc/UG/index.html
.. _Numpy: http://numpy.scipy.org

High-level classes
==================

While the ``h5py.h5*`` modules provide access to the guts of the HDF5 library,
they are not very convenient for everyday use.  For example, creating a new
dataset with chunking and compression requires creating datatype, dataspace and
property list objects, assigning the correct values to the property list in
the right order, and managing the group object to which the dataset will be
attached.  To make interacting with HDF5 less painful, a pure-Python
"high-level" interface is provided to encapsulate these common operations.

It consists of three classes:

* File: Represents an HDF5 file on disk
* Group: Represents an HDF5 group, containing datasets and other groups
* Dataset: An HDF5 dataset

All communication with HDF5 is done through Numpy arrays, dtype objects, and
slicing conventions.  No low-level HDF5 objects (datatype/dataspace objects,
etc.) are exposed, and no additional Python-side abstractions are introduced.
Objects typically have a small number of methods.

Paths in HDF5 files
-------------------

HDF5 files are organized like a filesystem.  Groups are analagous to
directories, while datasets are like the files stored in them.  Paths are
always specified UNIX-style, starting at ``/`` (the "root" group).  It's good
to limit yourself to ASCII, but you're welcome to use spaces, quotes,
punctuation and other symbol characters.


Group objects
-------------

These represent HDF5 *groups*, directory-like objects which contain *links* to
HDF5 objects like datasets, or other groups.  To a good approximation, you
can think of them as dictionaries which map a string name to an HDF5 object.
Like objects stored in Python dictionaries, the same HDF5 object can be
referred to by more than one group.  Groups can even contain themselves!

This dictionary metaphor is how h5py treats groups.  They support the following
Python behavior:

* Item access (``group["name"]``)

  Accessing a member by name returns the appropriate HDF5 object; usually a
  dataset or another group.  Assigning to a name stores the object in the file
  in the appropriate way (see the docstring for details).  Deleting an item
  "unlinks" it from the group.  Like Python objects in dictionaries, when zero
  groups refer to an object, it's permanently gone.

* Iteration (``for x in group...``, etc.)

  Iterating over a group yields the *names* of its members, like a Python dict.
  You can use the method ``iteritems()`` to get ``(name, obj)`` tuples.  The
  same restrictions as in Python apply for iteration; don't modify the group
  while iterating.

* Length (``len(group)``)

  This is just the number of entries in the group.

* Membership (``if "name" in group...``)

  Test if a name appears in the group.

They also support the following methods:

.. method:: create_group(name)

   Create a new, empty group attached to this one, called "name".

.. method:: create_dataset(name, *args, **kwds)

   Create a new dataset attached to this group.  The arguments are passed to
   the Dataset constructor below.


File objects
------------

These represent the HDF5 file itself.  Since every HDF5 file contains at least
the *root group* (called ``/``, as in a POSIX filesystem), it also provides
your entry point into the file.  File objects inherit from Group objects; all
the Group behavior and methods are available, and will operate on the root
(``/``) group.  For example, to access datasets::

    ds1 = file_obj["/ds1"]          (or file_obj["ds1"])
    ds2 = file_obj["/grp1/ds2"]     (or file_obj["grp1/ds2"])


Opening (or creating) a file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You use the File constructor directly to open or create an HDF5 file.  Like the
standard Python function ``open()``, the syntax is:

.. method:: File(name, mode='a')

   Allowed modes are:

   - r   Readonly, file must exist
   - r+  Read/write, file must exist
   - w   Create file, truncate if exists
   - w-  Create file, fail if exists
   - a   Read/write if exists, create otherwise (default)

.. method:: close()

    When you're done, as with Python files, it's important to close the file so
    that all the data gets written.


Python attributes
~~~~~~~~~~~~~~~~~

.. attribute:: name

   Name used to open the file

.. attribute:: mode

   Mode character used to open the file


Browsing a file
~~~~~~~~~~~~~~~

.. method:: browse()

    Specifying the full name of an HDF5 resource can be tedious and error-prone.
    Therefore, h5py includes a small command-line browser which can be used like
    a UNIX shell to explore an HDF5 file and import datasets and groups into an
    interactive session.  It includes things like ``ls`` and tab-completion. To
    open the browser, simply call browse().  Type ``help`` at the prompt for a
    list of commands.



    







