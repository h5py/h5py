Bug Reports & Contributions
===========================

Contributions and bug reports are welcome from anyone!  Some of the best
features in h5py, including thread support, dimension scales, and the
scale-offset filter, came from user code contributions.

Since we use GitHub, the workflow will be familiar to many people.
If you have questions about the process or about the details of implementing
your feature, always feel free to ask on the Google Groups list, either
by emailing:

     h5py@googlegroups.com

or via the web interface at:

    https://groups.google.com/forum/#!forum/h5py

Anyone can post to this list. Your first message will be approved by a
moderator, so don't worry if there's a brief delay.

This guide is divided into three sections.  The first describes how to file
a bug report.

The second describes the mechanics of
how to submit a contribution to the h5py project; for example, how to
create a pull request, which branch to base your work on, etc.
We assume you're are familiar with Git, the version control system used by h5py.
If not, `here's a great place to start <http://git-scm.com/book>`_.

Finally, we describe the various subsystems inside h5py, and give
technical guidance as to how to implement your changes.


How to File a Bug Report
------------------------

Bug reports are always welcome!  The issue tracker is at:

    http://github.com/h5py/h5py/issues


If you're unsure whether you've found a bug
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Always feel free to ask on the mailing list (h5py at Google Groups).
Discussions there are seen by lots of people and are archived by Google.
Even if the issue you're having turns out not to be a bug in the end, other
people can benefit from a record of the conversation.

By the way, nobody will get mad if you file a bug and it turns out to be
something else.  That's just how software development goes.


What to include
~~~~~~~~~~~~~~~

When filing a bug, there are two things you should include.  The first is
the output of ``h5py.version.info``::

    >>> import h5py
    >>> print h5py.version.info

The second is a detailed explanation of what went wrong.  Unless the bug
is really trivial, **include code if you can**, either via GitHub's
inline markup::

    ```
        import h5py
        h5py.explode()    # Destroyed my computer!
    ```

or by uploading a code sample to `Github Gist <http://gist.github.com>`_.

How to Get Your Code into h5py
------------------------------

This section describes how to contribute changes to the h5py code base.
Before you start, be sure to read the h5py license and contributor
agreement in "license.txt".  You can find this in the source distribution,
or view it online at the main h5py repository at GitHub.

The basic workflow is to clone h5py with git, make your changes in a topic
branch, and then create a pull request at GitHub asking to merge the changes
into the main h5py project.

Here are some tips to getting your pull requests accepted:

1. Let people know you're working on something.  This could mean posting a
   comment in an open issue, or sending an email to the mailing list.  There's
   nothing wrong with just opening a pull request, but it might save you time
   if you ask for advice first.
2. Keep your changes focused.  If you're fixing multiple issues, file multiple
   pull requests.  Try to keep the amount of reformatting clutter small so
   the maintainers can easily see what you've changed in a diff.
3. Unit tests are mandatory for new features.  This doesn't mean hundreds
   (or even dozens) of tests!  Just enough to make sure the feature works as
   advertised.  The maintainers will let you know if more are needed.


.. _git_checkout:

Clone the h5py repository
~~~~~~~~~~~~~~~~~~~~~~~~~

The best way to do this is by signing in to GitHub and cloning the
h5py project directly.  You'll end up with a new repository under your
account; for example, if your username is ``yourname``, the repository
would be at http://github.com/yourname/h5py.

Then, clone your new copy of h5py to your local machine::

    $ git clone http://github.com/yourname/h5py


Create a topic branch for your feature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're fixing a bug, you'll want to check out a branch against the
appropriate stable branch.  For example, to fix a bug you found in version
2.1.3, you'll want to check out against branch "2.1"::

    $ git checkout -b bugfix 2.1

If you're contributing a new feature, it's appropriate to develop against the
"master" branch, so you would instead do::

    $ git checkout -b newfeature master

The exact name of the branch can be anything you want.  For bug fixes, one
approach is to put the issue number in the branch name.


Implement the feature!
~~~~~~~~~~~~~~~~~~~~~~

You can implement the feature as a number of small changes, or as one big
commit; there's no project policy.  Double-check to make sure you've
included all your files; run ``git status`` and check the output.


Push your changes back and open a pull request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Push your topic branch back up to your GitHub clone::

    $ git push origin newfeature

Then, `create a pull request <https://help.github.com/articles/creating-a-pull-request>`_ based on your topic branch.


Work with the maintainers
~~~~~~~~~~~~~~~~~~~~~~~~~

Your pull request might be accepted right away.  More commonly, the maintainers
will post comments asking you to fix minor things, like add a few tests, clean
up the style to be PEP-8 compliant, etc.

The pull request page also shows whether the project builds correctly,
using Travis CI. Check to see if the build succeeded (takes about 5 minutes),
and if not, try to modify your changes to make it work.

When making changes after creating your pull request, just add commits to
your topic branch and push them to your GitHub repository.  Don't try to
rebase or open a new pull request!  We don't mind having a few extra
commits in the history, and it's helpful to keep all the history together
in one place.


How to Modify h5py
------------------

This section is a little more involved, and provides tips on how to modify
h5py.  The h5py package is built in layers.  Starting from the bottom, they
are:

1. The HDF5 C API (provided by libhdf5)
2. Auto-generated Cython wrappers for the C API (``api_gen.py``)
3. Low-level interface, written in Cython, using the wrappers from (2)
4. High-level interface, written in Python, with things like ``h5py.File``.
5. Unit test code

Rather than talk about the layers in an abstract way, the parts below are
guides to adding specific functionality to various parts of h5py.
Most sections span at least two or three of these layers.


Adding a function from the HDF5 C API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is one of the most common contributed changes.  The example below shows
how one would add the function ``H5Dget_storage_size``,
which determines the space on disk used by an HDF5 dataset.  This function
is already partially wrapped in h5py, so you can see how it works.

It's recommended that
you follow along, if not by actually adding the feature then by at least
opening the various files as we work through the example.

First, get ahold of
the function signature; the easiest place for this is at the `online
HDF5 Reference Manual <http://www.hdfgroup.org/HDF5/doc/RM/RM_H5Front.html>`_.
Then, add the function's C signature to the file ``api_functions.txt``::

  hsize_t   H5Dget_storage_size(hid_t dset_id)

This particular signature uses types (``hsize_t``, ``hid_t``) which are already
defined elsewhere.  But if
the function you're adding needs a struct or enum definition, you can
add it using Cython code to the file ``api_types_hdf5.pxd``.

The next step is to add a Cython function or method which calls the function
you added.  The h5py modules follow the naming convention
of the C API; functions starting with ``H5D`` are wrapped in ``h5d.pyx``.

Opening ``h5d.pyx``, we notice that since this function takes a dataset
identifier as the first argument, it belongs as a method on the DatasetID
object.  We write a wrapper method::

    def get_storage_size(self):
        """ () => LONG storage_size

            Determine the amount of file space required for a dataset.  Note
            this only counts the space which has actually been allocated; it
            may even be zero.
        """
        return H5Dget_storage_size(self.id)

The first line of the docstring gives the method signature.
This is necessary because Cython will use a "generic" signature like
``method(*args, **kwds)`` when the file is compiled.  The h5py documentation
system will extract the first line and use it as the signature.

Next, we decide whether we want to add access to this function to the
high-level interface.  That means users of the top-level ``h5py.Dataset``
object will be able to see how much space on disk their files use.  The
high-level interface is implemented in the subpackage ``h5py._hl``, and
the Dataset object is in module ``dataset.py``.  Opening it up, we add
a property on the ``Dataset`` object::

    @property
    def storagesize(self):
        """ Size (in bytes) of this dataset on disk. """
        return self.id.get_storage_size()

You'll see that the low-level ``DatasetID`` object is available on the
high-level ``Dataset`` object as ``obj.id``.  This is true of all the
high-level objects, like ``File`` and ``Group`` as well.

Finally (and don't skip this step), we write **unit tests** for this feature.
Since the feature is ultimately exposed at the high-level interface, it's OK
to write tests for the ``Dataset.storagesize`` property only.  Unit tests for
the high-level interface are located in the "tests" subfolder, right near
``dataset.py``.

It looks like the right file is ``test_dataset.py``. Unit tests are
implemented as methods on custom ``unittest.UnitTest`` subclasses;
each new feature should be tested by its own new class.  In the
``test_dataset`` module, we see there's already a subclass called
``BaseDataset``, which implements some simple set-up and cleanup methods and
provides a ``h5py.File`` object as ``obj.f``.  We'll base our test class on
that::

    class TestStorageSize(BaseDataset):

        """
            Feature: Dataset.storagesize indicates how much space is used.
        """

        def test_empty(self):
            """ Empty datasets take no space on disk """
            dset = self.f.create_dataset("x", (100,100))
            self.assertEqual(dset.storagesize, 0)

        def test_data(self):
            """ Storage size is correct for non-empty datasets """
            dset = self.f.create_dataset("x", (100,), dtype='uint8')
            dset[...] = 42
            self.assertEqual(dset.storagesize, 100)

This set of tests would be adequate to get a pull request approved.  We don't
test every combination under the sun (different ranks, datasets with more
than 2**32 elements, datasets with the string "kumquat" in the name...), but
the basic, commonly encountered set of conditions.

To build and test our changes, we have to do a few things.  First of all,
run the file ``api_gen.py`` to re-generate the Cython wrappers from
``api_functions.txt``::

    $ python api_gen.py

Then build the project, which recompiles ``h5d.pyx``::

    $ python setup.py build

Finally, run the test suite, which includes the two methods we just wrote::

    $ python setup.py test

If the tests pass, the feature is ready for a pull request.


Adding a function only available in certain versions of HDF5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the moment, h5py must be backwards-compatible all the way back to
HDF5 1.8.4.  Starting with h5py 2.2.0, it's possible to conditionally
include functions which only appear in newer versions of HDF5.  It's also
possible to mark functions which require Parallel HDF5.  For example, the
function ``H5Fset_mpi_atomicity`` was introduced in HDF5 1.8.9 and requires
Parallel HDF5.  Specifiers before the signature in ``api_functions.txt``
communicate this::

  MPI 1.8.9 herr_t H5Fset_mpi_atomicity(hid_t file_id, hbool_t flag)

You can specify either, both or none of "MPI" or a version number in "X.Y.Z"
format.

In the Cython code, these show up as "preprocessor" defines ``MPI`` and
``HDF5_VERSION``.  So the low-level implementation (as a method on
``h5py.h5f.FileID``) looks like this::

    IF MPI and HDF5_VERSION >= (1, 8, 9):

        def set_mpi_atomicity(self, bint atomicity):
            """ (BOOL atomicity)

            For MPI-IO driver, set to atomic (True), which guarantees sequential
            I/O semantics, or non-atomic (False), which improves  performance.

            Default is False.

            Feature requires: 1.8.9 and Parallel HDF5
            """
            H5Fset_mpi_atomicity(self.id, <hbool_t>atomicity)

High-level code can check the version of the HDF5 library, or check to see if
the method is present on ``FileID`` objects.
