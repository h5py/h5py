.. _release_guide:

..
    This is derived from the matplotlib release guide:
    https://matplotlib.org/devdocs/devel/release_guide.html#all-releases
    but there will be differences.

Release Guide
=============
h5py uses `rever <https://regro.github.io/rever-docs/>`_ for release management.
To install rever, use either pip or conda:

.. code-block:: sh

    # pip
    $ pip install re-ver

    # conda
    $ conda install -c conda-forge rever


Performing releases
-------------------
Once rever is installed, always run the ``check`` command to make sure
that everything you need to perform the release is correctly installed
and that you have the correct permissions. All rever commands should be
run in the root level of the repository.

..
    rever currently is set to upload to PyPI, which will likely cause issues as
    we need to upload the sdist and wheels at the same time

..
    We don't have any of the github stats set up, so we can ignore that I think
    For docs, running tox should be sufficient.


Step 1: Pre-release checks (repeat until successful)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: sh

    $ rever check

Resolve any issues that may have come up, and keep running ``rever check``
until it passes. After it is successful, simply pass the version number
you want to release (e.g. ``X.Y.Z``) into the rever command.

..
    This doesn't run the tests, should we have it run tox?

Step 2: Create the release
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: sh

    $ rever X.Y.Z

You probably want to make sure (with ``git tag``) that the new version
number is available. If any release activities fail while running this
command, you may safely re-run this command. You can also safely undo
previously run activities. Please see the rever docs for more details.

..
    I'm guessing we want this to just push the tag to GitHub.
    Things that could differ from matplotlib in this step
    * Empty commit
    * Tag signing (rever appears not to have that) - probably not needed
    * The format for the github release (we may not care as long as it's
      consistent)

    We should also work out what need to happen for zenodo

    The documentation is automatically rebuilt, so we don't need to worry about
    that step

    We should explain how to do the upload of the wheels (are we using rever for
    this?)

Step 3: Contact downstream packagers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
    This is directly from matplotlib docs, we should probably explain how to
    contact/any information we need to send them?


If this is a final release the following downstream packagers should be
contacted:

* Debian
* Fedora
* Arch
* Gentoo
* Macports
* Homebrew
* Continuum
* Enthought

This can be done ahead of collecting all of the binaries and uploading to pypi.

Step 4: Announce the release
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For all releases, send a email with a shortened version of the changelog with
acknowledgements to:

    h5py@googlegroups.com

For the final release, also send an email to the numpy and scipy mailing lists:

    numpy-discussion@python.org
    scipy-user@python.org

..
    Is there anywhere else we should be announcing?
