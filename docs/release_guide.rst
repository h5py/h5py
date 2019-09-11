.. _release_guide:

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


**Step 1 (repeat until successful)**

.. code-block:: sh

    $ rever check

Resolve any issues that may have come up, and keep running ``rever check``
until it passes. After it is successful, simply pass the version number
you want to release (e.g. ``X.Y.Z``) into the rever command.

**Step 2**

.. code-block:: sh

    $ rever X.Y.Z

You probably want to make sure (with ``git tag``) that the new version
number is available. If any release activities fail while running this
command, you may safely re-run this command. You can also safely undo
previously run activities. Please see the rever docs for more details.
