.. _release_guide:

Release Guide
=============

Steps to make an h5py release:

1. Have a look for any open issues or PRs which we might want to solve/merge
   before the release.
2. Prepare a new branch, e.g. ``git switch -c prepare-3.14``
3. Prepare the release notes. The goal is to describe changes visible to users
   & repackagers of h5py in a form that doesn't assume 'internal' development
   knowledge. If something breaks, or behaviour changes unexpectedly, the
   release notes should let someone make a good guess which change is involved.

   - Check for recent `merged PRs with no milestone <https://github.com/h5py/h5py/pulls?q=is%3Amerged+is%3Apr+no%3Amilestone>`_,
     and assign them to the current release milestone. We can ignore PRs which
     only touch CI with no consequences for users.
   - Go to the milestone page. If there are open issues/PRs there, decide whether
     to include or defer them.
   - Assemble the release notes in ``docs/whatsnew`` based on the list of merged
     PRs. Commit the changes.

4. Update the version number & commit the changes. The files that need changing
   are:

    - ``pyproject.toml``
    - ``h5py/version.py``
    - ``docs/conf.py``

5. Push the branch, open a PR, wait for the CI. Check the RTD build for the
   newly added release notes (formatting & cross-links). Optionally, wait for
   other maintainers to check it as well.
6. When everything looks good, merge the PR.
7. Checkout the master branch, pull, make and push the tag, which will cause
   CI to build sdist, wheels & make a GitHub release::

    git switch master
    git pull
    git tag 3.14.0  # <-- change this
    git push --tags

8. Download the artifacts: ``gh release download 3.14.0 -D dist/``
9. Upload to PyPI: ``twine upload dist/h5py-3.14.0*`` - this requires a token
    from PyPI, which must be `supplied to twine <https://twine.readthedocs.io/en/stable/#configuration>`_.
10. Close the GitHub milestone for this release and open one for the next
    version.
