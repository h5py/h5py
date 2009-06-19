====================
Supplemental modules
====================

Filters module
--------------

This module has only two public attributes, and allows you to check which
filters are currently available for compression.  It is available at
``h5py.filters``.

.. attribute:: encode

    Tuple of filters currently available for encoding.  Possible element
    values are "gzip", "szip" and "lzf"

.. attribute:: decode

    Tuple of filters currently available for decoding.  Possible element
    values are "gzip", "szip" and "lzf".  Note that for some HDF5
    distributions, an SZIP decoder may be present, but not the encoder.

All other functions in this module are internal and subject to change without
warning.

Selections module
-----------------

This module implements the details of mapping Python-style selection arguments
to HDF5 hyperslab and point selections.  Selections are represented by
instances of ``h5py.selections.Selection`` subclasses.
