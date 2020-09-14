..  -*- rst -*-
.. _benchmark:

================
H5Py benchmarks
================

Benchmarking H5Py with `Airspeed Velocity`_.

.. _Airspeed Velocity: https://github.com/airspeed-velocity/asv

Usage
-----

Airspeed Velocity(asv)is a tool for benchmarking Python packages
over their lifetime. The latest released version can be installed
from PyPI using::

    pip install asv

NOTE

If you cannot successfully install asv, you may need to install
some requirements first::

    pip install six
    pip install virtualenv

Run ASV commands(record results and generate HTML)::

    cd benchmarks
    asv run
    asv publish
    asv preview

More on how to use ``asv`` can be found in `ASV documentation`_
Command-line help is available as usual via ``asv --help`` and
``asv run --help``.

.. _ASV documentation: https://asv.readthedocs.io/


Writing benchmarks
------------------

See `ASV documentation`_ for basics on how to write benchmarks.
