..  -*- rst -*-

================
H5Py benchmarks
================

Benchmarking H5Py with `Airspeed Velocity`_.

.. _Airspeed Velocity: https://github.com/airspeed-velocity/asv.git

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

Some things to consider:

- The benchmark suite should be importable with any NumPy version.

- The benchmark parameters etc. should not depend on which NumPy version
  is installed.

- Try to keep the runtime of the benchmark reasonable.

- Prefer ASV's ``time_`` methods for benchmarking times rather than cooking up
  time measurements via ``time.clock``, even if it requires some juggling when
  writing the benchmark.

- Preparing arrays etc. should generally be put in the ``setup`` method rather
  than the ``time_`` methods, to avoid counting preparation time together with
  the time of the benchmarked operation.

- Be mindful that large arrays created with ``np.empty`` or ``np.zeros`` might
  not be allocated in physical memory until the memory is accessed. If this is
  desired behaviour, make sure to comment it in your setup function. If
  you are benchmarking an algorithm, it is unlikely that a user will be
  executing said algorithm on a newly created empty/zero array. One can force
  pagefaults to occur in the setup phase either by calling ``np.ones`` or
  ``arr.fill(value)`` after creating the array,
