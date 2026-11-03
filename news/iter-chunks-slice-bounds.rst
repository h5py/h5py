Bug fixes
---------

* ``Dataset.iter_chunks()`` no longer treats a slice that stops at 0 as one that
  runs to the end of the dataset: ``np.s_[0:0]`` and ``np.s_[5:0]`` now raise
  ``ValueError`` like other empty selections, instead of iterating over the whole
  dataset (or everything after index 5). Slices with a step other than 1, which
  were silently iterated as if the step were 1, now raise ``ValueError`` too.
