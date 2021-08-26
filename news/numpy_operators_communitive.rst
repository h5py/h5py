New features
------------

* Numpy / h5py comparisons will be commutative in __eq__ and __neq__. This extends masking functionality, and enhances the compatibility of the two types.

Bug fixes
---------

* Numpy typing for __eq__ and __neq__ will now be used when h5py cannot duck-type datasets
