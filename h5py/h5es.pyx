include "config.pxi"
from ._objects import phil, with_phil

@with_phil
def create(self):
    return H5EScreate()
