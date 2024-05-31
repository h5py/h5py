from types import FunctionType
from typing import Literal

from ._locks import FastRLock

# note: this annotation is invalidated if _objects.USE_LOCKING is set to False
phil: FastRLock

def with_phil(func) -> FunctionType: ...

class ObjectID:
    __weakref__: object
    id: int
    locked: Literal[0, 1]
    _hash: object
    _pyid: int
