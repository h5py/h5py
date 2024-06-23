import os
from typing import Literal
from typing_extensions import TypeAlias

WINDOWS_ENCODING: Literal["utf-8", "mbcs"]

_PathT: TypeAlias = str | os.PathLike[str] | os.PathLike[bytes]

def filename_encode(filename: _PathT) -> bytes: ...
def filename_decode(filename: _PathT) -> str: ...
