from collections.abc import Mapping
from typing import Any

decode: tuple[str, ...]
encode: tuple[str, ...]

class FilterRefBase(Mapping):
    filter_id: int | None
    filter_options: tuple[Any, ...]

    @property
    def _kwargs(self) -> dict[str, Any]: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __getitem__(self, item: str) -> Any: ...
