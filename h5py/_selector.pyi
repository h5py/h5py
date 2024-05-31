class MultiBlockSlice:
    start: int
    stride: int
    count: int | None
    block: int
    def __init__(
        self,
        start: int = 0,
        stride: int = 1,
        count: int | None = None,
        block: int = 1,
    ) -> None: ...
    def indices(self, length: int) -> tuple[int, int, int, int]: ...
    def _repr(self, count: int | None = None) -> str: ...
    def __repr__(self) -> str: ...
