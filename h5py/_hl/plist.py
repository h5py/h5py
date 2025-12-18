from typing import Literal


def set_file_locking(fapl, locking: Literal["true", "false", "best-effort"] | bool):
    if locking in ("false", False):
        fapl.set_file_locking(False, ignore_when_disabled=False)
    elif locking in ("true", True):
        fapl.set_file_locking(True, ignore_when_disabled=False)
    elif locking == "best-effort":
        fapl.set_file_locking(True, ignore_when_disabled=True)
    else:
        raise ValueError(f"Unsupported locking value: {locking}")
