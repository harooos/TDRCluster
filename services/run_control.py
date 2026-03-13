"""
Run cancellation primitives shared by UI and pipeline services.
"""
from __future__ import annotations

from typing import Callable, Optional


class RunCancelledError(RuntimeError):
    """Raised when a user requests the current run to stop."""


CancelCallback = Optional[Callable[[], bool]]


def check_cancelled(should_cancel: CancelCallback, message: str = "Run cancelled by user."):
    if should_cancel and should_cancel():
        raise RunCancelledError(message)
