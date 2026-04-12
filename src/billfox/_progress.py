from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum


class Stage(Enum):
    """Pipeline processing stages."""

    LOADING = "LOADING"
    PREPROCESSING = "PREPROCESSING"
    EXTRACTING = "EXTRACTING"
    PARSING = "PARSING"
    STORING = "STORING"


class Status(Enum):
    """Status of a pipeline stage."""

    STARTED = "STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ProgressEvent:
    """A typed progress event emitted at pipeline stage boundaries."""

    stage: Stage
    status: Status
    message: str | None = None
    metadata: dict[str, object] | None = None


ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]
