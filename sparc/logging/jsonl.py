"""Structured JSON-lines training log (ROADMAP Week 14 Log.out equivalent)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class JsonlLogger:
    """Append-only JSONL log; one JSON object per line."""

    path: Path | None = None
    stream: TextIO | None = None
    _file: TextIO = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("a", encoding="utf-8")
        else:
            self._file = self.stream or sys.stdout

    def log(self, event: str, **fields: Any) -> None:
        record = {"ts": _utc_now(), "event": event, **fields}
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self.path is not None and hasattr(self, "_file"):
            self._file.close()
