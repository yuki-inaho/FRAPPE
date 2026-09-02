"""Printing a table and writing a report, in one place rather than nine.

Every tool ended with the same two things: a fixed-width table on stdout and a
JSON file for whatever reads the numbers next. The formatting was reinvented each
time, so column widths and significant figures drifted between tools whose output
is meant to be compared.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..experiment import atomic_json_dump


class Table:
    """A fixed-width table whose columns know their own format.

    Columns are ``(heading, key, format)``; a row is any mapping. Missing keys
    print as ``-`` instead of raising, because a tool that cannot measure one
    column for one row should still be able to print the rest.
    """

    def __init__(self, columns: Sequence[tuple[str, str, str]], indent: int = 4) -> None:
        self.columns = list(columns)
        self.indent = indent

    def _width(self, heading: str, spec: str) -> int:
        digits = "".join(c for c in spec.split(".", maxsplit=1)[0] if c.isdigit())
        return max(len(heading), int(digits) if digits else len(heading))

    def header(self) -> str:
        cells = [f"{heading:>{self._width(heading, spec)}}"
                 for heading, _, spec in self.columns]
        return " " * self.indent + "  ".join(cells)

    def row(self, values: dict) -> str:
        cells = []
        for heading, key, spec in self.columns:
            width = self._width(heading, spec)
            value = values.get(key)
            cells.append(f"{'-':>{width}}" if value is None
                         else format(value, spec).rjust(width))
        return " " * self.indent + "  ".join(cells)

    def render(self, rows: Iterable[dict]) -> str:
        return "\n".join([self.header(), *(self.row(row) for row in rows)])


def write_report(payload: dict[str, Any], destination: str | Path | None) -> None:
    """Write a report atomically, or do nothing when no destination was asked for."""
    if destination is None:
        return
    path = Path(destination)
    atomic_json_dump(json.loads(json.dumps(payload, default=str)), path)
    print(f"wrote {path}")
