"""Shared source-table export for figure scripts."""

from __future__ import annotations

import csv
from pathlib import Path


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a nonempty source table with stable column order."""
    if not rows:
        raise ValueError(f'Refusing to write an empty table: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
