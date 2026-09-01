"""JSON / CSV output helpers."""

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_json(path: Path | str, payload: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, p)


def write_csv(path: Path | str, rows: Iterable[dict]) -> None:
    rows = list(rows)
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
