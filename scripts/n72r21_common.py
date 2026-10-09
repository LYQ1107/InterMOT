"""N72R21-only artifact writers; no historical stage writer is imported."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/N72R21"
ASSETS = Path(os.environ.get("N72R21_ASSET_ROOT", str(ROOT.parent / "InterMOT_N72R21_assets")))
DATA = Path(os.environ.get("N72R21_DATA_ROOT", str(ROOT.parent / "datasets/InterMOT_OneClickLT")))


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(relative, value):
    path = OUT / relative
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError("N72R21 output escape")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def storage(reserve_bytes=0):
    usage = shutil.disk_usage(ROOT)
    floor = 60 * (1 << 30)
    if usage.free - reserve_bytes < floor:
        raise RuntimeError("N72R21 personal-mount 60 GiB safety reserve would be violated")
    return {"utc": utcnow(), "free_bytes": usage.free, "free_gib": usage.free / (1 << 30),
            "safety_reserve_gib": 60, "requested_reserve_bytes": reserve_bytes}
