"""R2 output isolation, immutable evidence and bounded shared resources."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/N72R21R2"
R1 = ROOT.parent / "InterMOT_N72R21R1_worktree"
HISTORY = ROOT.parent / "InterMOT"
ASSETS = Path(os.environ.get("N72R21R2_ASSET_ROOT", str(ROOT.parent / "InterMOT_N72R21R2_assets"))).resolve()
TRAIN = ROOT.parent / "InterMOT_N72R16_assets/dataset/train"
OLD_CANDIDATES = ROOT.parent / "InterMOT_N72R20R2_assets"
PYTHON = HISTORY / ".venv/bin/python"
GOAL = "Event-Level Causal Identity Association: Safe One-Click Intervention with Fresh-Sequence Generalization for Online MOT"


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


def write_json(relative, value, *, mutable=False):
    path = OUT / relative
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError("R2 evidence cannot escape stage output root")
    for old in (HISTORY, R1):
        if path.resolve().is_relative_to(old.resolve()):
            raise ValueError("Historical worktrees are immutable")
    if path.exists() and not mutable:
        if read_json(path) == value:
            return path
        raise FileExistsError("Preserve sealed/partial evidence; version new attempts")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)
    return path


def update_status(**fields):
    value = read_json(OUT / "stage_status.json")
    assert value["goal"] == GOAL
    value.update(fields, last_progress_utc=utcnow())
    write_json("stage_status.json", value, mutable=True)


def append_log(action, **fields):
    with (OUT / "EXECUTION_LOG.jsonl").open("a") as handle:
        handle.write(json.dumps({"utc": utcnow(), "action": action, **fields}, sort_keys=True, allow_nan=False) + "\n")


def storage(reserve_bytes=0):
    free = shutil.disk_usage(ASSETS if ASSETS.exists() else ROOT).free
    floor = 60 * (1 << 30)
    if free - reserve_bytes < floor:
        raise RuntimeError("R2 would violate the60GiB actual filesystem reserve")
    return {"utc": utcnow(), "filesystem_free_bytes": free, "filesystem_free_GiB": free / (1 << 30), "floor_GiB": 60,
            "personal_limit_not_inferred_from_filesystem_free_space": True}


def preregistration():
    protocol = read_json(OUT / "PREREGISTRATION.json")
    assert protocol["goal"] == GOAL and protocol["frozen"]
    groups = protocol["split"]
    flat = [s for values in groups.values() for s in values]
    assert len(flat) == len(set(flat)) == 40
    assert {k: len(v) for k, v in groups.items()} == {"fit": 16, "inner": 8, "confirmation": 8, "historical_development": 8}
    return protocol


def development_sequence(sequence):
    protocol = preregistration()
    if sequence not in protocol["split"]["fit"] + protocol["split"]["inner"]:
        raise ValueError("Fresh development worker may not open confirmation/historical/VAL/TEST")
    return protocol
