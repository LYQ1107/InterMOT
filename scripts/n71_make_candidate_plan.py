"""Freeze a deterministic, outcome-independent N71 candidate smoke plan."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_tape_common import DATA_ROOT, image_files

EVENT_MANIFEST = ROOT / "outputs/n37/real_event_manifest.json"
OUT = ROOT / "outputs/N71/candidate_branch/window_plan.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def main() -> None:
    events = json.loads(EVENT_MANIFEST.read_text(encoding="utf-8"))["events"]
    selected = {}
    for item in events:
        ev = item["event"]
        selected.setdefault(str(ev["sequence"]), item)
    rows = []
    for sequence in sorted(selected)[:6]:
        item = selected[sequence]
        ev = item["event"]
        frame_count = int(item["sequence_frame_count"])
        # The window is selected from current/event metadata only.  It has a
        # fixed 20-frame prefix and 100-frame future core; no post-treatment
        # field or future label participates in selection.
        event_frame = int(ev["frame"])
        frame_start = max(0, event_frame - 20)
        frame_end = min(frame_count - 1, event_frame + 139)
        core_start = event_frame
        core_end = min(frame_count - 1, event_frame + 100)
        if not (frame_start <= core_start <= core_end <= frame_end):
            raise RuntimeError(f"invalid candidate window {sequence}")
        rows.append({
            "window_id": f"n71-{sequence}-{event_frame:04d}",
            "event_id": str(ev["event_id"]),
            "sequence": sequence,
            "frame_count_total": frame_count,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "core_start": core_start,
            "core_end": core_end,
            "prefix_overlap_frames": int(event_frame - frame_start),
            "future_core_frames": int(core_end - core_start + 1),
            "action_type": str(ev["action_type"]),
            "selection_basis": "first_six_unique_sequences_lexicographic; event metadata only; no replay/post-treatment fields",
            "interaction_source": "simulated_from_gt",
            "runtime_future_gt_used": False,
        })
    if len(rows) != 6 or len({r["sequence"] for r in rows}) != 6:
        raise RuntimeError("candidate plan did not produce six distinct sequences")
    payload = {
        "schema": "N71_CANDIDATE_WINDOW_PLAN_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_manifest": str(EVENT_MANIFEST),
        "event_manifest_sha256": sha(EVENT_MANIFEST),
        "data_root": str(DATA_ROOT),
        "branch": "D_NEW_SAM3_CANDIDATE_BRANCH",
        "settings": {"max_num_objects": 16, "multiplex_count": 16, "output_prob_thresh": 0.30, "chunk_frames": 160, "overlap_frames": 20, "feature_dim": 512, "feature_source": "FrozenMachineOSNet", "runtime_future_gt_used": False},
        "windows": rows,
        "status": "FROZEN_BEFORE_CANDIDATE_EXPORT",
    }
    output = Path(os.environ.get("N71_CANDIDATE_PLAN_OUTPUT", str(OUT))).resolve()
    atomic(output, payload)
    print(json.dumps({**payload, "output": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
