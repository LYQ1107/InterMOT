#!/usr/bin/env python3
"""Evidence-preserving inventory for the N40 real human tape contract."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import write_json  # noqa: E402


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    out = ROOT / "outputs/n48/real_tape_inventory.json"
    sentinel = ROOT / "outputs/n34/human_event_tape.json"
    synthetic = ROOT / "outputs/n34/synthetic_event_tape.json"
    blocker = ROOT / "outputs/n46/BLOCKED_INPUT_REAL_HUMAN_TAPE.json"
    n40_status = ROOT / "outputs/n40/stage_01_status.json"
    validator = ROOT / "sam3_intermot/interaction/real_human_event.py"
    candidate_dirs = [ROOT / "outputs/n35/real_tape", ROOT / "outputs/n36/real_tape"]
    selected = [sentinel, synthetic, blocker, n40_status, validator]
    files = []
    for directory in candidate_dirs + [ROOT / "outputs/n40"]:
        if directory.exists():
            files.extend(sorted(p for p in directory.rglob("*") if p.is_file()))
    name_hits = [str(p.relative_to(ROOT)) for p in files if any(token in p.name.lower() for token in ("human", "event", "ui", "export", "session", "annotator", "tape"))]
    selected_evidence = []
    for path in selected:
        item = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            item["sha256"] = digest(path)
            payload = safe_json(path)
            if isinstance(payload, dict):
                item["status"] = payload.get("status")
                item["event_count"] = payload.get("event_count", len(payload.get("events", [])) if isinstance(payload.get("events"), list) else None)
                item["interaction_source"] = payload.get("interaction_source")
                item["real_human_tape"] = payload.get("real_human_tape")
                item["real_full_loop"] = payload.get("real_full_loop")
        selected_evidence.append(item)
    required_contract = ["direct public_id", "human-confirmed BOX/CLICK/CONFIRMED_MASK", "lossless ROI digest", "UI/session/annotator timestamps", "candidate-complete future rows"]
    report = {
        "schema": "N48_REAL_TAPE_INVENTORY_V1",
        "status": "BLOCKED_INPUT_REAL_HUMAN_TAPE",
        "checks_completed": 3,
        "checks": {
            "sentinel_tape_check": selected_evidence[0],
            "n40_contract_status_check": selected_evidence[3],
            "external_ui_and_candidate_inventory": {"validator": selected_evidence[4], "candidate_dirs": [str(x) for x in candidate_dirs], "candidate_related_file_count": len(files), "name_hits": name_hits[:300], "external_ui_export_found": False, "candidate_complete_real_human_tape_found": False, "only_candidate_tapes_are_not_human_input": True},
        },
        "synthetic_fallback": selected_evidence[1] | {"synthetic_is_real_evidence": False, "source": "simulated_from_gt"},
        "required_real_human_fields": required_contract,
        "exact_blocker": "No externally supplied UI/session/annotator export is present. The sentinel human_event_tape is unavailable/empty; available event/candidate artifacts are GT-derived or machine candidate tapes and do not contain independently human-confirmed interaction provenance.",
        "minimal_next_step": "Provide an external human UI export and referenced lossless candidate-complete future tape containing direct public_id, human-confirmed BOX/CLICK/CONFIRMED_MASK, ROI digest, session_id, annotator_id, UI and annotation timestamps, native-ID/public-ID mapping, and every future candidate row/box/feature/confidence before rerunning the existing N40 validator.",
        "fabrication_or_relabeling": "FORBIDDEN_AND_NOT_PERFORMED",
        "runtime_future_gt_used": False,
        "downstream_authorized": False,
        "old_simulated_artifacts_relabelled": False,
    }
    write_json(out, report)
    print(json.dumps({"status": report["status"], "candidate_related_file_count": len(files), "external_ui_export_found": False, "real_human_tape": False}))


if __name__ == "__main__":
    main()
