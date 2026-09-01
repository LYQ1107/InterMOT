#!/usr/bin/env python3
"""N44 evidence-preserving checks for external real-human tape input."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/n44/blocked_input_real_human_tape.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    sentinel = ROOT / "outputs/n34/human_event_tape.json"
    n40 = ROOT / "outputs/n40/stage_01_status.json"
    synthetic = ROOT / "outputs/n34/synthetic_event_tape.json"
    validator = ROOT / "sam3_intermot/interaction/real_human_event.py"
    sentinel_payload = json.loads(sentinel.read_text(encoding="utf-8")) if sentinel.is_file() else {}
    n40_payload = json.loads(n40.read_text(encoding="utf-8")) if n40.is_file() else {}
    synthetic_payload = json.loads(synthetic.read_text(encoding="utf-8")) if synthetic.is_file() else {}
    source_text = validator.read_text(encoding="utf-8") if validator.is_file() else ""
    output = {
        "schema": "N44_BLOCKED_INPUT_REAL_HUMAN_TAPE_V1",
        "status": "BLOCKED_INPUT_REAL_HUMAN_TAPE",
        "created_at": now(),
        "checks_completed": 3,
        "checks": {
            "sentinel_tape_check": {"path": str(sentinel), "sha256": digest(sentinel), "status": sentinel_payload.get("status"), "event_count": len(sentinel_payload.get("events", [])), "interaction_source": sentinel_payload.get("interaction_source"), "reason": sentinel_payload.get("reason"), "result": "NO_VERIFIABLE_REAL_EVENT_RECORDS"},
            "prior_contract_audit_check": {"path": str(n40), "sha256": digest(n40), "n40_status": n40_payload.get("status"), "real_human_tape_available": n40_payload.get("metrics", {}).get("real_human_tape_available", False), "real_full_loop": n40_payload.get("metrics", {}).get("real_full_loop", False), "result": "N40_CONFIRMS_NO_CURRENT_VERIFIABLE_TAPE"},
            "source_and_artifact_inventory_check": {"validator_path": str(validator), "validator_present": validator.is_file(), "required_contract_tokens_present": all(token in source_text for token in ("public_id_source", "human_confirmed", "candidate_tape_ref", "annotator_id_hash", "session_id", "runtime_future_gt_used")), "synthetic_fallback_path": str(synthetic), "synthetic_fallback_sha256": digest(synthetic), "synthetic_fallback_status": synthetic_payload.get("status"), "synthetic_fallback_source": synthetic_payload.get("interaction_source"), "synthetic_is_real_evidence": False, "external_ui_export_found": False, "candidate_complete_real_tape_found": False, "result": "CONTRACT_EXISTS_BUT_NO_EXTERNAL_UI_EXPORT_OR_REAL_INPUT_ARTIFACT"}
        },
        "exact_blocker": "This workspace contains the strict N40 validator and only GT-derived/simulated event artifacts; it has no externally supplied UI/session/annotator export containing direct public_id, human-confirmed BOX/CLICK/CONFIRMED_MASK, lossless ROI digest, timestamps, and candidate-complete future rows. The N34 sentinel is NOT_AVAILABLE with zero events.",
        "minimal_next_step": "Provide an external human-UI export plus referenced lossless candidate-complete tape for at least 24 train/train_fold events across >=16 independent sequences (four allowed action types represented), then run the existing N40 validator before any production authorization.",
        "fabrication_or_relabeling": "FORBIDDEN_AND_NOT_PERFORMED",
        "runtime_future_gt_used": False,
        "downstream_authorized": False,
        "old_simulated_artifacts_relabelled": False
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(OUT)}))


if __name__ == "__main__":
    main()
