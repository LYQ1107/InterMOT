#!/usr/bin/env python3
"""Audit whether the frozen N72R10 event pool can support the planned corpus.

This is a read-only planning audit.  It does not duplicate events, extend the
future horizon, reuse the old static re-query stream, or generate synthetic
examples.  The older 40-event policy pool is inspected only to identify
records that lack the N72R9 same-run public-authority inputs required by the
true future-session executor.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
N72R9_PROTOCOL = ROOT / "outputs/N72R9/protocol.json"
OLD_POOL = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
CORPUS = ROOT / "outputs/N72R10/training/corpus_manifest.json"
TRAINING = ROOT / "outputs/N72R10/stage_06_training_status.json"
FUTURE_AUDIT = ROOT / "outputs/N72R10/stage_03_true_future_requery/batch_integrity_audit.json"
OUTPUT = ROOT / "outputs/N72R10/stage_10_training_distribution_audit.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    protocol = read_json(N72R9_PROTOCOL)
    old_pool = read_json(OLD_POOL)
    corpus = read_json(CORPUS)
    training = read_json(TRAINING)
    future_audit = read_json(FUTURE_AUDIT)
    frozen_events = [dict(item) for item in protocol.get("source_event_selection", {}).get("events", [])]
    frozen_by_id = {str(item["event_id"]): item for item in frozen_events}
    old_events = [dict(item) for item in old_pool.get("events", [])]
    old_by_id = {str(item["event_id"]): item for item in old_events}
    extra_ids = sorted(set(old_by_id) - set(frozen_by_id))
    extra_records = []
    for event_id in extra_ids:
        item = old_by_id[event_id]
        extra_records.append({
            "event_id": event_id,
            "sequence": item.get("sequence"),
            "action_type": item.get("action_type"),
            "event_frame": item.get("event_frame"),
            "old_target_public_id": item.get("target_public_id"),
            "old_public_id_authority": item.get("public_id_authority"),
            "has_old_candidate_tape": bool(item.get("candidate_tape_ref")),
            "has_n72r9_c0_source": False,
            "has_n72r9_c1_source": False,
            "has_n72r9_target_stream_source": False,
            "compatible_with_true_future_executor": False,
            "exclusion_reason": "NO_FROZEN_N72R9_SAME_RUN_PUBLIC_AUTHORITY_OR_CAUSAL_SOURCE_STREAM",
        })
    train = dict(corpus.get("splits", {}).get("train", {}))
    validation = dict(corpus.get("splits", {}).get("validation", {}))
    validation_eval = dict(training.get("validation_evaluation", {}))
    old_actions = Counter(str(item.get("action_type")) for item in old_events)
    frozen_actions = Counter(str(item.get("action_type")) for item in frozen_events)
    potential_causal_frames = 100 * len(frozen_events)
    payload = {
        "schema_version": "N72R10_TRAINING_DISTRIBUTION_AUDIT_V1",
        "status": "LIMITED_FROZEN_EVENT_POOL_REQUIRES_NEW_CAUSAL_INTERACTIONS",
        "created_at_utc": now_utc(),
        "protocol": str(N72R9_PROTOCOL),
        "protocol_sha256": sha256_file(N72R9_PROTOCOL),
        "old_event_pool": str(OLD_POOL),
        "old_event_pool_sha256": sha256_file(OLD_POOL),
        "corpus_manifest": str(CORPUS),
        "corpus_manifest_sha256": sha256_file(CORPUS),
        "training_status": str(TRAINING),
        "training_status_sha256": sha256_file(TRAINING),
        "future_audit": str(FUTURE_AUDIT),
        "future_audit_sha256": sha256_file(FUTURE_AUDIT),
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "runtime_future_gt_used": False,
        "frozen_n72r10_pool": {
            "event_count": len(frozen_events),
            "sequence_count": len({str(item.get("sequence")) for item in frozen_events}),
            "action_counts": dict(sorted(frozen_actions.items())),
            "fixed_future_horizon_frames_per_event": 100,
            "maximum_causal_future_frame_examples_without_new_events": potential_causal_frames,
            "train_event_count": int(train.get("event_count", 0)),
            "validation_event_count": int(validation.get("event_count", 0)),
            "train_sequence_count": int(train.get("sequence_count", 0)),
            "validation_sequence_count": int(validation.get("sequence_count", 0)),
        },
        "materialized_corpus": {
            "train_examples": int(train.get("example_count", 0)),
            "validation_examples": int(validation.get("example_count", 0)),
            "train_target_minimum": 30000,
            "validation_target_minimum": 5000,
            "train_shortfall": max(0, 30000 - int(train.get("example_count", 0))),
            "validation_shortfall": max(0, 5000 - int(validation.get("example_count", 0))),
            "train_source_counts": train.get("source_counts", {}),
            "validation_source_counts": validation.get("source_counts", {}),
            "train_future_rows_total": int(train.get("future_rows_total", 0)),
            "train_future_rows_selected_as_label": int(train.get("future_rows_selected_as_label", 0)),
            "validation_future_rows_total": int(validation.get("future_rows_total", 0)),
            "validation_future_rows_selected_as_label": int(validation.get("future_rows_selected_as_label", 0)),
            "validation_none_accuracy": validation_eval.get("none_accuracy"),
        },
        "old_pool_comparison": {
            "old_event_count": len(old_events),
            "old_sequence_count": len({str(item.get("sequence")) for item in old_events}),
            "old_action_counts": dict(sorted(old_actions.items())),
            "overlap_with_frozen_n72r10_count": len(set(old_by_id) & set(frozen_by_id)),
            "extra_old_event_count": len(extra_records),
            "extra_old_events_are_not_added": True,
            "extra_events": extra_records,
            "why_not_reused": "The extra historical policy records expose candidate_tape/candidate_pool evidence but no N72R9 same-run c0/c1/target stream with explicit public authority. Reusing them would mix the old static/event policy with N72R10 FUTURE_FRAME_REQUERY and would violate the causal/public-ID contract.",
        },
        "lawful_expansion": {
            "can_reach_target_by_duplication": False,
            "can_reach_target_by_horizon_extension": False,
            "can_reach_target_with_current_frozen_events_only": False,
            "additional_valid_events_available_locally_without_new_official_run": 0,
            "new_event_generation_required": True,
            "required_input": "Additional train/train_fold events with an explicit same-run public authority, complete c0/c1/target streams, causal current-frame trigger, fresh FUTURE_FRAME_REQUERY artifact, and a sequence-disjoint validation split containing positive FUTURE_FRAME_REQUERY labels.",
            "prohibited_shortcuts": [
                "duplicate the 32 frozen events",
                "reuse N72R7/N72R9 static re-query rows",
                "extend the 100-frame protocol horizon",
                "use posthoc future labels for runtime decisions",
                "promote simulated_from_gt to real-human evidence",
            ],
        },
        "next_action": "Freeze a larger legal causal interaction pool before another model-selection or production-interface round; keep the current trained checkpoint as an isolated development probe and do not authorize calibration, selector, decoder LoRA, or production promotion.",
    }
    atomic_write(OUTPUT, payload)
    print(json.dumps({
        "status": payload["status"],
        "frozen_events": len(frozen_events),
        "old_events": len(old_events),
        "extra_old_events_not_added": len(extra_records),
        "train_examples": train.get("example_count"),
        "validation_examples": validation.get("example_count"),
        "validation_future_positive_labels": validation.get("future_rows_selected_as_label"),
        "output": str(OUTPUT),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
