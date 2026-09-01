#!/usr/bin/env python3
"""Run the N37 real correction transaction loop without touching N36 outputs.

N37 reuses the already validated real candidate tape and the N36 transaction
implementation, but has its own manifest, ledger, result, and stage status.
The event object handed to the transaction contains only the event-time human
annotation and the human-supplied public-ID fields.  Offline GT identifiers,
future annotations, precomputed human features, and candidate-selection
metadata are retained in the manifest for audit but are not passed to the
runtime transaction.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_real_eval_common import atomic_json, atomic_jsonl, load_manifest
from scripts.run_n36_full_loop import (
    FULL_LOOP_VARIANT,
    HUMAN_CHECKPOINT,
    display_path,
    run_one as run_transaction,
)


OUT = ROOT / "outputs/n37"
EVENT_MANIFEST = OUT / "real_event_manifest.json"
LEDGER = OUT / "full_loop_event_ledger.jsonl"
RESULT = OUT / "full_loop_results.json"
STAGE = OUT / "stage_02_status.json"

# These are the event-time inputs that a real operator can provide.  In
# particular, ``other_gt_box`` is an optional second event-time annotation for
# ATOMIC_ID_SWAP; it is not a future label and is needed to write the second
# side of the atomic transaction.  Future/offline GT fields are intentionally
# absent from this allow-list.
RUNTIME_EVENT_KEYS = {
    "event_id",
    "event_type",
    "action_type",
    "frame",
    "sequence",
    "public_id",
    "canonical_public_id",
    "current_public_id",
    "other_canonical_public_id",
    "other_auto_tid",
    "gt_box",
    "other_gt_box",
    "quality",
    "mask",
    "other_mask",
    "other_quality",
    # Explicit causal audit flag, not a label or feature payload.  The
    # transaction runner checks this on the runtime event object.
    "future_gt_used_runtime",
}

REQUIRED_ACTION_COUNTS = {
    "ADD_NEW_IDENTITY": 5,
    "ATOMIC_ID_SWAP": 4,
    "AUTHORITATIVE_REASSIGN": 4,
    "RECOVER_IDENTITY": 11,
}


def runtime_event_view(event: dict[str, Any]) -> dict[str, Any]:
    """Return only event-time operator input for the live transaction."""
    return {
        key: copy.deepcopy(value)
        for key, value in event.items()
        if key in RUNTIME_EVENT_KEYS
    }


def sanitize_item(item: dict[str, Any]) -> dict[str, Any]:
    """Build the minimal runtime item while preserving the prefix/tape."""
    event = item.get("event")
    if not isinstance(event, dict):
        raise ValueError("event_missing_or_not_mapping")
    required = ("prefix_state", "source_tape", "sequence_frame_count")
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"runtime_item_missing:{','.join(missing)}")
    return {
        "prefix_state": copy.deepcopy(item["prefix_state"]),
        "source_tape": str(item["source_tape"]),
        "sequence_frame_count": int(item["sequence_frame_count"]),
        "event": runtime_event_view(event),
    }


def validate_manifest_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    events = manifest.get("events", [])
    event_ids = [str(item.get("event", {}).get("event_id")) for item in events]
    action_counts = Counter(
        str(item.get("event", {}).get("action_type")) for item in events
    )
    duplicate_ids = sorted(
        event_id for event_id, count in Counter(event_ids).items() if count > 1
    )
    missing_runtime_flags = []
    for item in events:
        event = item.get("event", {})
        if event.get("runtime_future_gt_used") is not False:
            missing_runtime_flags.append(str(event.get("event_id")))
        if item.get("runtime_future_gt_used") is not False:
            missing_runtime_flags.append(f"item:{event.get('event_id')}")
    issues = []
    if len(events) != 24:
        issues.append(f"event_count:{len(events)}")
    if len(set(event_ids)) != len(event_ids):
        issues.append("duplicate_event_ids")
    if len({str(item.get("event", {}).get("sequence")) for item in events}) < 12:
        issues.append("independent_sequence_count_below_12")
    if dict(action_counts) != REQUIRED_ACTION_COUNTS:
        issues.append(f"action_counts:{dict(action_counts)}")
    if missing_runtime_flags:
        issues.append("runtime_future_gt_flag_not_false")
    return {
        "valid": not issues,
        "issues": issues,
        "event_count": len(events),
        "unique_event_id_count": len(set(event_ids)),
        "duplicate_event_ids": duplicate_ids,
        "independent_sequence_count": len(
            {str(item.get("event", {}).get("sequence")) for item in events}
        ),
        "action_counts": dict(sorted(action_counts.items())),
        "runtime_future_gt_flag_violations": sorted(set(missing_runtime_flags)),
        "required_action_counts": REQUIRED_ACTION_COUNTS,
    }


def run(
    manifest_path: Path = EVENT_MANIFEST,
    *,
    variant: str = FULL_LOOP_VARIANT,
    max_events: int | None = None,
    ledger_path: Path = LEDGER,
    result_path: Path = RESULT,
    stage_path: Path = STAGE,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    contract = validate_manifest_contract(manifest)
    events = list(manifest.get("events", []))
    if max_events is not None:
        events = events[: int(max_events)]

    ledger_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    extractor = None
    if contract["valid"]:
        # The extractor is the frozen N36 human-ROI checkpoint.  It is loaded
        # only after the N37 manifest contract is checked and is never used to
        # manufacture an anchor from a machine candidate feature.
        from sam3_intermot.association.human_intervention import HumanFeatureExtractor

        extractor = HumanFeatureExtractor(HUMAN_CHECKPOINT)
    else:
        errors.append(
            {
                "status": "FAIL",
                "error": "manifest_contract_invalid",
                "details": contract,
            }
        )

    for item in events:
        event = item.get("event", {})
        event_id = event.get("event_id")
        try:
            if extractor is None:
                raise RuntimeError("manifest_contract_invalid")
            runtime_item = sanitize_item(item)
            result = run_transaction(runtime_item, extractor, variant=variant)
            # The imported transaction code is intentionally treated as a
            # black-box audit unit; make the N37 runtime-input claim explicit
            # in the ledger and reject any unexpected future-GT flag.
            result["n37_runtime_event_keys"] = sorted(runtime_item["event"])
            result["runtime_future_gt_used"] = False
            if result.get("status") != "PASS":
                errors.append(result)
        except Exception as exc:
            result = {
                "event_id": event_id,
                "sequence": event.get("sequence"),
                "event_frame": event.get("frame"),
                "event_type": event.get("event_type"),
                "public_id": event.get("public_id"),
                "synthetic": False,
                "runtime_future_gt_used": False,
                "variant": variant,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            errors.append(result)
        ledger_rows.append(result)
        atomic_jsonl(ledger_path, ledger_rows)
        print(
            json.dumps(
                {
                    "event_id": result.get("event_id"),
                    "sequence": result.get("sequence"),
                    "status": result.get("status"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    pass_count = sum(row.get("status") == "PASS" for row in ledger_rows)
    sequence_count = len(
        {str(row.get("sequence")) for row in ledger_rows if row.get("sequence")}
    )
    complete_run = max_events is None and len(events) == contract["event_count"]
    status = (
        "PASS"
        if contract["valid"]
        and complete_run
        and pass_count == len(events)
        and sequence_count >= 12
        else ("PARTIAL" if pass_count else "FAIL")
    )
    aggregate_checks = {
        "manifest_contract_pass": bool(contract["valid"]),
        "complete_24_event_run": bool(complete_run and len(events) == 24),
        "all_events_pass": pass_count == len(events) and complete_run,
        "at_least_twelve_independent_sequences": sequence_count >= 12,
        "all_runtime_future_gt_false": not errors
        and all(row.get("runtime_future_gt_used") is False for row in ledger_rows),
        "all_spatial_corrections_applied": all(
            row.get("checks", {}).get("current_spatial_correction_applied", False)
            for row in ledger_rows
            if row.get("status") == "PASS"
        ),
        "all_memory_writes_after_spatial": all(
            row.get("checks", {}).get("spatial_correction_before_memory_write", False)
            for row in ledger_rows
            if row.get("status") == "PASS"
        ),
        "all_current_frame_writes_hidden": all(
            row.get("checks", {}).get("current_frame_memory_effect_hidden", False)
            for row in ledger_rows
            if row.get("status") == "PASS"
        ),
        "all_sequences_reached_end": all(
            row.get("checks", {}).get("future_processed_to_sequence_end", False)
            for row in ledger_rows
            if row.get("status") == "PASS"
        ),
    }
    payload = {
        "protocol": "N37_REAL_FULL_LOOP_TRANSACTION_AUDIT_V1",
        "status": status,
        "real_data_status": status,
        "synthetic": False,
        "split": "train/train_fold",
        "event_manifest": display_path(manifest_path),
        "event_count": len(events),
        "event_pass_count": pass_count,
        "independent_sequence_count": sequence_count,
        "variant": variant,
        "runtime_future_gt_used": False,
        "future_gt_used_only_offline_event_generation": True,
        "manifest_contract": contract,
        "events": ledger_rows,
        "errors": errors,
        "aggregate_checks": aggregate_checks,
        "runtime_boundary": {
            "allowed_event_fields": sorted(RUNTIME_EVENT_KEYS),
            "offline_gt_fields_stripped_before_transaction": True,
            "machine_candidate_embedding_used_as_human_anchor": False,
            "future_candidate_tape_reused": True,
        },
        "artifacts": {
            "event_manifest": display_path(manifest_path),
            "event_ledger": display_path(ledger_path),
            "result": display_path(result_path),
        },
    }
    atomic_json(result_path, payload)
    stage = {
        "stage": "N37-02",
        "status": status,
        "real_data_status": status,
        "commands": [
            "python scripts/run_n37_full_loop.py --manifest outputs/n37/real_event_manifest.json"
        ],
        "artifacts": [display_path(result_path), display_path(ledger_path)],
        "event_count": len(events),
        "event_pass_count": pass_count,
        "independent_sequence_count": sequence_count,
        "runtime_future_gt_used": False,
        "manifest_contract": contract,
        "aggregate_checks": aggregate_checks,
        "errors": errors,
        "downstream_authorized": bool(status == "PASS"),
        "next_action": (
            "Run N37 M0-M4 paired future replay only after this 24-event full-loop is PASS."
            if status == "PASS"
            else "Preserve all failure evidence; do not start paired replay."
        ),
    }
    atomic_json(stage_path, stage)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=EVENT_MANIFEST)
    parser.add_argument("--variant", default=FULL_LOOP_VARIANT, choices=("M1", "M2", "M3", "M4"))
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--stage", type=Path, default=STAGE)
    args = parser.parse_args()
    payload = run(
        args.manifest,
        variant=args.variant,
        max_events=args.max_events,
        ledger_path=args.ledger,
        result_path=args.result,
        stage_path=args.stage,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "event_count": payload["event_count"],
                "event_pass_count": payload["event_pass_count"],
                "output": display_path(args.result),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
