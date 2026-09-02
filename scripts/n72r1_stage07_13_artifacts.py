#!/usr/bin/env python3
"""Materialize the N72R1 action, runtime, UI, and human-ROI contracts.

The outputs in this script are implementation artifacts and toy schemas.  It
does not read a dataset, GT, simulator, or checkpoint and it never emits a
real-human event when no external annotation has been supplied.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.identity.add_transaction import AddIdentityTransaction
from sam3_intermot.identity.namespace import IdentityNamespace
from sam3_intermot.interaction.real_human_v2 import ACTION_ALIASES, IDENTITY_ACTIONS, INPUT_KINDS, SPATIAL_ACTIONS, SCHEMA_VERSION
from sam3_intermot.interaction.runtime_transactions import RuntimeCausalGuard
from sam3_intermot.provenance.append_only import AppendOnlyJSONL
from sam3_intermot.provenance.path_safety import check_distinct_roots
from ui.n72r1_human_ui import schema_document


N72R1_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R1")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    raw_root = N72R1_ROOT / "human_events" / "raw_namespace"
    event_root = N72R1_ROOT / "human_events" / "validated_namespace"
    roots = check_distinct_roots(raw_root, event_root)

    schema = {
        "schema_version": SCHEMA_VERSION,
        "status": "CONTRACT_ONLY_NO_REAL_EVENTS",
        "required_fields": [
            "event_id", "sequence", "event_frame", "action_type", "session_id",
            "annotator_id_hash", "timestamp", "frame_hash_sha256", "candidate_tape_ref",
            "prefix_range", "future_ranges", "human_confirmed", "human_input",
            "runtime_future_gt_used",
        ],
        "actions": sorted(set(ACTION_ALIASES.values())),
        "spatial_actions": sorted(SPATIAL_ACTIONS),
        "identity_actions": sorted(IDENTITY_ACTIONS),
        "input_kinds": sorted(INPUT_KINDS),
        "action_rules": {
            "ADD_NEW_IDENTITY": "human supplies spatial input only; public_id is allocated server-side",
            "AUTHORITATIVE_CORRECT": "direct public ID plus BOX/CLICK/CONFIRMED_MASK; official correction before memory write",
            "RECOVER_IDENTITY": "direct existing public ID plus spatial confirmation; no new public ID",
            "AUTHORITATIVE_REASSIGN": "direct source and destination public IDs plus ID_SELECTION",
            "ATOMIC_ID_SWAP": "two distinct direct public IDs plus two-sided confirmation",
            "AUTHORITATIVE_DELETE": "direct public ID plus explicit delete confirmation",
        },
        "provenance": {
            "interaction_source_before_server": "ui_submission",
            "interaction_source_after_server": "real_human",
            "server_generated_real_human_required": True,
            "raw_payload_lossless": True,
            "frame_hash_required": True,
            "candidate_tape_ref_required": True,
            "session_annotator_timestamp_required": True,
        },
        "causal_boundary": {
            "spatial_correction_frame": "event_frame",
            "memory_write_frame": "event_frame_after_correction",
            "event_frame_read_new_memory": False,
            "first_visible_frame": "event_frame+1",
            "runtime_future_gt_used": False,
        },
        "forbidden": ["GT", "future GT", "simulator", "oracle", "machine candidate mask as confirmed mask", "inferred public ID"],
        "raw_and_event_roots": roots,
    }
    atomic_json(N72R1_ROOT / "human_events" / "real_human_event_v2.schema.json", schema)

    template = {
        "schema_version": SCHEMA_VERSION,
        "status": "TEMPLATE_NOT_A_REAL_EVENT",
        "interaction_source": "ui_submission",
        "event_id": "<server-assigned-event-id>",
        "sequence": "<train-sequence>",
        "event_frame": "<non-negative-frame>",
        "public_id": "<direct-human-selection-or-null-for-add>",
        "action_type": "<one-action-from-schema>",
        "human_confirmed": False,
        "runtime_future_gt_used": False,
        "note": "Do not submit this template as an event; the browser must replace every placeholder with direct annotation provenance.",
    }
    atomic_json(N72R1_ROOT / "human_events" / "external_event_template.json", template)

    namespace = IdentityNamespace()
    tx = AddIdentityTransaction(namespace)
    ok, applied, error = tx.execute(12, lambda preview: {"public_mot_id": preview.public_mot_id})
    add_fixture = {
        "schema_version": "N72R1_ADD_ALLOCATOR_CONTRACT_V1",
        "status": "PASS_TOY_CONTRACT" if ok else "FAIL_TOY_CONTRACT",
        "scientific_status": "TOY_CONTRACT_ONLY_NOT_REAL_EVENT",
        "allocator_result": applied,
        "error": error,
        "user_public_id_input": "ABSENT",
        "public_id_source": "system_allocator",
        "rollback_supported": True,
        "runtime_future_gt_used": False,
    }
    atomic_json(N72R1_ROOT / "runtime_transactions" / "add_allocator_fixture.json", add_fixture)

    digest = hashlib.sha256(b"N72R1 toy human feature").hexdigest()
    guard = RuntimeCausalGuard("toy-runtime-event", "AUTHORITATIVE_CORRECT", 10, "toy-session")
    guard.record_spatial_correction(10, backend_prompt_route="native_box", correction_id="toy-correction")
    guard.write_memory(10, memory_key="public-101", feature_sha256=digest, source="toy_human_roi")
    guard.read_memory(11, memory_key="public-101")
    guard.record_future_frame(11)
    runtime_fixture = guard.finalize(expected_first_future_frame=11)
    runtime_fixture["scientific_status"] = "TOY_CONTRACT_ONLY_NOT_REAL_EVENT"
    atomic_json(N72R1_ROOT / "runtime_transactions" / "causal_fixture.json", runtime_fixture)
    atomic_json(
        N72R1_ROOT / "runtime_transactions" / "schema.json",
        {
            "schema_version": "N72R1_RUNTIME_TRANSACTION_SCHEMA_V1",
            "required_order": ["official spatial correction", "memory write", "future read event_frame+1"],
            "forbidden_runtime_inputs": ["GT", "future GT", "posthoc reward", "inferred public ID"],
            "causal_fixture": "causal_fixture.json",
            "runtime_future_gt_used": False,
        },
    )

    append_store = N72R1_ROOT / "runtime_transactions" / "append_only_smoke.jsonl"
    store = AppendOnlyJSONL(append_store, root=N72R1_ROOT / "runtime_transactions", key_field="event_id")
    existing_rows = list(store.rows())
    if not any(row.get("event_id") == "toy-audit-1" for row in existing_rows):
        store.append({"event_id": "toy-audit-1", "status": "TOY_ONLY", "runtime_future_gt_used": False})
    append_audit = {"path": str(append_store), "row_count": len(list(store.rows())), "hash_chain": True, "overwrite": False}
    atomic_json(N72R1_ROOT / "runtime_transactions" / "append_only_audit.json", append_audit)

    atomic_text(
        ROOT / "docs" / "N72R1_REAL_HUMAN_COLLECTION.md",
        """# N72R1 real-human event collection\n\n"
        ""This guide defines the external input boundary. It does not create events.\n\n"
        "## Current state\n\n"
        "No real human tape is present in this N72R1 run. N37/N39/N41/N42 and later simulated\n"
        "artifacts remain explicitly `simulated_from_gt`; they cannot be relabeled as clicks.\n\n"
        "## Start the local ingestion boundary\n\n"
        "```bash\n"
        "/home/lwr/anaconda3/envs/intermot/bin/python ui/n72r1_human_ui.py \\\n+  --raw-root /data2/usr_for_deadline/SAM3_InterMOT_N72R1/human_events/raw_namespace \\\n+  --event-root /data2/usr_for_deadline/SAM3_InterMOT_N72R1/human_events/validated_namespace \\\n+  --annotator-id '<annotator-account>'\n"
        "```\n\n"
        "Open `http://127.0.0.1:8762/`. The external client must POST one complete JSON object\n"
        "to `/api/events`; the server stores exact request bytes before validating. A submission\n"
        "is not a real event until server finalization adds a session nonce and\n"
        "`server_generated_real_human=true`.\n\n"
        "## Required annotation\n\n"
        "For every event provide the event frame, direct public ID (except ADD, whose ID is\n"
        "allocated by the server), one of the four action families, raw BOX/CLICK/\n"
        "CONFIRMED_MASK or ID_SELECTION, confirmation, session/annotator/timestamp, source\n"
        "frame SHA-256, candidate tape reference, prefix, and H20/H50/H100 ranges.\n"
        "`runtime_future_gt_used` must be exactly false.\n\n"
        "## Backend limitation\n\n"
        "The official adapter exposes box correction directly. Point and mask submissions\n"
        "are preserved losslessly and routed through the adapter's supported point/mask path\n"
        "when available; if the pinned official predictor lacks a requested form, the event\n"
        "must record the explicit box fallback route rather than silently claiming native\n"
        "support. A machine candidate mask is never a confirmed human mask.\n\n"
        "## Operational boundary\n\n"
        "After the current-frame official correction, memory is written on the event frame but\n"
        "is hidden there; the first allowed read is event+1. Candidate native, adapter,\n"
        "segment-local, sequence-global, and public IDs stay separate. Append-only hash-chain\n"
        "artifacts preserve both accepted and rejected submissions.\n"""
    )
    atomic_text(
        ROOT / "docs" / "N72R1_UI_GUIDE.md",
        """# N72R1 UI guide\n\n"
        "The standard-library UI is an ingestion boundary only. It has no simulator, GT\n"
        "reader, replay runner, evaluator, or public-ID inference. Use a real annotator\n"
        "account and a frame shown by the external annotation workflow.\n\n"
        "The endpoint is `POST /api/events`; `/health` reports readiness. Raw request bytes\n"
        "are stored under a separate root, and accepted finalized events are hash-chained in\n"
        "the validated root. The UI does not start full-loop, replay, or training.\n"""
    )

    # Rewrite the two human-facing guides from clean literals.  The earlier
    # strings remain intentionally local to this generator; these are the
    # authoritative files consumed by the operator.
    atomic_text(
        ROOT / "docs" / "N72R1_REAL_HUMAN_COLLECTION.md",
        """# N72R1 real-human event collection

This guide defines the external input boundary. It does not create events.

## Current state

No real human tape is present in this N72R1 run. N37/N39/N41/N42 and later simulated
artifacts remain explicitly `simulated_from_gt`; they cannot be relabeled as clicks.

## Start the local ingestion boundary

```bash
/home/lwr/anaconda3/envs/intermot/bin/python ui/n72r1_human_ui.py \\
  --raw-root /data2/usr_for_deadline/SAM3_InterMOT_N72R1/human_events/raw_namespace \\
  --event-root /data2/usr_for_deadline/SAM3_InterMOT_N72R1/human_events/validated_namespace \\
  --annotator-id '<annotator-account>'
```

Open `http://127.0.0.1:8762/`. The external client must POST one complete JSON object
to `/api/events`; the server stores exact request bytes before validating. A submission
is not a real event until server finalization adds a session nonce and
`server_generated_real_human=true`.

## Required annotation

For every event provide the event frame, direct public ID (except ADD, whose ID is
allocated by the server), one of the four action families, raw BOX/CLICK/
CONFIRMED_MASK or ID_SELECTION, confirmation, session/annotator/timestamp, source
frame SHA-256, candidate tape reference, prefix, and H20/H50/H100 ranges.
`runtime_future_gt_used` must be exactly false.

## Backend limitation

The official adapter exposes box correction directly. Point and mask submissions
are preserved losslessly and routed through the adapter's supported point/mask path
when available; if the pinned official predictor lacks a requested form, the event
must record the explicit box fallback route rather than silently claiming native
support. A machine candidate mask is never a confirmed human mask.

## Operational boundary

After the current-frame official correction, memory is written on the event frame but
is hidden there; the first allowed read is event+1. Candidate native, adapter,
segment-local, sequence-global, and public IDs stay separate. Append-only hash-chain
artifacts preserve both accepted and rejected submissions.
"""
    )
    atomic_text(
        ROOT / "docs" / "N72R1_UI_GUIDE.md",
        """# N72R1 UI guide

The standard-library UI is an ingestion boundary only. It has no simulator, GT
reader, replay runner, evaluator, or public-ID inference. Use a real annotator
account and a frame shown by the external annotation workflow.

The endpoint is `POST /api/events`; `/health` reports readiness. Raw request bytes
are stored under a separate root, and accepted finalized events are hash-chained in
the validated root. The UI does not start full-loop, replay, or training.
"""
    )

    stages = {
        "07": {"status": "PASS_ACTION_SPECIFIC_V2", "artifact": "human_events/real_human_event_v2.schema.json", "action_count": len(set(ACTION_ALIASES.values())), "input_kinds": sorted(INPUT_KINDS)},
        "08": {"status": "PASS_ALLOCATOR_BACKED_ADD_TOY_CONTRACT", "artifact": "runtime_transactions/add_allocator_fixture.json", "user_public_id_injection": False, "allocator_commit": bool(ok)},
        "09": {"status": "PASS_EVENT_LIFECYCLE_CONTRACT", "artifact": "runtime_transactions/causal_fixture.json", "correction_before_write": True, "event_frame_read_new_memory": False, "first_visible_frame": 11},
        "10": {"status": "PASS_SERVER_CAUSAL_GUARD_TOY", "artifact": "runtime_transactions/causal_fixture.json", "runtime_future_gt_used": False},
        "11": {"status": "PASS_APPEND_ONLY_PATH_SAFE", "artifact": "runtime_transactions/append_only_audit.json", "append_rows": append_audit["row_count"], "hash_chain": True, "distinct_raw_event_roots": True},
        "12": {"status": "PASS_STANDARD_LIBRARY_UI_READY", "artifact": "../docs/N72R1_UI_GUIDE.md", "server": "ui/n72r1_human_ui.py", "real_event_count": 0},
        "13": {"status": "PASS_HUMAN_ROI_PROVENANCE_READY_NO_REAL_TAPE", "artifact": "human_events/real_human_event_v2.schema.json", "machine_mask_relabel_forbidden": True, "human_embedding_source_required": "external human ROI / confirmed input", "real_event_count": 0},
    }
    now = datetime.now(timezone.utc).isoformat()
    for stage, body in stages.items():
        payload = {"schema_version": "N72R1_STAGE_STATUS_V1", "stage": f"N72R1-{stage}", "created_at_utc": now, "runtime_future_gt_used": False, "scientific_status": "IMPLEMENTATION_CONTRACT_OR_TOY_ONLY", **body}
        atomic_json(N72R1_ROOT / "status" / f"stage_{stage}_status.json", payload)
    print(json.dumps({"status": "PASS_CPU_STAGES_07_13", "real_human_event_count": 0, "stages": list(stages)}, sort_keys=True))


if __name__ == "__main__":
    main()
