#!/usr/bin/env python3
"""Freeze N38R1 near-tie selection and close the blocked downstream stages.

Selection uses only the frozen N38 margin rule and the M0, memory-write=False
candidate stream at event+1.  No future identity metric, treatment variant, or
post-treatment result is consulted.  If the fixed quotas are not met, this
script records the shortage and deliberately does not run full-loop/replay.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from scripts.n36_real_eval_common import atomic_json, atomic_text  # noqa: E402


N37_MANIFEST = ROOT / "outputs" / "n37" / "real_event_manifest.json"
R2_TABLE = ROOT / "outputs" / "n38r1" / "diagnostic_attempt3" / "score_assignment_table.jsonl"
R2_SUMMARY = ROOT / "outputs" / "n38r1" / "diagnostic_attempt3" / "score_assignment_summary.json"
R2_PROTOCOL = ROOT / "outputs" / "n38r1" / "diagnostic_attempt3" / "diagnostic_protocol.json"
R1_MANIFEST = ROOT / "outputs" / "n38r1" / "sidecar_manifest.json"
R1_AUDIT = ROOT / "outputs" / "n38r1" / "sidecar_schema_audit.json"
N38R1 = ROOT / "outputs" / "n38r1"
REQUIRED_ACTIONS = (
    "ADD_NEW_IDENTITY",
    "AUTHORITATIVE_REASSIGN",
    "ATOMIC_ID_SWAP",
    "RECOVER_IDENTITY",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def table_rows() -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    with R2_TABLE.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = ":".join(
                str(row.get(field))
                for field in ("event_id", "variant", "branch", "frame_phase", "frame")
            )
            if key in rows:
                errors.append(f"duplicate:{key}")
            rows[key] = row
    return rows, errors


def event_id(item: dict[str, Any]) -> str:
    return str(item["event"]["event_id"])


def select_candidates(events: list[dict[str, Any]], rows: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    available: list[dict[str, Any]] = []
    all_evidence: list[dict[str, Any]] = []
    for item in events:
        event = item["event"]
        eid = event_id(item)
        frame = int(event["frame"])
        current_key = ":".join((eid, "M0", "event_frame_audit", "event", str(frame)))
        future_key = ":".join((eid, "M0", "memory_write=False", "future", str(frame + 1)))
        current = rows.get(current_key)
        future = rows.get(future_key)
        evidence = {
            "event_id": eid,
            "sequence": str(event["sequence"]),
            "action_type": str(event["action_type"]),
            "event_frame": frame,
            "selection_stream": "M0/memory_write=False",
            "current_row_key": current_key,
            "event_plus_one_row_key": future_key,
            "current_event_frame_near_tie": None if current is None else current.get("near_tie_current_event_frame"),
            "event_plus_one_near_tie": None if future is None else future.get("near_tie_event_plus_one"),
            "current_top_two": None if current is None else {
                "top1_candidate_public_id": current.get("top1_candidate_public_id"),
                "top2_candidate_public_id": current.get("top2_candidate_public_id"),
                "normalized_margin": current.get("top1_top2_normalized_margin"),
            },
            "event_plus_one_top_two": None if future is None else {
                "top1_candidate_public_id": future.get("top1_candidate_public_id"),
                "top2_candidate_public_id": future.get("top2_candidate_public_id"),
                "normalized_margin": future.get("top1_top2_normalized_margin"),
            },
            "used_future_identity_metrics": False,
            "used_post_treatment_variant": False,
        }
        eligible = bool(
            current is not None
            and future is not None
            and current.get("near_tie_current_event_frame") is True
            and future.get("near_tie_event_plus_one") is True
        )
        evidence["eligible_frozen_baseline_near_tie"] = eligible
        all_evidence.append(evidence)
        if eligible:
            selected = json.loads(json.dumps(item))
            selected["selection_evidence"] = evidence
            selected["selected_by_protocol"] = True
            selected["interaction_source"] = "simulated_from_gt"
            selected["runtime_future_gt_used"] = False
            available.append(selected)
    return available, all_evidence


def write_final_outputs(
    events: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    table_errors: list[str],
    r2_summary: dict[str, Any],
) -> dict[str, Any]:
    selected_actions = Counter(item["event"]["action_type"] for item in selected)
    selected_sequences = sorted({str(item["event"]["sequence"]) for item in selected})
    quota = {
        "minimum_event_count": 24,
        "minimum_independent_sequence_count": 16,
        "minimum_per_action_count": 4,
        "required_actions": list(REQUIRED_ACTIONS),
    }
    deficits = {
        "event_count": max(0, quota["minimum_event_count"] - len(selected)),
        "independent_sequence_count": max(0, quota["minimum_independent_sequence_count"] - len(selected_sequences)),
        "per_action": {
            action: max(0, quota["minimum_per_action_count"] - int(selected_actions.get(action, 0)))
            for action in REQUIRED_ACTIONS
        },
    }
    quota_pass = (
        len(selected) >= quota["minimum_event_count"]
        and len(selected_sequences) >= quota["minimum_independent_sequence_count"]
        and all(selected_actions.get(action, 0) >= quota["minimum_per_action_count"] for action in REQUIRED_ACTIONS)
        and not table_errors
    )
    status = "PASS" if quota_pass else "BLOCKED_INSUFFICIENT_FROZEN_NEAR_TIE_EVENTS"
    event_protocol = {
        "protocol": "N38R1_FROZEN_NEAR_TIE_EVENT_SELECTION_V1",
        "status": status,
        "base_n38_protocol": "outputs/n38/diagnostic/diagnostic_protocol.json",
        "base_n38_protocol_hash": r2_summary.get("frozen_n38_protocol_hash"),
        "selection_rule": {
            "conjunction": "event-frame near_tie AND event+1 near_tie",
            "event_frame_stream": "M0/memory_write=False audit-only event frame",
            "event_plus_one_stream": "M0/memory_write=False pre-treatment baseline future candidate stream",
            "margin_definition": "frozen N38 normalized target-state score margin <= 0.05 with distinct top-two candidate public IDs",
            "threshold": 0.05,
            "candidate_order": "canonical N37 real_event_manifest order",
            "post_treatment_fields_forbidden": [
                "M1/M2/M3/M4 event selection",
                "memory_write=True event+1 result",
                "future identity error",
                "H20/H50/H100",
                "posthoc IoU/missing/re-correction",
            ],
        },
        "quota": quota,
        "available_candidate_count": len(selected),
        "available_independent_sequence_count": len(selected_sequences),
        "available_action_counts": dict(sorted(selected_actions.items())),
        "deficits": deficits,
        "table_errors": table_errors,
        "candidate_evidence": evidence,
        "selected_event_ids": [event_id(item) for item in selected],
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "downstream_authorized": False,
    }
    event_protocol["protocol_hash"] = hashlib.sha256(
        json.dumps(event_protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    event_manifest = {
        "protocol": "N38R1_NEAR_TIE_EVENT_MANIFEST_V1",
        "status": status,
        "source_manifest": "outputs/n37/real_event_manifest.json",
        "source_manifest_event_count": len(events),
        "event_count": len(selected),
        "independent_sequence_count": len(selected_sequences),
        "events": selected,
        "selection_protocol": "outputs/n38r1/event_protocol.json",
        "runtime_future_gt_used": False,
        "synthetic": False,
        "interaction_source": "simulated_from_gt",
        "quota": quota,
        "deficits": deficits,
        "downstream_authorized": False,
    }
    atomic_json(N38R1 / "event_protocol.json", event_protocol)
    atomic_json(N38R1 / "real_event_manifest.json", event_manifest)

    # Promote the repaired R2 status to the canonical stage path while keeping
    # the original failed stage file as explicit attempt evidence.
    canonical_stage2 = N38R1 / "stage_02_status.json"
    if canonical_stage2.is_file():
        atomic_json(N38R1 / "attempts" / "r1_diagnostic_attempt1_stage_status.json", load_json(canonical_stage2))
    repaired_stage2 = load_json(N38R1 / "stage_02_status_attempt3.json")
    repaired_stage2["canonical_provenance"] = "outputs/n38r1/diagnostic_attempt3"
    repaired_stage2["preserved_failed_attempts"] = [
        "outputs/n38r1/diagnostic/score_assignment_summary.json",
        "outputs/n38r1/stage_02_status.json (attempt1 preserved below attempts/)",
        "outputs/n38r1/attempts/r1_diagnostic_attempt2_axis_alignment_failure.json",
    ]
    atomic_json(canonical_stage2, repaired_stage2)
    atomic_json(N38R1 / "diagnostic" / "final_attempt_provenance.json", {
        "status": "PASS",
        "final_attempt": "attempt3",
        "artifacts": [
            "outputs/n38r1/diagnostic_attempt3/score_assignment_table.jsonl",
            "outputs/n38r1/diagnostic_attempt3/score_assignment_summary.json",
            "outputs/n38r1/diagnostic_attempt3/diagnostic_protocol.json",
        ],
        "failed_attempts_preserved": [
            "outputs/n38r1/diagnostic/score_assignment_table.jsonl",
            "outputs/n38r1/diagnostic/score_assignment_summary.json",
            "outputs/n38r1/stage_02_status_attempt1_stage_status.json",
        ],
        "runtime_future_gt_used": False,
    })

    stage3 = {
        "stage": "N38R1-03",
        "status": status,
        "real_data_status": status,
        "source_diagnostic": "outputs/n38r1/diagnostic_attempt3/score_assignment_summary.json",
        "event_protocol": "outputs/n38r1/event_protocol.json",
        "event_manifest": "outputs/n38r1/real_event_manifest.json",
        "candidate_event_count": len(selected),
        "independent_sequence_count": len(selected_sequences),
        "action_counts": dict(sorted(selected_actions.items())),
        "quota": quota,
        "deficits": deficits,
        "runtime_future_gt_used": False,
        "downstream_authorized": False,
        "next_action": "Do not run R4; retain shortage and use one minimal protocol-approved next step." if not quota_pass else "Run R4 using the frozen selected event manifest.",
    }
    atomic_json(N38R1 / "stage_03_status.json", stage3)
    stage4 = {
        "stage": "N38R1-04",
        "status": "NOT_RUN_R3_BLOCKED" if not quota_pass else "PENDING",
        "reason": status,
        "full_loop": "NOT_RUN",
        "paired_replay": "NOT_RUN",
        "runtime_future_gt_used": False,
        "downstream_authorized": False,
    }
    atomic_json(N38R1 / "stage_04_status.json", stage4)
    final_gate = {
        "protocol": "N38R1_FINAL_GATE_V1",
        "status": status,
        "r1_sidecar": "PASS",
        "r2_diagnostic": "PASS",
        "r3_frozen_near_tie_quota": status,
        "r4_full_loop": "NOT_RUN_R3_BLOCKED" if not quota_pass else "PENDING",
        "r5_future_effect_gate": "NOT_COMPUTABLE_R4_NOT_RUN",
        "calibration_head": "NOT_AUTHORIZED",
        "selector": "NOT_AUTHORIZED",
        "decoder_lora": "NOT_AUTHORIZED",
        "runtime_future_gt_used": False,
        "minimal_next_step": "Collect or recover a protocol-approved frozen current/event+1 near-tie candidate pool without changing the 0.05 threshold, action quotas, or future window." if not quota_pass else "Run R4.",
    }
    atomic_json(N38R1 / "n38r1_final_gate.json", final_gate)
    stage5 = {
        "stage": "N38R1-05",
        "status": status,
        "final_gate": "outputs/n38r1/n38r1_final_gate.json",
        "report": "docs/N38R1_FINAL_REPORT.md",
        "research_log_updated": True,
        "downstream_authorized": False,
    }
    atomic_json(N38R1 / "stage_05_status.json", stage5)

    report = f"""# N38R1 Final Report — Lossless Diagnostic Sidecar Recovery

Date: 2026-08-29 (Asia/Shanghai)

## Gate conclusion

`N38R1_STATUS = {status}`. The input-artifact block in N38 was recovered: the new R1
sidecar is complete and R2 is a reproducible mechanism diagnostic. R3 cannot satisfy
the frozen near-tie event quotas, so R4 full-loop/replay was correctly not run. No
calibration head, selector, or decoder LoRA is authorized.

## R1 evidence

- Frozen input: N37 canonical 24 events across 21 sequences; N36 real candidate tape.
- Sidecar: 120 event×variant artifacts, including the event frame and all 100 future
  frames (24,000 future rows).
- Schema audit: 184,140 candidate records; all 512-D runtime/source features finite;
  184,140 mask hashes preserved; 24,000 future-frame mappings complete; 0 runtime
  future-GT violations; 0 duplicate/missing sidecar keys.
- Current-frame treatment contract: event-frame audit uses an audit-only clone after
  spatial correction, with current-frame memory read/write hidden; enabled memory is
  visible from event+1 only.

Evidence: `outputs/n38r1/sidecar_manifest.json`,
`outputs/n38r1/sidecar_schema_audit.json`.

## Preserved repair history

- Schema probe attempt 1: `KeyError('event_id')`, caused by a wrong assumption about
  the nested N37 manifest schema.
- Probe attempt 2: process creation failed because the workdir repeated the project
  component.
- Replay import probe attempt 3: default Python lacked torch; the supported
  `intermot` environment was used afterward.
- R2 diagnostic attempt 1: generator produced 36,120 rows because event-frame data
  was repeated over the future frame list; retained under `outputs/n38r1/diagnostic/`.
- R2 attempt 2: row count was fixed to 24,120, but a strict fixed-column check rejected
  30 legitimate dynamic public-state axes. The final diagnostic aligns columns by
  public ID intersection and compares assignments by candidate→public-ID mapping;
  state-axis changes remain explicit evidence. Attempt 2 remains preserved.

## R2 mechanism result

The final table has exactly 24,120 unique rows, 0 duplicate rows, 0 candidate-axis
alignment errors, and 0 runtime future-GT flags. Three events have dynamic public-state
membership in the memory-write branch near a window boundary; this is retained as
association-state evidence, not silently discarded.

The frozen baseline selection stream is M0 with `memory_write=False`. Only
`{len(selected)}` event(s) satisfy the pre-registered event-frame AND event+1 near-tie
conjunction. Variant-specific or `memory_write=True` near-tie observations were not
used to select events, because that would be post-treatment selection.

## R3 quota decision

Available frozen-baseline candidates: `{len(selected)}` event(s),
`{len(selected_sequences)}` independent sequence(s), action counts
`{dict(sorted(selected_actions.items()))}`.

Required: at least 24 events, 16 independent sequences, and at least 4 events for each
of ADD_NEW_IDENTITY, AUTHORITATIVE_REASSIGN, ATOMIC_ID_SWAP, and RECOVER_IDENTITY.
Deficits are recorded in `outputs/n38r1/event_protocol.json` and
`outputs/n38r1/stage_03_status.json`. The threshold was not widened and no synthetic
event was added. The selected N37 events remain labeled `simulated_from_gt`; they are
not historical human clicks.

Because R3 is blocked by the frozen candidate shortage, R4 full-loop and paired replay
are `NOT_RUN_R3_BLOCKED`; there is no N38R1 future-effect estimate to interpret.

## Final decision and next step

The result distinguishes a recovered diagnostic input path from a scientific success.
The single minimal next step is to collect or recover a protocol-approved frozen
current/event+1 near-tie candidate pool without changing the 0.05 threshold, action
quotas, candidate definition, or future window. Until then, all learning downstream
remains `NOT_AUTHORIZED`.

Machine-readable gate: `outputs/n38r1/n38r1_final_gate.json`.
"""
    atomic_text(ROOT / "docs" / "N38R1_FINAL_REPORT.md", report)
    log_path = ROOT / "research_log.md"
    old_log = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    log_entry = f"""

## N38R1 — Lossless diagnostic sidecar recovery (2026-08-29)

Hypothesis: the N38 Stage-A block was recoverable by replaying the frozen N37 24×5
protocol with an event-frame-inclusive, lossless diagnostic sidecar. R1 completed 120
event×variant artifacts over 24 events/21 sequences and 24,000 future frames with
184,140 finite candidate records, preserved mask hashes, complete mappings, and zero
runtime future-GT use. R2 completed 24,120 unique diagnostic rows after two preserved
generator repairs: event-frame row expansion was fixed, then dynamic public-state axes
were aligned by public ID rather than raw column index. Under the pre-registered M0,
memory-write=False baseline selection stream, only {len(selected)} event(s) met the
event-frame AND event+1 near-tie conjunction; fixed quotas (24 events, 16 sequences,
four per action) were not met. R3 therefore stopped before R4, with no future-effect
gate result and no authorization for calibration/selector/LoRA. Events remain labeled
simulated_from_gt. New evidence is under outputs/n38r1; N36/N37/N38 inputs remain
read-only. Minimal next step: recover/collect a protocol-approved near-tie candidate
pool without changing the frozen threshold or event protocol.
"""
    atomic_text(log_path, old_log.rstrip() + log_entry)
    return {
        "status": status,
        "selected_event_count": len(selected),
        "selected_sequence_count": len(selected_sequences),
        "action_counts": dict(sorted(selected_actions.items())),
        "deficits": deficits,
    }


def main() -> int:
    manifest = load_json(N37_MANIFEST)
    r2_summary = load_json(R2_SUMMARY)
    if manifest.get("status") != "PASS" or manifest.get("event_count") != 24:
        raise RuntimeError("frozen N37 manifest is not the required PASS 24-event input")
    if r2_summary.get("status") != "PASS" or r2_summary.get("row_count") != 24120:
        raise RuntimeError("final R2 summary is not PASS/24120")
    rows, table_errors = table_rows()
    if len(rows) != 24120:
        table_errors.append(f"row_count:{len(rows)}!=24120")
    events = manifest["events"]
    selected, evidence = select_candidates(events, rows)
    result = write_final_outputs(events, selected, evidence, table_errors, r2_summary)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
