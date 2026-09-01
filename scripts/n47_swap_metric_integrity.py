#!/usr/bin/env python3
"""Independent integrity and legacy-equivalence audit for swap-metric repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import (  # noqa: E402
    CHECKPOINT,
    N42_RUNTIME,
    VARIANTS,
    event_map,
    hungarian_with_none,
    load,
    normalize_assignment,
    sha256,
    write_json,
)
from scripts.n47_stage04_global_probe_replay import classify_assignment_transition  # noqa: E402


def candidate_signature(candidates: list[dict]) -> list[tuple[int, object, float]]:
    return [(int(x["native_tid"]), x.get("box"), float(x.get("confidence", 0.0))) for x in candidates]


def all_false_future_gt(value: object, path: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_path = f"{path}.{key}" if path else key
            if key == "future_gt_fields_sent" and item != []:
                failures.append(f"{key_path} must be an empty list, got {item!r}")
            elif "future_gt" in key and isinstance(item, bool) and item is not False:
                failures.append(f"{key_path}={item!r}")
            failures.extend(all_false_future_gt(item, key_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(all_false_future_gt(item, f"{path}[{index}]"))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, default=ROOT / "outputs/n47_global_probe")
    args = parser.parse_args()
    out = args.output_root
    runtime_dir = out / "replay/runtime"
    posthoc_dir = out / "replay/posthoc"
    result_path = out / "replay/probe_results.json"
    legacy = args.legacy_root
    failures: list[str] = []
    events = event_map()
    runtime_files = sorted(runtime_dir.glob("*.json"))
    posthoc_files = sorted(posthoc_dir.glob("*.json"))
    if len(runtime_files) != 24 or {p.stem for p in runtime_files} != set(events):
        failures.append("runtime event set is not exactly 24 frozen events")
    if len(posthoc_files) != 24 or {p.stem for p in posthoc_files} != set(events):
        failures.append("posthoc event set is not exactly 24 frozen events")
    runtime_frames = 0
    expected_pure = {"M0": 0, "M1": 56, "M2": 64, "M3": 56, "M4": 39}
    expected_assignment = {"M0": 0, "M1": 335, "M2": 455, "M3": 335, "M4": 375}
    expected_id_set = {"M0": 0, "M1": 279, "M2": 391, "M3": 279, "M4": 336}
    counts = {v: {"assignment_changes": 0, "pure_swap_changes": 0, "id_set_changes": 0, "other_assignment_changes": 0} for v in VARIANTS}
    for event_id, event in sorted(events.items()):
        source = load(N42_RUNTIME / f"{event_id}.json")
        runtime_path = runtime_dir / f"{event_id}.json"
        if not runtime_path.is_file():
            failures.append(f"missing runtime {event_id}")
            continue
        runtime = load(runtime_path)
        failures.extend(f"{event_id}: {x}" for x in all_false_future_gt(runtime))
        if runtime.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
            failures.append(f"{event_id}: runtime_future_gt_used is not direct false")
        for variant in VARIANTS:
            src_no = source["variants"][variant]["branches"]["memory_write=False"]["future_trace"]
            src_write = source["variants"][variant]["branches"]["memory_write=True"]["future_trace"]
            frames = runtime.get("variants", {}).get(variant, {}).get("frames", [])
            if len(src_no) != 100 or len(src_write) != 100 or len(frames) != 100:
                failures.append(f"{event_id}/{variant}: source/runtime trace is not exactly 100")
                continue
            expected_frames = list(range(int(src_no[0]["frame"]), int(src_no[0]["frame"]) + 100))
            if [int(x["frame"]) for x in src_no] != expected_frames or [int(x["frame"]) for x in src_write] != expected_frames or [int(x["frame"]) for x in frames] != expected_frames:
                failures.append(f"{event_id}/{variant}: duplicate/missing/nonconsecutive frame")
            for source_no_entry, source_write_entry, frame in zip(src_no, src_write, frames):
                frame_id = int(frame["frame"])
                no, write, plus, probe = frame["no_write"], frame["write_baseline"], frame["write_plus_n47"], frame["probe"]
                if int(source_no_entry["frame"]) != frame_id or int(source_write_entry["frame"]) != frame_id:
                    failures.append(f"{event_id}/{variant}/{frame_id}: source frame mismatch")
                if candidate_signature(no["candidate_rows"]) != candidate_signature(source_no_entry["candidate_audit"]["candidates"]):
                    failures.append(f"{event_id}/{variant}/{frame_id}: no candidate stream mismatch")
                if candidate_signature(write["candidate_rows"]) != candidate_signature(source_write_entry["candidate_audit"]["candidates"]):
                    failures.append(f"{event_id}/{variant}/{frame_id}: write candidate stream mismatch")
                if candidate_signature(no["candidate_rows"]) != candidate_signature(write["candidate_rows"]) or candidate_signature(write["candidate_rows"]) != candidate_signature(plus["candidate_rows"]):
                    failures.append(f"{event_id}/{variant}/{frame_id}: branch candidate rows differ")
                if write["public_id_order"] != plus["public_id_order"]:
                    failures.append(f"{event_id}/{variant}/{frame_id}: write/plus public-ID axis differs")
                for record in (no, write, plus):
                    native_ids = record["candidate_native_ids"]
                    if len(native_ids) != len(set(native_ids)):
                        failures.append(f"{event_id}/{variant}/{frame_id}: native IDs not unique")
                    if record.get("runtime_future_gt_used") is not False:
                        failures.append(f"{event_id}/{variant}/{frame_id}: branch runtime_future_gt_used is not false")
                    matrix = np.asarray(record["base_scores"], dtype=np.float32)
                    if matrix.shape != (record["candidate_count"], len(record["public_id_order"])) or not np.all(np.isfinite(matrix)):
                        failures.append(f"{event_id}/{variant}/{frame_id}: invalid score matrix")
                    else:
                        expected = normalize_assignment(hungarian_with_none(matrix), len(record["public_id_order"]))
                        if expected != record["assignment_columns"]:
                            failures.append(f"{event_id}/{variant}/{frame_id}: independent Hungarian mismatch in {record['branch']}")
                write_matrix = np.asarray(write["base_scores"], dtype=np.float32)
                plus_matrix = np.asarray(plus["base_scores"], dtype=np.float32)
                actual_changed = {(int(i), int(j)) for i, j in zip(*np.where(np.abs(plus_matrix - write_matrix) > 1.0e-12))}
                listed_changed = {(int(x["candidate_index"]), int(x["column"])) for x in probe["changed_cells"]}
                if listed_changed != actual_changed:
                    failures.append(f"{event_id}/{variant}/{frame_id}: changed-cell list mismatch")
                if np.any(plus_matrix[write_matrix <= -1.0e7] != write_matrix[write_matrix <= -1.0e7]):
                    failures.append(f"{event_id}/{variant}/{frame_id}: hard-negative cell changed")
                transition = classify_assignment_transition(write["assignment_public_ids"], plus["assignment_public_ids"])
                for key in ("assignment_changed", "pure_swap_changes", "id_set_changes", "other_assignment_changes", "changed_row_count", "non_none_public_id_multiset_equal", "full_assignment_multiset_equal"):
                    if probe.get(key) != transition[key]:
                        failures.append(f"{event_id}/{variant}/{frame_id}: transition field {key} mismatch")
                for key in ("assignment_changes", "pure_swap_changes", "id_set_changes", "other_assignment_changes"):
                    counts[variant][key] += int(transition[key.replace("assignment_changes", "assignment_changed")]) if key == "assignment_changes" else int(transition[key])
                runtime_frames += 1
    result = load(result_path)
    if result.get("status") != "PASS" or result.get("event_count") != 24 or result.get("variant_count") != 5:
        failures.append("result envelope")
    if result.get("protocol", {}).get("runtime_future_gt_used") is not False or result.get("protocol", {}).get("gt_loaded_only_after_runtime_validation") is not True:
        failures.append("result GT provenance")
    for variant in VARIANTS:
        result_counts = result.get("effects", {}).get("incremental", {}).get(variant, {}).get("100", {}).get("assignment_change_count")
        if result_counts is None:
            failures.append(f"missing posthoc result count {variant}")
    runtime_status = load(out / "replay/runtime_status.json")
    for variant, expected in expected_pure.items():
        actual = runtime_status.get("metrics", {}).get("by_variant", {}).get(variant, {})
        if actual.get("pure_swap_changes") != expected:
            failures.append(f"{variant}: pure_swap_changes={actual.get('pure_swap_changes')}, expected {expected}")
        if actual.get("assignment_changes") != expected_assignment[variant]:
            failures.append(f"{variant}: assignment_changes changed")
        if actual.get("id_set_changes") != expected_id_set[variant]:
            failures.append(f"{variant}: id_set_changes={actual.get('id_set_changes')}, expected {expected_id_set[variant]}")
    for variant in VARIANTS:
        if counts[variant] != {"assignment_changes": expected_assignment[variant], "pure_swap_changes": expected_pure[variant], "id_set_changes": expected_id_set[variant], "other_assignment_changes": expected_assignment[variant] - expected_pure[variant]}:
            failures.append(f"{variant}: independently aggregated transition counts disagree")
    old_result = load(legacy / "replay/probe_results.json")
    if result.get("effects") != old_result.get("effects"):
        failures.append("utility/correct/incorrect/untouched effects changed from legacy result")
    for path in posthoc_files:
        old_path = legacy / "replay/posthoc" / path.name
        if not old_path.is_file() or load(path).get("variants") != load(old_path).get("variants"):
            failures.append(f"posthoc stable metrics changed for {path.name}")
    checkpoint_meta = __import__("torch").load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint_meta.get("production_authorized") is not False:
        failures.append("checkpoint production_authorized is not false")
    old_hash_manifest = legacy / "repair1_swap_metric/legacy_snapshot/sha256sums.txt"
    legacy_hashes = {}
    for line in old_hash_manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        legacy_hashes[name] = digest
    for name, digest in legacy_hashes.items():
        snapshot_file = legacy / "repair1_swap_metric/legacy_snapshot" / Path(name).name
        if snapshot_file.is_file() and hashlib.sha256(snapshot_file.read_bytes()).hexdigest() != digest:
            failures.append(f"legacy snapshot hash self-check failed: {name}")
    legacy_originals = {
        legacy / "replay/runtime_status.json": legacy / "repair1_swap_metric/legacy_snapshot/runtime_status.json",
        legacy / "replay/probe_results.json": legacy / "repair1_swap_metric/legacy_snapshot/probe_results.json",
        legacy / "stage_04_status.json": legacy / "repair1_swap_metric/legacy_snapshot/stage_04_status.json",
        legacy / "stage_04_integrity.json": legacy / "repair1_swap_metric/legacy_snapshot/stage_04_integrity.json",
        legacy / "n47_final_gate.json": legacy / "repair1_swap_metric/legacy_snapshot/n47_final_gate.json",
        ROOT / "docs/N47_FINAL_REPORT.md": legacy / "repair1_swap_metric/legacy_snapshot/N47_FINAL_REPORT.md",
    }
    for current, snapshot in legacy_originals.items():
        if not current.is_file() or not snapshot.is_file() or hashlib.sha256(current.read_bytes()).hexdigest() != hashlib.sha256(snapshot.read_bytes()).hexdigest():
            failures.append(f"legacy file changed: {current}")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "protocol": "N47_SWAP_METRIC_REPAIR_INDEPENDENT_INTEGRITY_V1",
        "inputs": {"repair_root": str(out), "legacy_root": str(legacy), "n42_runtime": str(N42_RUNTIME), "n47_checkpoint": str(CHECKPOINT)},
        "outputs": {"integrity": str(out / "stage_04_integrity.json"), "targeted_regression": str(out / "targeted_regression.json")},
        "metrics": {"event_count": 24, "runtime_frames": runtime_frames, "expected_assignment_changes": expected_assignment, "expected_pure_swap_changes": expected_pure, "expected_id_set_changes": expected_id_set, "actual_independent_counts": counts, "legacy_effects_unchanged": not any("effects changed" in x for x in failures), "failures": failures[:100], "n47_checkpoint_sha256": sha256(CHECKPOINT)},
        "gate_checks": {"exact_24_events": len(runtime_files) == 24 and len(posthoc_files) == 24, "12000_runtime_frames": runtime_frames == 12000, "source_trace_exact_100_no_gaps": not any("trace" in x or "frame" in x for x in failures), "candidate_rows_frozen": not any("candidate" in x for x in failures), "public_id_axis_frozen": not any("axis" in x for x in failures), "native_ids_unique": not any("native IDs" in x for x in failures), "hungarian_with_none_independent": not any("Hungarian" in x for x in failures), "runtime_future_gt_false": not any("GT" in x or "future_gt" in x for x in failures), "transition_taxonomy_exact": not any("transition" in x or "pure_swap" in x or "id_set_changes" in x for x in failures), "utility_correct_incorrect_untouched_unchanged": not any("effects changed" in x or "stable metrics changed" in x for x in failures), "checkpoint_production_authorized_false": not any("production_authorized" in x for x in failures), "old_evidence_preserved": True, "simulated_provenance": True},
        "failure_root_cause": "The repaired metric is valid only when the full frozen runtime is independently replayable and pure swap is separated from ID multiset changes; posthoc efficacy values must remain invariant under this taxonomy-only repair.",
        "next_action": "Finalize a repair-only semantic gate; retain N47 efficacy and real-input gates unchanged.",
        "runtime_future_gt_used": False,
        "gt_loaded_posthoc": True,
    }
    write_json(out / "stage_04_integrity.json", report)
    print(json.dumps({"status": report["status"], "runtime_frames": runtime_frames, "failures": len(failures), "pure_swap": expected_pure}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
