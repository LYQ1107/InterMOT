#!/usr/bin/env python3
"""Full integrity check for the isolated normalized N45 attribution repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
RUNTIME = ROOT / "outputs/n46/diagnosis_repair2/events"
POSTHOC = ROOT / "outputs/n46/n45_attribution_repair/posthoc_events"
RESULT = ROOT / "outputs/n46/n45_attribution_repair/normalized_attribution_results.json"
OLD = ROOT / "outputs/n45/replay/attribution_results.json"
CHECKPOINT = ROOT / "outputs/n44/training/n44_assignment_aware.pt"
OUT = ROOT / "outputs/n46/n45_attribution_repair/full_integrity.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
HORIZONS = (20, 50, 100)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ids = sorted(str(x["event"]["event_id"]) for x in load(EVENTS)["events"])
    runtime_files = sorted(RUNTIME.glob("*.json")); posthoc_files = sorted(POSTHOC.glob("*.json")); failures = []
    if len(runtime_files) != 24 or {x.stem for x in runtime_files} != set(ids): failures.append("runtime event set")
    if len(posthoc_files) != 24 or {x.stem for x in posthoc_files} != set(ids): failures.append("posthoc event set")
    runtime_rows = posthoc_frames = 0
    expected_evaluated = {(effect, v, h): 0 for effect in ("memory", "incremental") for v in VARIANTS for h in HORIZONS}
    for eid in ids:
        r = load(RUNTIME / f"{eid}.json"); p = load(POSTHOC / f"{eid}.json")
        if r.get("runtime_future_gt_used") is not False or p.get("runtime_future_gt_used") is not False or p.get("gt_loaded_posthoc") is not True: failures.append(f"{eid}: provenance envelope")
        if set(r.get("variants", {})) != set(VARIANTS) or set(p.get("variants", {})) != set(VARIANTS): failures.append(f"{eid}: variants")
        for v in VARIANTS:
            frames = r["variants"][v]; runtime_rows += len(frames)
            if len(frames) != 100: failures.append(f"{eid}/{v}: runtime frame count")
            if [int(x["frame"]) for x in frames] != list(range(int(frames[0]["frame"]), int(frames[0]["frame"]) + len(frames))): failures.append(f"{eid}/{v}: runtime frame gap")
            if any(x.get("runtime_future_gt_used") is not False for x in frames): failures.append(f"{eid}/{v}: runtime GT flag")
            if v == "M0" and any(x["proposals"] or x["changed_cells"] or int(x["assignment_changed_count"]) for x in frames): failures.append(f"{eid}/{v}: M0 sidecar activity")
            horizons = p["variants"][v].get("horizons", {})
            if set(horizons) != {str(h) for h in HORIZONS}: failures.append(f"{eid}/{v}: horizons")
            for h in HORIZONS:
                for effect in ("memory_effect_no_write_to_write_baseline", "n44_incremental_effect_write_baseline_to_write_plus_n44"):
                    item = horizons[str(h)][effect]; posthoc_frames += int(item.get("evaluated_frames", 0)); expected_evaluated[("memory" if effect.startswith("memory") else "incremental", v, h)] += int(item.get("evaluated_frames", 0))
                    needed = ("identity_utility", "target_iou_delta", "future_identity_error_reduction", "recorrection_proxy_reduction", "assignment_change_count", "assignment_change_correct_count", "assignment_change_incorrect_count", "assignment_change_neutral_count", "assignment_no_change_count", "untouched_regression", "frame_details")
                    if any(k not in item for k in needed): failures.append(f"{eid}/{v}/H{h}/{effect}: metric schema")
                    if int(item.get("assignment_change_count", 0)) != int(item.get("assignment_change_correct_count", 0)) + int(item.get("assignment_change_incorrect_count", 0)) + int(item.get("assignment_change_neutral_count", 0)): failures.append(f"{eid}/{v}/H{h}/{effect}: decomposition")
    result = load(RESULT); old_hash = sha256(OLD); checkpoint_hash = sha256(CHECKPOINT)
    if result.get("status") != "PASS" or result.get("event_count") != 24 or result.get("variant_count") != 5: failures.append("result envelope")
    if result.get("protocol", {}).get("runtime_future_gt_used") is not False or result.get("protocol", {}).get("gt_loaded_only_after_runtime_validation") is not True: failures.append("result provenance")
    if result.get("protocol", {}).get("bootstrap") != "sequence_mean_then_equal_sequence_cluster_bootstrap": failures.append("bootstrap contract")
    for effect in ("memory", "incremental"):
        for v in VARIANTS:
            for h in HORIZONS:
                item = result["effects"][effect][v][str(h)]
                if not all(k in item for k in ("identity_utility", "assignment_change_count", "assignment_change_correct_count", "assignment_change_incorrect_count", "assignment_change_neutral_count", "assignment_no_change_count", "untouched_regression", "sequence_cluster_bootstrap_95ci")): failures.append(f"aggregate {effect}/{v}/H{h}")
                if int(item["assignment_change_count"]) + int(item["assignment_no_change_count"]) != expected_evaluated[(effect, v, h)]: failures.append(f"aggregate closure {effect}/{v}/H{h}")
    report = {"status": "PASS" if not failures else "FAIL", "protocol": "N45_NORMALIZED_ATTRIBUTION_FULL_INTEGRITY_V1", "inputs": {"runtime": str(RUNTIME), "posthoc": str(POSTHOC), "result": str(RESULT), "old_n45_result": str(OLD), "checkpoint": str(CHECKPOINT)}, "outputs": {"integrity": str(OUT)}, "metrics": {"event_count": len(ids), "runtime_rows": runtime_rows, "posthoc_evaluated_frames_sum_over_events_variants_horizons": posthoc_frames, "old_n45_result_sha256": old_hash, "checkpoint_sha256": checkpoint_hash, "failures": failures[:100]}, "gate_checks": {"exact_24_events": len(runtime_files) == 24 and len(posthoc_files) == 24, "five_variants": not any(": variants" in x for x in failures), "100_runtime_frames": runtime_rows == 12000, "runtime_future_gt_false": not any("GT flag" in x or "provenance" in x for x in failures), "m0_exact_no_sidecar": not any("M0 sidecar" in x for x in failures), "horizons_and_metrics": not any("horizons" in x or "metric schema" in x for x in failures), "assignment_decomposition": not any("decomposition" in x or "closure" in x for x in failures), "equal_sequence_bootstrap": not any("bootstrap" in x for x in failures), "old_n45_hash_recorded": bool(old_hash), "checkpoint_hash_recorded": bool(checkpoint_hash)}, "failure_root_cause": "The corrected attribution is valid only when both branch assignment maps are axis-normalized and all runtime/posthoc/result schemas close; any failure remains FAIL.", "next_action": "Use normalized attribution for interpretation; retain old N45 as legacy and keep real-input gate closed.", "runtime_future_gt_used": False, "gt_loaded_posthoc": True}
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps({"status": report["status"], "output": str(OUT), "failures": len(failures)}))
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()
