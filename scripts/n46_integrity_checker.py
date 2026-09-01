#!/usr/bin/env python3
"""Full N46 integrity checker; writes only a new N46 report artifact."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
N42 = ROOT / "outputs/n42/replay/runtime/t0"
N45 = ROOT / "outputs/n45/replay/runtime"
RUNTIME = ROOT / "outputs/n46/diagnosis_repair1/events"
POSTHOC = ROOT / "outputs/n46/diagnosis_final/events"
OUT = ROOT / "outputs/n46/n46_integrity_report.json"
LEGACY_ROOTS = [ROOT / "outputs/n43", ROOT / "outputs/n44", ROOT / "outputs/n45", ROOT / "docs/N43_FINAL_REPORT.md", ROOT / "docs/N44_FINAL_REPORT.md", ROOT / "docs/N45_FINAL_REPORT.md"]
VARIANTS = {"M0", "M1", "M2", "M3", "M4"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def future_issues(value, path=""):
    out = []
    if isinstance(value, dict):
        for key, child in value.items():
            lower = str(key).lower(); child_path = f"{path}/{key}"
            if lower in {"runtime_future_gt_used", "future_gt_used", "gt_loaded_in_worker"} and child is not False:
                out.append(f"{child_path}={child!r}")
            if "future_gt_unused" in lower:
                out.append(f"reverse_name:{child_path}")
            out.extend(future_issues(child, child_path))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            out.extend(future_issues(child, f"{path}/{i}"))
    return out


def main() -> None:
    failures = []
    event_ids = sorted(str(x["event"]["event_id"]) for x in load(EVENTS)["events"])
    runtime_files = sorted(x for x in RUNTIME.glob("*.json") if not x.name.endswith(".posthoc.json"))
    posthoc_files = sorted(POSTHOC.glob("*.posthoc.json"))
    if len(runtime_files) != 24 or {x.stem for x in runtime_files} != set(event_ids):
        failures.append("N46 runtime event set is not exactly the frozen 24-event manifest")
    if len(posthoc_files) != 24 or {x.name.removesuffix(".posthoc.json") for x in posthoc_files} != set(event_ids):
        failures.append("N46 posthoc event set is not exactly the frozen 24-event manifest")
    runtime_rows = posthoc_rows = 0
    for event_id in event_ids:
        rp = load(RUNTIME / f"{event_id}.json"); pp = load(POSTHOC / f"{event_id}.posthoc.json")
        if rp.get("runtime_future_gt_used") is not False:
            failures.append(f"{event_id}: runtime future GT flag")
        if set(rp.get("variants", {})) != VARIANTS or set(pp.get("variants", {})) != VARIANTS:
            failures.append(f"{event_id}: variant set"); continue
        failures.extend(f"{event_id}: {x}" for x in future_issues(rp))
        for variant in sorted(VARIANTS):
            frames = rp["variants"][variant]; post = pp["variants"][variant]
            if len(frames) != 100 or len(post) != 100:
                failures.append(f"{event_id}/{variant}: frame count"); continue
            if [int(x["frame"]) for x in frames] != [int(x["frame"]) for x in post]:
                failures.append(f"{event_id}/{variant}: runtime/posthoc frames misaligned")
            for r, p in zip(frames, post):
                runtime_rows += 1; posthoc_rows += 1
                if r.get("runtime_future_gt_used") is not False or p.get("runtime_future_gt_used") is not False:
                    failures.append(f"{event_id}/{variant}/{r.get('frame')}: future GT flag")
                if p.get("gt_available") is True and p.get("gt_loaded_posthoc") is not True:
                    failures.append(f"{event_id}/{variant}/{r.get('frame')}: GT available without posthoc marker")
                if int(r.get("candidate_count", -1)) < 0 or int(r.get("public_id_count", -1)) < 0:
                    failures.append(f"{event_id}/{variant}/{r.get('frame')}: invalid axes")
                if int(r.get("assignment_changed_count", 0)) < 0 or int(r.get("selected_count", 0)) < 0:
                    failures.append(f"{event_id}/{variant}/{r.get('frame')}: invalid counts")
                if float(r.get("actual_score_delta_max", 0.0)) > 0.25 + 1e-6:
                    failures.append(f"{event_id}/{variant}/{r.get('frame')}: unbounded boost")
    summary = load(ROOT / "outputs/n46/diagnosis_final/structural_diagnosis.json")
    stage1 = load(ROOT / "outputs/n46/stage_01_status.json"); stage2 = load(ROOT / "outputs/n46/stage_02_status.json")
    if stage1.get("status") != "PASS" or stage1.get("metrics", {}).get("assignment_mismatch_count") != 0:
        failures.append("Stage 01 contract/mismatch gate")
    if stage2.get("status") != "PASS" or stage2.get("metrics", {}).get("runtime", {}).get("frames") != 12000:
        failures.append("Stage 02 complete runtime gate")
    if summary.get("runtime_future_gt_used") is not False or summary.get("gt_loaded_posthoc") is not True:
        failures.append("summary provenance flags")
    if set(summary.get("lambda_sensitivity_counterfactual", {})) != VARIANTS:
        failures.append("counterfactual variant set")
    n45_files = sorted(N45.glob("*.json"))
    preservation = {str(path): sha256(path) for path in n45_files}
    legacy_files = []
    for root in LEGACY_ROOTS:
        if root.is_file():
            legacy_files.append(root)
        elif root.is_dir():
            legacy_files.extend(path for path in root.rglob("*") if path.is_file())
    legacy_preservation = {str(path): sha256(path) for path in sorted(legacy_files)}
    result = {
        "status": "PASS" if not failures else "FAIL", "protocol": "N46_FULL_INTEGRITY_CHECK_V1",
        "inputs": {"n42": str(N42), "n45_runtime": str(N45), "n46_runtime": str(RUNTIME), "n46_posthoc": str(POSTHOC)},
        "metrics": {"event_count": len(event_ids), "runtime_rows": runtime_rows, "posthoc_rows": posthoc_rows, "n45_runtime_files_hashed": len(n45_files), "legacy_files_hashed": len(legacy_files), "failures": failures[:100]},
        "gate_checks": {"exact_24_runtime_events": len(runtime_files) == 24, "exact_24_posthoc_events": len(posthoc_files) == 24, "five_variants_100_frames": not any("variant" in x or "frame count" in x for x in failures), "runtime_future_gt_false": not any("future GT" in x for x in failures), "gt_posthoc_only": not any("posthoc marker" in x for x in failures), "stage01_assignment_mismatch_zero": stage1.get("metrics", {}).get("assignment_mismatch_count") == 0, "stage02_12000_frames": stage2.get("metrics", {}).get("runtime", {}).get("frames") == 12000, "n45_preservation_manifest_recorded": len(preservation) > 0},
        "failure_root_cause": "N46 is valid only when frozen branch alignment, future-GT provenance, and complete event/frame manifests pass; failures remain failures.",
        "next_action": "Use the N46 final diagnostic gate; do not authorize production or start blind training.",
        "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "n45_preservation_sha256": preservation, "legacy_n43_n44_n45_preservation_sha256": legacy_preservation,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(OUT), "failures": len(failures)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
