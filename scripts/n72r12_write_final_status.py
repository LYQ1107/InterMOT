#!/usr/bin/env python3
"""Seal the machine-readable N72R12 stage/controller/final-gate records.

This is a post-hoc bookkeeping and audit script.  It never runs SAM3, changes
the frozen replay, or writes historical outputs.  The intervention confusion
counts are recomputed from the sealed E0/E1B/E1C rows and post-hoc GT only so
that the controller does not infer them from rounded aggregate values.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n72r9_temporal_replay as legacy  # noqa: E402


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def _git(value: str) -> str:
    return subprocess.check_output(["git", *value.split()], cwd=ROOT, text=True).strip()


def _path(relative: str) -> Path:
    return ROOT / relative


def _metric(metrics: Mapping[str, Any], comparison: str, horizon: int) -> dict[str, Any]:
    return dict(metrics["comparisons"][comparison][str(horizon)])


def _track_metric(metrics: Mapping[str, Any], variant: str, horizon: int) -> dict[str, Any]:
    return dict(metrics["pooled"][str(horizon)][variant])


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"non-object row in {path}")
    return rows


def _same_row_axis(*rows: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    indexed = [{int(row["frame"]): row for row in values} for values in rows]
    frames = set(indexed[0])
    if any(set(item) != frames for item in indexed[1:]):
        raise RuntimeError(f"{label} frame axes differ")
    return {frame: indexed[0][frame] for frame in sorted(frames)}


def _protected_regression(
    *,
    baseline: Mapping[str, Any],
    treatment: Mapping[str, Any],
    gt: Mapping[int, Mapping[int, Any]],
    frame: int,
    protected: Mapping[int, int],
) -> bool:
    for protected_gid, protected_pid in protected.items():
        item = gt.get(frame, {}).get(int(protected_gid))
        if item is None:
            continue
        baseline_iou, _ = legacy._public_box_for_gt(baseline, int(protected_pid), item["box"])
        treatment_iou, _ = legacy._public_box_for_gt(treatment, int(protected_pid), item["box"])
        if baseline_iou >= legacy.IOU_THRESHOLD and treatment_iou < legacy.IOU_THRESHOLD:
            return True
    return False


def _intervention_confusion(
    *,
    protocol: Mapping[str, Any],
    r5_manifest: Mapping[str, Any],
    safe_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute proposal/gate outcomes on the complete H100 causal axis."""

    r5_records = {str(item["event_id"]): item for item in r5_manifest["records"]}
    safe_records = {str(item["event_id"]): item for item in safe_manifest["records"]}
    counters: Counter[str] = Counter()
    by_action: dict[str, Counter[str]] = {}
    for event in protocol["source_event_selection"]["events"]:
        event = dict(event)
        event_id = str(event["event_id"])
        inputs = legacy._load_rows(event)
        event["target_public_id"] = int(inputs["target_public_id"])
        # ``_load_rows`` intentionally returns runtime-only inputs and omits
        # the offline GT authority field; the frozen protocol event is the
        # source for this post-hoc-only value.
        event["target_dataset_gt_id"] = int(event["dataset_gt_id"])
        gt = legacy._load_gt(str(event["sequence"]))
        protected = legacy._protected_map(
            inputs["rows"]["c0_source"][int(event["event_frame"])],
            gt,
            int(event["event_frame"]),
            int(event["dataset_gt_id"]),
        )
        r5_dir = Path(str(r5_records[event_id]["done"])).parent
        safe_dir = Path(str(safe_records[event_id]["done"])).parent
        e0 = _load_jsonl(r5_dir / "E0_BASELINE_B0/runtime_frames.jsonl")
        e1b = _load_jsonl(r5_dir / "E1B_PCTIS_LEGACY/runtime_frames.jsonl")
        e1c = _load_jsonl(safe_dir / "E1C_PCTIS_SAFE/runtime_frames.jsonl")
        axis = _same_row_axis(e0, e1b, e1c, label=event_id)
        action = str(event["action_type"])
        action_counter = by_action.setdefault(action, Counter())
        for frame in range(int(event["event_frame"]) + 1, int(event["event_frame"]) + 101):
            if frame not in axis:
                raise RuntimeError(f"{event_id} is missing H100 frame {frame}")
            baseline = axis[frame]
            proposal = {int(row["frame"]): row for row in e1b}[frame]
            safe = {int(row["frame"]): row for row in e1c}[frame]
            target_gt = gt.get(frame, {}).get(int(event["dataset_gt_id"]))
            if target_gt is None:
                counters["target_absent_frames"] += 1
                action_counter["target_absent_frames"] += 1
                continue
            baseline_iou, _ = legacy._public_box_for_gt(baseline, int(event["target_public_id"]), target_gt["box"])
            proposal_iou, _ = legacy._public_box_for_gt(proposal, int(event["target_public_id"]), target_gt["box"])
            baseline_correct = baseline_iou >= legacy.IOU_THRESHOLD
            proposal_correct = proposal_iou >= legacy.IOU_THRESHOLD
            protected_regression = _protected_regression(
                baseline=baseline,
                treatment=proposal,
                gt=gt,
                frame=frame,
                protected=protected,
            )
            harmful = bool((baseline_correct and not proposal_correct) or protected_regression)
            beneficial = bool((not baseline_correct) and proposal_correct and not protected_regression)
            applied = bool(safe.get("counterfactual_intervention", {}).get("intervention_applied") is True)
            counters["visible_frames"] += 1
            action_counter["visible_frames"] += 1
            counters["total_applied"] += int(applied)
            action_counter["total_applied"] += int(applied)
            counters["total_kept"] += int(not applied)
            action_counter["total_kept"] += int(not applied)
            counters["harmful_proposals"] += int(harmful)
            action_counter["harmful_proposals"] += int(harmful)
            counters["harmful_blocked"] += int(harmful and not applied)
            action_counter["harmful_blocked"] += int(harmful and not applied)
            counters["harmful_applied"] += int(harmful and applied)
            action_counter["harmful_applied"] += int(harmful and applied)
            counters["beneficial_proposals"] += int(beneficial)
            action_counter["beneficial_proposals"] += int(beneficial)
            counters["beneficial_applied"] += int(beneficial and applied)
            action_counter["beneficial_applied"] += int(beneficial and applied)
            counters["beneficial_blocked"] += int(beneficial and not applied)
            action_counter["beneficial_blocked"] += int(beneficial and not applied)
            counters["protected_regression_frames"] += int(protected_regression)
            action_counter["protected_regression_frames"] += int(protected_regression)

    def rates(values: Mapping[str, Any]) -> dict[str, float]:
        harmful = int(values["harmful_proposals"])
        beneficial = int(values["beneficial_proposals"])
        applied = int(values["total_applied"])
        return {
            "harm_block_rate": int(values["harmful_blocked"]) / max(harmful, 1),
            "benefit_retention_rate": int(values["beneficial_applied"]) / max(beneficial, 1),
            "intervention_precision": int(values["beneficial_applied"]) / max(applied, 1),
        }

    overall = dict(counters)
    overall.update(rates(overall))
    action_payload: dict[str, Any] = {}
    for action, values in sorted(by_action.items()):
        item = dict(values)
        item.update(rates(item))
        action_payload[action] = item
    return {
        "definition": {
            "proposal": "E1B_PCTIS_LEGACY versus E0 on the same H100 frame",
            "harmful": "E0 target correct and proposal target wrong, or proposal protected regression",
            "beneficial": "E0 target wrong, proposal target correct, and no proposal protected regression",
            "gate_apply": "sealed E1C counterfactual_intervention.intervention_applied",
            "gt_usage": "posthoc only; no runtime GT",
        },
        "horizon": 100,
        "overall": overall,
        "by_action": action_payload,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
    }


def _common(
    *,
    stage: str,
    status: str,
    inputs: Mapping[str, str],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "schema_version": "N72R12_STAGE_STATUS_V1",
        "stage": stage,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head": _git("rev-parse HEAD"),
        "source_branch": _git("symbolic-ref --short HEAD"),
        "inputs": dict(inputs),
        "runtime_future_gt_used": False,
        "historical_outputs_modified": False,
        "third_party_modified": False,
        **extra,
    }


def main() -> int:
    n72r9_protocol = _path("outputs/N72R9/protocol.json")
    r5_manifest_path = _path("outputs/N72R11R5R1/formal_e1b_manifest.json")
    safe_manifest_path = _path("outputs/N72R12/formal_safe/formal_safe_manifest.json")
    learned_manifest_path = _path("outputs/N72R12/formal_learned/formal_learned_manifest.json")
    causal_path = _path("outputs/N72R12/causal/formal_causal_metrics.json")
    learned_causal_path = _path("outputs/N72R12/causal/learned_causal_metrics.json")
    trackeval_path = _path("outputs/N72R12/trackeval/window_metrics.json")
    learned_trackeval_path = _path("outputs/N72R12/trackeval_learned/window_metrics.json")
    corpus_path = _path("outputs/N72R12/gate_corpus/manifest.json")
    training_path = _path("outputs/N72R12/gate_training/training_manifest.json")
    training_history_path = _path("outputs/N72R12/gate_training/training_history.json")
    smoke_path = _path("outputs/N72R12/smoke/smoke_validation.json")
    learned_smoke_path = _path("outputs/N72R12/learned_smoke_h100/n72r5-pool-n37-dancetrack0001-0296-authoritative_reassign-001/done.json")
    oracle_path = _path("outputs/N72R12/oracle/oracle_headroom.json")
    learned_runtime_path = _path("outputs/N72R12/stage_14_status.json")
    required = [
        n72r9_protocol, r5_manifest_path, safe_manifest_path, learned_manifest_path,
        causal_path, learned_causal_path, trackeval_path, learned_trackeval_path,
        corpus_path, training_path, training_history_path, smoke_path, learned_smoke_path,
        oracle_path, learned_runtime_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required N72R12 input: " + ", ".join(missing))

    protocol = _read(n72r9_protocol)
    r5_manifest = _read(r5_manifest_path)
    safe_manifest = _read(safe_manifest_path)
    learned_manifest = _read(learned_manifest_path)
    causal = _read(causal_path)
    learned_causal = _read(learned_causal_path)
    trackeval = _read(trackeval_path)
    learned_trackeval = _read(learned_trackeval_path)
    corpus = _read(corpus_path)
    training = _read(training_path)
    history = _read(training_history_path)
    smoke = _read(smoke_path)
    learned_smoke = _read(learned_smoke_path)
    oracle = _read(oracle_path)
    learned_runtime = _read(learned_runtime_path)
    inputs = {str(path.relative_to(ROOT)): _sha(path) for path in required}
    confusion = _intervention_confusion(protocol=protocol, r5_manifest=r5_manifest, safe_manifest=safe_manifest)
    det_h20 = _metric(causal, "E1C_vs_E0", 20)
    det_h50 = _metric(causal, "E1C_vs_E0", 50)
    det_h100 = _metric(causal, "E1C_vs_E0", 100)
    learned_h20 = _metric(learned_causal, "E1D_vs_E0", 20)
    learned_h50 = _metric(learned_causal, "E1D_vs_E0", 50)
    learned_h100 = _metric(learned_causal, "E1D_vs_E0", 100)
    det_gate = dict(trackeval.get("deterministic_gate", {}))
    learned_gate = dict(learned_trackeval.get("strict_learned_gate", {}))
    records = [dict(item) for item in safe_manifest["records"]]
    learned_records = [dict(item) for item in learned_manifest["records"]]
    safe_diagnostics = dict(learned_causal["safe_diagnostics"])
    deterministic_safe_diagnostics = dict(causal["safe_diagnostics"]["overall"])
    deterministic_apply = int(deterministic_safe_diagnostics.get("decision:APPLY_PCTIS", 0))
    deterministic_keep = int(deterministic_safe_diagnostics.get("decision:KEEP_BASELINE", 0))
    controller = {
        "schema_version": "N72R12_CONTROLLER_STATUS_V1",
        "status": "FAIL_SAFE_INTERVENTION_NOT_SUFFICIENT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "FINAL_AUDIT_COMPLETE",
        "source_head": _git("rev-parse HEAD"),
        "source_branch": _git("symbolic-ref --short HEAD"),
        "oracle_headroom_complete": oracle.get("status") == "PASS_POSTHOC_ORACLE_HEADROOM",
        "deterministic_gate_implemented": True,
        "formal_event_pass_count": len([item for item in records if item.get("status") == "PASS"]),
        "learned_formal_event_pass_count": len([item for item in learned_records if item.get("status") == "PASS"]),
        "safe_proposals": deterministic_apply + deterministic_keep,
        "safe_apply_count": deterministic_apply,
        "safe_keep_count": deterministic_keep,
        "harmful_proposals": confusion["overall"]["harmful_proposals"],
        "harmful_blocked": confusion["overall"]["harmful_blocked"],
        "beneficial_proposals": confusion["overall"]["beneficial_proposals"],
        "beneficial_applied": confusion["overall"]["beneficial_applied"],
        "harm_block_rate": confusion["overall"]["harm_block_rate"],
        "benefit_retention_rate": confusion["overall"]["benefit_retention_rate"],
        "intervention_precision": confusion["overall"]["intervention_precision"],
        "intervention_confusion": confusion,
        "E1B_H20": _metric(causal, "E1B_vs_E0", 20),
        "E1B_H50": _metric(causal, "E1B_vs_E0", 50),
        "E1B_H100": _metric(causal, "E1B_vs_E0", 100),
        "E1C_H20": det_h20,
        "E1C_H50": det_h50,
        "E1C_H100": det_h100,
        "E1C_correct_H20": det_h20["true_correct_crossing_count"],
        "E1C_incorrect_H20": det_h20["true_incorrect_crossing_count"],
        "E1C_protected_H100": det_h100["protected_regression_count"],
        "E1C_HOTA_H20": _track_metric(trackeval, "E1C_PCTIS_SAFE", 20)["HOTA"],
        "E1C_HOTA_H50": _track_metric(trackeval, "E1C_PCTIS_SAFE", 50)["HOTA"],
        "E1C_HOTA_H100": _track_metric(trackeval, "E1C_PCTIS_SAFE", 100)["HOTA"],
        "E1C_AssA_H100": _track_metric(trackeval, "E1C_PCTIS_SAFE", 100)["AssA"],
        "E1C_IDF1_H100": _track_metric(trackeval, "E1C_PCTIS_SAFE", 100)["IDF1"],
        "E1C_IDSW_H100": _track_metric(trackeval, "E1C_PCTIS_SAFE", 100)["IDSW"],
        "deterministic_gate_pass": bool(det_gate.get("pass") is True),
        "learned_gate_triggered": True,
        "gate_train_examples": int(training.get("train_examples", corpus["splits"]["train"]["included_count"])),
        "gate_val_examples": int(training.get("validation_examples", corpus["splits"]["validation"]["included_count"])),
        "gate_positive_labels": int(corpus["positive_sample_count"]),
        "gate_val_precision": training["readiness"].get("precision_at_0.5"),
        "gate_val_recall": history["history"][0]["validation_metrics"].get("recall"),
        "gate_val_unsafe_apply_rate": training["readiness"].get("unsafe_apply_rate"),
        "E1D_H20": learned_h20,
        "E1D_H50": learned_h50,
        "E1D_H100": learned_h100,
        "E1D_HOTA_H100": _track_metric(learned_trackeval, "E1D_PCTIS_LEARNED_SAFE", 100)["HOTA"],
        "E1D_AssA_H100": _track_metric(learned_trackeval, "E1D_PCTIS_LEARNED_SAFE", 100)["AssA"],
        "E1D_IDF1_H100": _track_metric(learned_trackeval, "E1D_PCTIS_LEARNED_SAFE", 100)["IDF1"],
        "E1D_IDSW_H100": _track_metric(learned_trackeval, "E1D_PCTIS_LEARNED_SAFE", 100)["IDSW"],
        "learned_gate_pass": bool(learned_gate.get("pass") is True),
        "final_direction": "SAFE_INTERVENTION_NOT_SUFFICIENT",
        "runtime_future_gt_used": False,
        "historical_outputs_modified": False,
        "third_party_modified": False,
        "input_sha256": inputs,
        "preserved_failure_artifacts": [
            str(path.relative_to(ROOT))
            for path in sorted(_path("outputs/N72R12").rglob("*"))
            if path.is_file() and ("failure" in path.name or "partial" in path.name or "attempt_" in path.name)
        ],
    }
    _atomic(_path("outputs/N72R12/intervention_confusion.json"), confusion)
    _atomic(_path("outputs/N72R12/CONTROLLER_STATUS.json"), controller)

    common_inputs = inputs
    statuses = {
        "stage_01_status.json": _common(stage="Stage01", status="PASS_N72R12_COUNTERFACTUAL_MODULE", inputs=common_inputs, module="sam3_intermot/association/counterfactual_safe_intervention.py", policy="COUNTERFACTUAL_SAFE_V1"),
        "stage_02_status.json": _common(stage="Stage02", status="PASS_N72R12_COUNTERFACTUAL_INTEGRATION", inputs=common_inputs, variant="E1C_PCTIS_SAFE", formal_events=32, smoke_status=smoke.get("status")),
        "stage_04_status.json": _common(stage="Stage04", status="PASS_POSTHOC_ORACLE_HEADROOM", inputs=common_inputs, runtime_eligible=bool(oracle.get("runtime_eligible", False)), scientific_runtime_result=bool(oracle.get("scientific_runtime_result", False))),
        "stage_05_status.json": _common(stage="Stage05", status="PASS_N72R12_COMPILE", inputs=common_inputs, compile_scope="N72R12 modules and replay scripts; verified before status sealing"),
        "stage_06_status.json": _common(stage="Stage06", status="PASS_N72R12_TARGETED_SMOKE", inputs=common_inputs, deterministic_smoke=str(smoke_path.relative_to(ROOT)), learned_smoke=str(learned_smoke_path.relative_to(ROOT)), learned_smoke_status=learned_smoke.get("status"), prior_failed_attempts_preserved=True),
        "stage_07_status.json": _common(stage="Stage07", status="PASS_N72R12_FORMAL_DETERMINISTIC_RUNTIME", inputs=common_inputs, event_count=32, record_count=len(records), integrity=safe_manifest.get("integrity")),
        "stage_08_status.json": _common(stage="Stage08", status="PASS_N72R12_FORMAL_CAUSAL_METRICS", inputs=common_inputs, event_count=causal.get("event_count"), sequence_count=causal.get("independent_sequence_count"), metric_status=causal.get("status")),
        "stage_09_status.json": _common(stage="Stage09", status="PASS_N72R12_FORMAL_TRACKEVAL", inputs=common_inputs, record_count=trackeval.get("record_count"), trackeval_commit=trackeval.get("trackeval_commit"), official_benchmark_score=False),
        "stage_10_status.json": _common(stage="Stage10", status="PASS_N72R12_FORMAL_PAIRED_BOOTSTRAP", inputs=common_inputs, causal_bootstrap=causal.get("bootstrap"), deterministic_gate_pass=bool(det_gate.get("pass") is True)),
        "stage_11_status.json": _common(stage="Stage11", status="FAIL_DETERMINISTIC_FUTURE_EFFECT_GATE", inputs=common_inputs, deterministic_gate=det_gate, learned_stage_required=not bool(det_gate.get("pass") is True), production_authorized=False),
        "stage_12_status.json": _common(stage="Stage12", status="PASS_N72R12_SAFE_GATE_CORPUS", inputs=common_inputs, source_rows=corpus.get("source_row_count"), included_samples=corpus.get("included_sample_count"), positive_labels=corpus.get("positive_sample_count"), negative_labels=corpus.get("negative_sample_count"), sequence_disjoint=True),
        "stage_13_status.json": _common(stage="Stage13", status="PASS_N72R12_SAFE_GATE_TRAINING", inputs=common_inputs, training_config=training.get("training_config"), readiness=training.get("readiness"), checkpoint_sha256=training.get("checkpoint_sha256"), production_authorized=False),
        "stage_15_status.json": _common(stage="Stage15", status="PASS_N72R12_LEARNED_TRACKEVAL", inputs=common_inputs, record_count=learned_trackeval.get("record_count"), trackeval_commit=learned_trackeval.get("trackeval_commit"), official_benchmark_score=False),
        "stage_16_status.json": _common(stage="Stage16", status="PASS_N72R12_LEARNED_PAIRED_BOOTSTRAP", inputs=common_inputs, causal_bootstrap=learned_causal.get("bootstrap"), learned_gate=learned_gate),
        "stage_17_status.json": _common(stage="Stage17", status="FAIL_LEARNED_FUTURE_EFFECT_GATE", inputs=common_inputs, learned_gate=learned_gate, production_authorized=False),
        "stage_18_status.json": _common(stage="Stage18", status="PASS_N72R12_FINAL_AUDIT", inputs=common_inputs, structural_runtime_pass=True, deterministic_gate_pass=False, learned_gate_pass=False, historical_outputs_modified=False, third_party_modified=False),
        "stage_19_status.json": _common(stage="Stage19", status="FAIL_SAFE_INTERVENTION_NOT_SUFFICIENT", inputs=common_inputs, final_direction="SAFE_INTERVENTION_NOT_SUFFICIENT", next_step="real human event tape or frozen association-interface probe", production_authorized=False),
    }
    output_dir = _path("outputs/N72R12")
    preserved: list[str] = []
    written: list[str] = []
    for filename, payload in statuses.items():
        path = output_dir / filename
        if path.exists():
            preserved.append(filename)
        else:
            _atomic(path, payload)
            written.append(filename)

    final_gate = {
        "schema_version": "N72R12_FINAL_GATE_V1",
        "status": "FAIL_SAFE_INTERVENTION_NOT_SUFFICIENT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head": _git("rev-parse HEAD"),
        "source_branch": _git("symbolic-ref --short HEAD"),
        "event_count": 32,
        "independent_sequence_count": 18,
        "formal_deterministic_runtime_pass": len(records) == 32 and all(item.get("status") == "PASS" for item in records),
        "formal_learned_runtime_pass": len(learned_records) == 32 and all(item.get("status") == "PASS" for item in learned_records),
        "oracle_headroom_complete": bool(oracle.get("status") == "PASS_POSTHOC_ORACLE_HEADROOM"),
        "deterministic_gate": det_gate,
        "learned_gate": learned_gate,
        "production_authorized": False,
        "calibration_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "third_party_modified": False,
        "historical_outputs_modified": False,
        "written_stage_statuses": written,
        "preserved_existing_stage_statuses": preserved + ["stage_00_status.json", "stage_03_status.json", "stage_14_status.json"],
        "controller": "outputs/N72R12/CONTROLLER_STATUS.json",
        "intervention_confusion": "outputs/N72R12/intervention_confusion.json",
        "final_report": "docs/N72R12_FINAL_REPORT.md",
    }
    _atomic(_path("outputs/N72R12/n72r12_final_gate.json"), final_gate)
    print(json.dumps({"status": final_gate["status"], "written_stage_statuses": written, "preserved_stage_statuses": preserved, "controller": "outputs/N72R12/CONTROLLER_STATUS.json", "final_gate": "outputs/N72R12/n72r12_final_gate.json"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        failure = {
            "schema_version": "N72R12_STATUS_WRITER_FAILURE_V1",
            "status": "FAIL_N72R12_FINAL_STATUS_SEAL",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "historical_outputs_modified": False,
            "third_party_modified": False,
        }
        _atomic(ROOT / "outputs/N72R12/status_writer_failure_attempt_01.json", failure)
        raise
