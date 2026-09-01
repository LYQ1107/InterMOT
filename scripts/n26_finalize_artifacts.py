#!/usr/bin/env python3
"""Finalize N26 manifests, failure artifacts, commands and resume state."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(".")
OUT = ROOT / "outputs/n26"
DENSE = OUT / "dense_dataset"
sys.path.insert(0, str(ROOT / "scripts"))
from n26_evaluate_ccsam import b10_scores  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def combine_curves() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("training_curves_round0.csv", "training_curves_round1.csv"):
        rows.extend(csv.DictReader((OUT / name).open(newline="", encoding="utf-8")))
    write_csv(OUT / "training_curves.csv", rows)
    round0 = torch.load(OUT / "checkpoints/n26_round0_final.pt", map_location="cpu", weights_only=False)
    round1 = torch.load(OUT / "checkpoints/n26_round1_final.pt", map_location="cpu", weights_only=False)
    selection = [row for row in rows if row["stage"] == "round0_selection"]
    best = min(selection, key=lambda row: (float(row["val_total_loss"]), -float(row["val_h5_top1"]), int(row["epoch"])))
    extra_seconds = []
    for path in sorted((OUT / "logs").glob("extra_clip_s*_repair1.log")):
        match = re.search(r"N26_EXTRA_CLIP_DONE.*runtime_s=([0-9.]+)", path.read_text(encoding="utf-8"))
        if match:
            extra_seconds.append(float(match.group(1)))
    summary = {
        "model": "CC-SAM", "parameters": round1["parameter_count"], "precision": "bfloat16 AMP",
        "ddp_world_size": 4, "round0_selection_epochs": 12, "selected_epoch_count": round0["selected_epoch_count"],
        "selection_validation_sequences": round0["selection_validation_sequences"],
        "selection_best": {"epoch": int(best["epoch"]), "val_total_loss": float(best["val_total_loss"]), "val_h5_top1": float(best["val_h5_top1"])},
        "round0_full_epochs": round0["epoch"], "round1_aggregate_epochs": round1["epoch"],
        "round0_gpu_hours": round0["gpu_hours"], "round1_gpu_hours": round1["gpu_hours"],
        "association_training_gpu_hours_total": round0["gpu_hours"] + round1["gpu_hours"],
        "extra_clip_feature_gpu_hours": sum(extra_seconds) / 3600.0,
        "last_round0_parent_weight": float([row for row in rows if row["stage"] == "round0_full_fit"][-1]["train_parent_weight"]),
        "last_round1_parent_weight": float([row for row in rows if row["stage"] == "round1_aggregate_refit"][-1]["train_parent_weight"]),
        "last_round1_loss": float([row for row in rows if row["stage"] == "round1_aggregate_refit"][-1]["train_total_loss"]),
        "last_round1_parent_weighted_accuracy": float([row for row in rows if row["stage"] == "round1_aggregate_refit"][-1]["train_parent_weighted_accuracy"]),
        "sam3_parameters_in_optimizer": False, "reid_parameters_in_optimizer": False,
        "round0_checkpoint_sha256": sha256(OUT / "checkpoints/n26_round0_final.pt"),
        "round1_checkpoint_sha256": sha256(OUT / "checkpoints/n26_round1_final.pt"),
        "val25_read": False,
    }
    write_json(OUT / "training_summary.json", summary)
    return rows, summary


def failure_artifacts() -> dict[str, Any]:
    with np.load(DENSE / "round1_cal10.npz", allow_pickle=False) as z:
        arrays = {name: z[name].copy() for name in z.files}
    with np.load(DENSE / "round0_cal10.npz", allow_pickle=False) as z:
        round0 = {name: z[name].copy() for name in z.files}
    with np.load(OUT / "evaluation_predictions.npz", allow_pickle=False) as z:
        prediction = {name: z[name].copy() for name in z.files}
    rows = [json.loads(line) for line in (DENSE / "round1_cal10_parents.jsonl").open(encoding="utf-8") if line.strip()]
    canonical = prediction["canonical_state"].astype(int)
    target = prediction["target"].astype(int)
    main_logits = prediction["CCSAM_POSITIVE_AND_NEGATIVE_logits"].astype(np.float32)
    candidate_mask = arrays["candidate_mask"][canonical]
    b10 = b10_scores(round0, canonical)
    cases: list[dict[str, Any]] = []
    for index, truth in enumerate(target):
        if truth >= 5 or not candidate_mask[index, truth]:
            continue
        valid = np.flatnonzero(candidate_mask[index])
        main_order = sorted(valid, key=lambda candidate: (-float(main_logits[index, candidate]), int(candidate)))
        b10_order = sorted(valid, key=lambda candidate: (-float(b10[index, candidate]), int(candidate)))
        main_rank, b10_rank = main_order.index(truth) + 1, b10_order.index(truth) + 1
        if main_rank == 1 or b10_rank != 1:
            continue
        hardest = main_order[0]
        b10_negatives = [candidate for candidate in valid if candidate != truth]
        parent = int(arrays["parent"][canonical[index]])
        kinds = arrays["memory_kind"][parent][arrays["memory_mask"][parent]]
        cases.append({
            "event_key": rows[index]["event_key"], "sequence": rows[index]["sequence"], "frame": rows[index]["frame"],
            "gid": rows[index]["gid"], "state_label": rows[index]["state_label"], "target_candidate": int(truth),
            "ccsam_kplus1_prediction": int(np.argmax(main_logits[index])), "ccsam_candidate_rank": main_rank,
            "b10_candidate_rank": b10_rank, "ccsam_target_logit": float(main_logits[index, truth]),
            "ccsam_hardest_wrong_logit": float(main_logits[index, hardest]),
            "ccsam_margin": float(main_logits[index, truth] - main_logits[index, hardest]),
            "b10_margin": float(b10[index, truth] - max(b10[index, candidate] for candidate in b10_negatives)) if b10_negatives else math.nan,
            "positive_memory_tokens": int(np.sum(kinds == 1)), "explicit_negative_memory_tokens": int(np.sum(kinds == 2)),
            "hard_negative_memory_tokens": int(np.sum(kinds == 3)),
            "failure_type": "B10_CORRECT_CCSAM_CANDIDATE_RANK_WRONG",
        })
    cases.sort(key=lambda row: (row["ccsam_margin"], row["sequence"], row["frame"]))
    write_csv(OUT / "failure_cases.csv", cases[:100])

    ablations = {row["method"]: row for row in csv.DictReader((OUT / "ablation_results.csv").open(newline="", encoding="utf-8"))}
    gate = json_file(OUT / "n26b_gate.json")
    analysis = {
        "status": "SCIENTIFIC_FAILURE_NO_THIRD_ROUTE",
        "aligned_baseline_reproduction": {
            "dense_B10_top1": float(ablations["B10_EXPLICIT_NEGATIVE_DENSE"]["top1"]),
            "N26A_same_policy_B10_top1": json_file(OUT / "n26a_gate.json")["same_policy_b10_rank"]["cal10"]["top1"],
            "difference": float(ablations["B10_EXPLICIT_NEGATIVE_DENSE"]["top1"]) - json_file(OUT / "n26a_gate.json")["same_policy_b10_rank"]["cal10"]["top1"],
            "interpretation": "exact reproduction rules out candidate/order misalignment in the evaluator",
        },
        "generalization_collapse": {
            "round1_train_parent_weighted_accuracy": json_file(OUT / "training_summary.json")["last_round1_parent_weighted_accuracy"],
            "round0_fixed_final_history_cal_top1": float(ablations["CCSAM_ROUND0_CHECKPOINT_FIXED_FINAL_HISTORY_DIAGNOSTIC"]["top1"]),
            "round1_main_cal_top1": float(ablations["CCSAM_POSITIVE_AND_NEGATIVE"]["top1"]),
            "dense_B10_cal_top1": float(ablations["B10_EXPLICIT_NEGATIVE_DENSE"]["top1"]),
            "round1_main_margin": float(ablations["CCSAM_POSITIVE_AND_NEGATIVE"]["hardest_negative_margin"]),
            "interpretation": "candidate scorer, not just NONE/risk threshold, fails sequence transfer",
        },
        "memory_effect": gate["memory_recorrection"],
        "correction_counterfactual": gate["correction_response"],
        "failure_case_rows": min(100, len(cases)), "failure_case_population": len(cases),
        "no_third_route_started": True, "full_loop_authorized": False, "val25_read": False,
    }
    write_json(OUT / "failure_analysis.json", analysis)
    return analysis


def combined_audits() -> None:
    n26a_dataset = json_file(OUT / "on_policy_dataset_manifest.json")
    n26a_memory = json_file(OUT / "correction_memory_audit.json")
    if "N26A" in n26a_dataset:
        n26a_dataset = n26a_dataset["N26A"]
    if "N26A" in n26a_memory:
        n26a_memory = n26a_memory["N26A"]
    round0_manifest = json_file(DENSE / "round0_manifest.json")
    round1_train = json_file(DENSE / "round1_train30_summary.json")
    final_cal = json_file(DENSE / "round1_cal10_summary.json")
    dataset_manifest = {
        "phase": "N26", "candidate_stream": "N25-R repaired frozen GFN top-5; no union",
        "N26A": n26a_dataset, "N26B_round0": round0_manifest,
        "N26B_round1_train": round1_train, "N26B_final_cal_rollout": final_cal,
        "human_event_unit": "parent_event_id; temporal prefixes and policy rounds are clustered, not independent humans",
        "current_feedback_used_by_current_prediction": False, "val25_read": False,
    }
    write_json(OUT / "on_policy_dataset_manifest.json", dataset_manifest)
    memory_audit = {
        "phase": "N26", "N26A": n26a_memory,
        "N26B": {
            "round0_train": json_file(DENSE / "round0_train30_summary.json"),
            "round0_cal": json_file(DENSE / "round0_cal10_summary.json"),
            "round1_train": round1_train, "final_cal": final_cal,
            "memory_kinds": {
                "HUMAN_EXPLICIT_POSITIVE": "simulated corrected/verified target after an error",
                "HUMAN_EXPLICIT_NEGATIVE": "only policy-selected then simulated-human-rejected candidate",
                "MODEL_INDUCED_HARD_NEGATIVE": "unselected ordinary contrast source; never claimed as human",
            },
            "state_scope": "sequence+public_identity_id", "current_feedback_changes_current_state": False,
            "sequence_reset": True, "unselected_candidate_as_human_negative": False,
        },
        "val25_read": False,
    }
    write_json(OUT / "correction_memory_audit.json", memory_audit)


def status_and_reproduction(training: dict[str, Any]) -> None:
    (OUT / "full_loop").mkdir(parents=True, exist_ok=True)
    (OUT / "trackeval").mkdir(parents=True, exist_ok=True)
    reason = "N26-B scientific gate failed ranking, sequence-OOF safety, commit diversity, and significant correction-response criteria"
    write_json(OUT / "full_loop/NOT_RUN.json", {"status": "NOT_RUN", "reason": reason, "stress_sequences_run": False, "complete_cal10_run": False, "val25_read": False})
    write_json(OUT / "trackeval/NOT_RUN.json", {"status": "NOT_RUN", "reason": "FULL_LOOP was not authorized", "HOTA": "NOT_RUN", "DetA": "NOT_RUN", "AssA": "NOT_RUN", "IDF1": "NOT_RUN", "IDSW": "NOT_RUN", "val25_read": False})

    commands = f"""# N26 commands

All commands ran from `{ROOT}`. Long generation/training/evaluation jobs used one blocking shell command; no process was killed. `val25` was never read.

## N26-A

```bash
envs/sam3_intermot/bin/python scripts/n26a_onpolicy_gate.py 2>&1 | tee outputs/n26/logs/n26a_onpolicy_gate.log
```

## Additional frozen CLIP-ReID features

Four shards used physical GPUs 0--3. The first launch failed before model/data work because of a `scripts` import; `_repair1.log` files are the successful minimal import repair. Each shard command was:

```bash
CUDA_VISIBLE_DEVICES=<gpu> envs/sam3_intermot/bin/python scripts/n26_extract_extra_clip.py --gpu 0 --shard <0..3> --num-shards 4 --batch-size 96
```

## Dense data and regression smoke

```bash
envs/sam3_intermot/bin/python scripts/n26_build_dense_dataset.py --split all
CUDA_VISIBLE_DEVICES=6 envs/sam3_intermot/bin/python <single-batch backward smoke>
CUDA_VISIBLE_DEVICES=6,7,8,9 envs/sam3_intermot/bin/torchrun --standalone --nproc_per_node=4 scripts/n26_ddp_smoke.py
```

The first single-batch smoke found a `numpy.bool_` collation incompatibility and was minimally repaired. The first DDP smoke completed all-reduce but exposed a teardown race; `n26_ddp_smoke_repair1.log` is the synchronized regression pass.

## Round 0 full four-GPU training

```bash
CUDA_VISIBLE_DEVICES=6,7,8,9 OMP_NUM_THREADS=4 envs/sam3_intermot/bin/torchrun --standalone --nproc_per_node=4 scripts/n26_train_ccsam.py --stage round0 --selection-epochs 12 --batch-size 32 --workers 2
```

## Round 1 rollout and aggregate refit

```bash
CUDA_VISIBLE_DEVICES=6 envs/sam3_intermot/bin/python scripts/n26_round1_rollout.py --split train30 --checkpoint outputs/n26/checkpoints/n26_round0_final.pt --device cuda:0
CUDA_VISIBLE_DEVICES=6,7,8,9 OMP_NUM_THREADS=4 envs/sam3_intermot/bin/torchrun --standalone --nproc_per_node=4 scripts/n26_train_ccsam.py --stage round1 --data outputs/n26/dense_dataset/round0_train30.npz --data2 outputs/n26/dense_dataset/round1_train30.npz --init outputs/n26/checkpoints/n26_round0_final.pt --epochs 5 --batch-size 32 --workers 2
```

The first Round-1 rollout completed arrays but failed while formatting a relative checkpoint path in its summary. The relative-path regression was fixed; `n26_round1_rollout_train30_repair1.log` is canonical.

## Final cal10 rollout and evaluation

```bash
CUDA_VISIBLE_DEVICES=6 envs/sam3_intermot/bin/python scripts/n26_round1_rollout.py --split cal10 --checkpoint outputs/n26/checkpoints/n26_round1_final.pt --device cuda:0
CUDA_VISIBLE_DEVICES=6 envs/sam3_intermot/bin/python scripts/n26_evaluate_ccsam.py --checkpoint outputs/n26/checkpoints/n26_round1_final.pt --device cuda:0
envs/sam3_intermot/bin/python scripts/n26_finalize_artifacts.py
```
"""
    (OUT / "commands.md").write_text(commands, encoding="utf-8")

    resume = f"""# N26 resume state

Status: `SCIENTIFIC_GATE_FAIL`; the protocol forbids a third route. FULL_LOOP, TrackEval, final calibration, and val25 are `NOT_RUN`.

The final resumable model is `outputs/n26/checkpoints/n26_round1_final.pt` (SHA-256 `{training['round1_checkpoint_sha256']}`). It contains model/optimizer states, RNG states, model/data configuration and source data hashes. Round0 is `outputs/n26/checkpoints/n26_round0_final.pt` (SHA-256 `{training['round0_checkpoint_sha256']}`).

To reproduce evaluation without changing the frozen protocol:

```bash
CUDA_VISIBLE_DEVICES=6 envs/sam3_intermot/bin/python scripts/n26_round1_rollout.py --split cal10 --checkpoint outputs/n26/checkpoints/n26_round1_final.pt --device cuda:0
CUDA_VISIBLE_DEVICES=6 envs/sam3_intermot/bin/python scripts/n26_evaluate_ccsam.py --checkpoint outputs/n26/checkpoints/n26_round1_final.pt --device cuda:0
```

Do not use cal10 to replace the main checkpoint with the Round0 diagnostic. Do not read val25 unless a future, separately frozen method passes all required gates from scratch.
"""
    (OUT / "RESUME.md").write_text(resume, encoding="utf-8")


def build_manifest() -> None:
    artifacts = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or path.suffix == ".tmp":
            continue
        artifacts.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    for pattern in ("docs/N26*.md", "scripts/n26*.py"):
        for path in sorted(ROOT.glob(pattern)):
            artifacts.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    unique = {item["path"]: item for item in artifacts}
    manifest = {
        "phase": "N26", "status": json_file(OUT / "n26b_gate.json")["status"],
        "artifact_count_excluding_manifest": len(unique), "artifacts": [unique[key] for key in sorted(unique)],
        "candidate_union": False, "full_loop_authorized": False, "full_loop_run": False,
        "trackeval_run": False, "val25_read": False, "val25_frozen_manifest_created": False,
    }
    write_json(OUT / "manifest.json", manifest)


def main() -> None:
    _, training = combine_curves()
    analysis = failure_artifacts()
    combined_audits()
    status_and_reproduction(training)
    build_manifest()
    print(json.dumps({"status": analysis["status"], "training_gpu_hours": training["association_training_gpu_hours_total"], "manifest_artifacts": json_file(OUT / "manifest.json")["artifact_count_excluding_manifest"]}, sort_keys=True))
    print("N26_ARTIFACT_FINALIZATION_COMPLETE")


if __name__ == "__main__":
    main()
