#!/usr/bin/env python3
"""Offline root-cause audit after the first learned N72R7 decoder.

This audit reads sealed corpus metadata and learned posthoc metrics.  It does
not alter a checkpoint, select a configuration from future outcomes, or
enter runtime replay.  Its purpose is to route the next single mechanism
round and to make the failed validation evidence reproducible.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n72r7_train_target_id_decoder import (  # noqa: E402
    CHECKPOINT,
    VAL_META,
    VAL_NPZ,
    load_split,
    new_model,
    tensor_batch,
)


POSTHOC = ROOT / "outputs/N72R7/learned_posthoc_attempt1/n72r7_learned_d1_d2_posthoc_results.json"
ORACLE = ROOT / "outputs/N72R7/forensic/union_pool_oracle.json"
TRAINING_MANIFEST = ROOT / "outputs/N72R7/training/decoder_training_manifest.json"
OUTPUT_ROOT = ROOT / "outputs/N72R7/root_cause"
RESULT = OUTPUT_ROOT / "decoder_v1_root_cause.json"
STAGE = ROOT / "outputs/N72R7/stage_08_root_cause_status.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{__import__('os').getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def corpus_baselines() -> dict[str, Any]:
    arrays, metadata = load_split(VAL_NPZ, VAL_META)
    candidate_count = arrays["candidate_features"].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = new_model().to(device)
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    top1: list[int] = []
    top3: list[list[int]] = []
    with torch.no_grad():
        for start in range(0, len(arrays["labels"]), 256):
            indices = np.arange(start, min(start + 256, len(arrays["labels"])))
            candidate, mask, context, _ = tensor_batch(arrays, indices, device)
            logits = model(candidate, mask, context)
            order = torch.argsort(logits, dim=1, descending=True).cpu().tolist()
            top1.extend(int(row[0]) for row in order)
            top3.extend([[int(value) for value in row[:3]] for row in order])
    labels = arrays["labels"].astype(np.int64)
    target = labels < candidate_count
    target_top3 = float(np.mean([int(label) in row[:3] for label, row in zip(labels[target], np.asarray(top3, dtype=object)[target])])) if bool(target.any()) else None
    predicted_none = int(np.sum(np.asarray(top1) == candidate_count))
    target_top1 = float(np.mean(np.asarray(top1)[target] == labels[target])) if bool(target.any()) else None
    nearest_cos: list[bool] = []
    for index, label in enumerate(labels):
        valid = arrays["candidate_mask"][index]
        values = arrays["candidate_features"][index, :, :512] @ arrays["context_features"][index, :512]
        values = np.where(valid, values, -np.inf)
        nearest = int(np.argmax(values)) if bool(valid.any()) else candidate_count
        nearest_cos.append(nearest == int(label))
    return {
        "validation_examples": int(len(labels)),
        "validation_target_examples": int(target.sum()),
        "validation_none_examples": int((~target).sum()),
        "decoder_top1_accuracy": float(np.mean(np.asarray(top1) == labels)),
        "decoder_target_top1_accuracy": target_top1,
        "decoder_target_top3_recall": target_top3,
        "decoder_predicted_none_count": predicted_none,
        "anchor_cosine_nearest_accuracy": float(np.mean(nearest_cos)),
        "anchor_cosine_target_top1_accuracy": float(np.mean(np.asarray(nearest_cos)[target])) if bool(target.any()) else None,
        "validation_sequences": sorted({str(item["sequence"]) for item in metadata}),
        "validation_actions": dict(Counter(str(item["action_type"]) for item in metadata)),
        "device": str(device),
    }


def main() -> None:
    posthoc = json.loads(POSTHOC.read_text(encoding="utf-8"))
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    training = json.loads(TRAINING_MANIFEST.read_text(encoding="utf-8"))
    learned_d1 = posthoc["aggregate"]["D1_vs_D0"]["20"]
    learned_d2 = posthoc["aggregate"]["D2_vs_D0"]["20"]
    result = {
        "schema_version": "N72R7_DECODER_V1_ROOT_CAUSE_AUDIT_V1",
        "status": "PASS_ROOT_CAUSE_AUDIT_SELECTOR_AND_FEATURE_LIMITATION",
        "created_at_utc": now_utc(),
        "inputs": {
            "training_manifest": str(TRAINING_MANIFEST),
            "training_manifest_sha256": sha256_file(TRAINING_MANIFEST),
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "learned_posthoc": str(POSTHOC),
            "learned_posthoc_sha256": sha256_file(POSTHOC),
            "union_pool_oracle": str(ORACLE),
            "union_pool_oracle_sha256": sha256_file(ORACLE),
        },
        "validation_diagnostics": corpus_baselines(),
        "candidate_pool_oracle": oracle,
        "learned_effect": {
            "D1_vs_D0_H20": {"identity_error_reduction": learned_d1.get("identity_error_reduction"), "ci": learned_d1.get("sequence_cluster_bootstrap_95ci"), "correct_crossings": learned_d1.get("true_correct_crossing_count"), "incorrect_crossings": learned_d1.get("true_incorrect_crossing_count"), "protected_regression": learned_d1.get("protected_regression_count"), "candidate_recall": learned_d1.get("candidate_recall")},
            "D2_vs_D0_H20": {"identity_error_reduction": learned_d2.get("identity_error_reduction"), "ci": learned_d2.get("sequence_cluster_bootstrap_95ci"), "correct_crossings": learned_d2.get("true_correct_crossing_count"), "incorrect_crossings": learned_d2.get("true_incorrect_crossing_count"), "protected_regression": learned_d2.get("protected_regression_count"), "candidate_recall": learned_d2.get("candidate_recall")},
        },
        "diagnosis": {
            "primary": "SELECTOR_GENERALIZATION_AND_APPEARANCE_SEPARATION_LIMITATION",
            "secondary": "UNION_POOL_RECALL_REMAINS_A_CEILING_FOR_ADD_AND_RECOVER",
            "evidence": [
                "sequence-disjoint validation accuracy and target top-1 are recorded without using development future outcomes",
                "candidate-level anchor cosine baseline is recorded to distinguish feature separation from decoder capacity",
                "D1 learned route makes almost no assignment change while D2 gains are not CI-positive and include protected regression",
                "D2 candidate recall is inherited from the frozen union pool and therefore cannot be credited to the learned decoder",
            ],
            "not_supported": [
                "no evidence authorizes SAM3 fine-tuning",
                "no evidence authorizes public-ID decoder or LoRA",
                "no evidence authorizes threshold search on development future metrics",
            ],
        },
        "next_round": {
            "route": "R2_CAUSAL_CONTEXT_AND_DISTRACTOR_AWARE_TARGET_DECODER",
            "single_mechanism_change": "rebuild training examples with a deterministic causal motion/raw context and add sealed hard-negative weighting; keep candidate generator, exact solver and public authority frozen",
            "why": "the first decoder was trained on a fixed event-box context unlike the replay's evolving causal context, while validation target/distractor appearance separation is weak",
            "confirmation": "deferred; current 32-event development set is not independent confirmation",
        },
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "production_authorized": False,
    }
    atomic_json(RESULT, result)
    atomic_json(STAGE, {"schema_version": "N72R7_STAGE_08_ROOT_CAUSE_STATUS_V1", "status": result["status"], "created_at_utc": now_utc(), "result": str(RESULT), "result_sha256": sha256_file(RESULT), "next_route": result["next_round"], "runtime_future_gt_used": False, "production_authorized": False})
    print(json.dumps({"status": result["status"], "route": result["next_round"]["route"], "validation": result["validation_diagnostics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
