#!/usr/bin/env python3
"""Dynamic causal replay of frozen B10 and the final APCR-S checkpoint.

The two policies own separate memories.  Each event is scored before its own
feedback is written, so a changed APCR decision can change later memory and
correction outcomes.  This is the rollout used for safety-head selection and
the correction-response gate; static feature arrays are not substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from n27_apcr_model import APCRS, APCRConfig, feature_tensors


ROOT = Path(".")
OUT = ROOT / "outputs/n27"
DATA = OUT / "data"
N26 = ROOT / "outputs/n26/dense_dataset"
MAX_CANDIDATES = 5
NONE_INDEX = 5
POS_CAP, NEG_CAP, HARD_CAP = 4, 8, 4
B10_LAMBDA, B10_MARGIN = 0.8, 0.02


@dataclass
class Token:
    embedding: np.ndarray
    frame: int
    correction_id: int


class FeatureStore:
    def __init__(self) -> None:
        ids, embeddings = [], []
        for path in sorted(DATA.glob("clipreid_shard*.npz")):
            with np.load(path, allow_pickle=False) as payload:
                ids.append(payload["crop_id"].copy())
                embeddings.append(payload["embedding"].copy())
        self.embedding = np.concatenate(embeddings).astype(np.float32)
        self.index = {item.decode("ascii"): index for index, item in enumerate(np.concatenate(ids))}

    def vector(self, crop_id: str) -> np.ndarray:
        return self.embedding[self.index[crop_id]]


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def max_sim(candidates: np.ndarray, tokens: list[Token]) -> np.ndarray:
    if not tokens:
        return np.zeros(len(candidates), dtype=np.float32)
    return np.max(candidates @ np.stack([token.embedding for token in tokens]).T, axis=1).astype(np.float32)


def recency(frame: int, tokens: list[Token]) -> float:
    return min(1.0, math.log1p(max(0, frame - tokens[-1].frame)) / 8.0) if tokens else 1.0


def snapshot(candidates: np.ndarray, mask: np.ndarray, root: np.ndarray, frame: int, memory: dict[str, list[Token]]) -> dict[str, Any]:
    root_sim = candidates @ root
    pos_sim = max_sim(candidates, memory["positive"])
    neg_sim = max_sim(candidates, memory["negative"])
    hard_sim = max_sim(candidates, memory["hard"])
    base = np.maximum(root_sim, pos_sim) if memory["positive"] else root_sim.copy()
    penalty = np.maximum(0.0, neg_sim - base + B10_MARGIN) if memory["negative"] else np.zeros_like(base)
    b10 = base - B10_LAMBDA * penalty
    b10[~mask] = -2.0
    valid = b10[mask]
    if len(valid):
        order = np.sort(valid)
        margin = float(order[-1] - order[-2]) if len(order) > 1 else 1.0
        prob = np.exp(valid - valid.max())
        prob /= max(float(prob.sum()), 1e-12)
        entropy = -float(np.sum(prob * np.log(np.maximum(prob, 1e-12)))) / math.log(MAX_CANDIDATES)
    else:
        margin, entropy = 0.0, 0.0
    return {
        "root_similarity": root_sim.astype(np.float32), "positive_similarity": pos_sim,
        "negative_similarity": neg_sim, "hard_similarity": hard_sim, "positive_base": base.astype(np.float32),
        "b10_score": b10.astype(np.float32), "has_positive": bool(memory["positive"]),
        "has_negative": bool(memory["negative"]), "has_hard": bool(memory["hard"]),
        "positive_count": len(memory["positive"]) / POS_CAP, "negative_count": len(memory["negative"]) / NEG_CAP,
        "hard_count": len(memory["hard"]) / HARD_CAP, "positive_age": recency(frame, memory["positive"]),
        "negative_age": recency(frame, memory["negative"]), "hard_age": recency(frame, memory["hard"]),
        "margin": margin, "entropy": entropy,
    }


def model_inputs(snap: dict[str, Any], mask: np.ndarray, detector: np.ndarray) -> dict[str, torch.Tensor]:
    def vector(key: str, fill: float = 0.0) -> np.ndarray:
        value = np.asarray(snap[key], dtype=np.float32)
        if value.ndim == 0:
            return np.full(MAX_CANDIDATES, float(value), dtype=np.float32)
        return value

    return feature_tensors({
        "candidate_mask": torch.from_numpy(mask[None]), "b10_score": torch.from_numpy(snap["b10_score"][None]),
        "positive_similarity": torch.from_numpy(vector("positive_similarity")[None]),
        "negative_similarity": torch.from_numpy(vector("negative_similarity")[None]),
        "hard_similarity": torch.from_numpy(vector("hard_similarity")[None]),
        "detector_score": torch.from_numpy(detector[None]), "candidate_count": torch.tensor([float(mask.sum()) / MAX_CANDIDATES]),
        "has_positive": torch.tensor([float(snap["has_positive"])]), "has_negative": torch.tensor([float(snap["has_negative"])]),
        "has_hard": torch.tensor([float(snap["has_hard"])]), "positive_count": torch.tensor([snap["positive_count"]]),
        "negative_count": torch.tensor([snap["negative_count"]]), "hard_count": torch.tensor([snap["hard_count"]]),
        "positive_age": torch.tensor([snap["positive_age"]]), "negative_age": torch.tensor([snap["negative_age"]]),
        "hard_age": torch.tensor([snap["hard_age"]]),
    })


@torch.no_grad()
def apcr_scores(model: APCRS, device: torch.device, snap: dict[str, Any], mask: np.ndarray, detector: np.ndarray) -> np.ndarray:
    features = model_inputs(snap, mask, detector)
    features = {key: value.to(device) for key, value in features.items()}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return model(features)["scores"][0].float().cpu().numpy()


def target_of(event: dict[str, Any]) -> int:
    indices = [index for index, row in enumerate(event["candidates"]) if bool(row["correct"])]
    if len(indices) > 1:
        raise RuntimeError(f"multiple correct candidates {event['event_key']}")
    return indices[0] if indices else NONE_INDEX


def latest_counterfactual(memory: dict[str, list[Token]], latest: int) -> dict[str, list[Token]]:
    if latest < 0:
        return {key: list(value) for key, value in memory.items()}
    return {
        "positive": [token for token in memory["positive"] if token.correction_id != latest],
        "negative": [token for token in memory["negative"] if token.correction_id != latest],
        "hard": list(memory["hard"]),
    }


def update_memory(memory: dict[str, list[Token]], candidates: np.ndarray, event: dict[str, Any], selected: int, target: int, frame: int, feedback_vector: np.ndarray, correction_id: int) -> tuple[bool, bool, int]:
    wrong = selected != target
    negative = positive = False
    if wrong:
        memory["negative"].append(Token(candidates[selected].copy(), frame, correction_id))
        memory["negative"] = memory["negative"][-NEG_CAP:]
        negative = True
        if target < MAX_CANDIDATES:
            memory["positive"].append(Token(feedback_vector.copy(), frame, correction_id))
            memory["positive"] = memory["positive"][-POS_CAP:]
            positive = True
    hard_candidates = [index for index in range(len(candidates)) if index != target and index != selected]
    hard_index = -1
    if hard_candidates:
        # Caller stores the current policy score temporarily on the event.
        hard_index = max(hard_candidates, key=lambda index: float(event["_policy_score"][index]))
        memory["hard"].append(Token(candidates[hard_index].copy(), frame, -1))
        memory["hard"] = memory["hard"][-HARD_CAP:]
    return negative, positive, hard_index


def entropy_margin(scores: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = scores[mask]
    if len(valid) == 0:
        return 0.0, 0.0
    order = np.sort(valid)
    margin = float(order[-1] - order[-2]) if len(order) > 1 else 1.0
    prob = np.exp(valid - valid.max())
    prob /= max(float(prob.sum()), 1e-12)
    entropy = -float(np.sum(prob * np.log(np.maximum(prob, 1e-12)))) / math.log(MAX_CANDIDATES)
    return margin, entropy


def process_events(events: list[dict[str, Any]], model: APCRS, device: torch.device, store: FeatureStore | None, dance_arrays: dict[str, np.ndarray] | None, dance_parents: list[dict[str, Any]] | None, role: str) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    if role == "external_heldout":
        items = [(event, None) for event in events]
    else:
        assert dance_arrays is not None and dance_parents is not None
        items = []
        order = sorted(range(len(dance_parents)), key=lambda index: (dance_parents[index]["sequence"], int(dance_parents[index]["frame"]), int(dance_parents[index]["public_identity_id"])))
        for parent_index in order:
            parent = dance_parents[parent_index]
            state = int(parent["canonical_state_index"])
            items.append((parent, (parent_index, state)))

    memories_b10: dict[tuple[str, str, str], dict[str, list[Token]]] = defaultdict(lambda: {"positive": [], "negative": [], "hard": []})
    memories_apcr: dict[tuple[str, str, str], dict[str, list[Token]]] = defaultdict(lambda: {"positive": [], "negative": [], "hard": []})
    corrections = {"B10": 0, "APCR": 0}
    ledger_counts = Counter()
    correction_ids = {"B10": 0, "APCR": 0}
    rows: dict[str, list[Any]] = defaultdict(list)
    metadata: list[dict[str, Any]] = []
    sequence_to_index: dict[str, int] = {}
    identity_to_index: dict[tuple[str, str, str], int] = {}
    b10_expected: list[int] = []
    for item_index, (event, dance_ref) in enumerate(items):
        if role == "external_heldout":
            assert store is not None
            candidates = store.embedding[[store.index[row["crop_id"]] for row in event["candidates"]]]
            mask = np.ones(len(candidates), dtype=bool)
            root = store.vector(event["root_crop_id"])
            detector = np.asarray([float(row.get("detector_score", 0.0)) for row in event["candidates"]], dtype=np.float32)
            dataset_name, video, track_id = event["dataset"], event["video"], str(event["track_id"])
            frame = int(event["decision_frame"])
            target = target_of(event)
            feedback_vector = store.vector(event["feedback_positive_crop_id"])
            fold = -1 if event.get("fold") is None else int(event["fold"])
            dataset_index = {"BDD100K": 0, "KITTI": 1, "MOT17": 2, "MOT20": 3}[dataset_name]
            source_index = {"GT_BOX": 0, "PUBLIC_DETECTOR_BOX": 1}[event["candidate_source"]]
            identity_key = (dataset_name, video, track_id)
            event_key = event["event_key"]
        else:
            assert dance_arrays is not None and dance_ref is not None
            parent_index, state = dance_ref
            parent = event
            candidates = dance_arrays["candidate_clip"][state].astype(np.float32)
            mask = dance_arrays["candidate_mask"][state].astype(bool)
            detector = dance_arrays["candidate_scalar"][state, :, 5].astype(np.float32)
            # N26 memory row 0 is the frozen target root query.
            root = dance_arrays["memory_clip"][parent_index, 0].astype(np.float32)
            dataset_name, video, track_id = "DanceTrack", parent["sequence"], str(parent["public_identity_id"])
            frame = int(parent["frame"])
            target = int(dance_arrays["target"][state])
            feedback_vector = candidates[target].copy() if target < MAX_CANDIDATES else root.copy()
            fold = int({"dancetrack0001": 2, "dancetrack0002": 0, "dancetrack0006": 4, "dancetrack0008": 4, "dancetrack0012": 3, "dancetrack0015": 4, "dancetrack0016": 4, "dancetrack0020": 4, "dancetrack0023": 0, "dancetrack0024": 2, "dancetrack0027": 0, "dancetrack0029": 1, "dancetrack0032": 2, "dancetrack0033": 0, "dancetrack0037": 3, "dancetrack0039": 4, "dancetrack0044": 2, "dancetrack0045": 0, "dancetrack0049": 3, "dancetrack0051": 3, "dancetrack0052": 2, "dancetrack0055": 2, "dancetrack0057": 1, "dancetrack0062": 2, "dancetrack0066": 1, "dancetrack0068": 2, "dancetrack0069": 1, "dancetrack0072": 2}.get(video, -1))
            dataset_index, source_index = 4, 2
            identity_key = (dataset_name, video, track_id)
            event_key = parent["event_key"]
        scope = (dataset_name, video, track_id)
        if scope not in identity_to_index:
            identity_to_index[scope] = len(identity_to_index)
        if f"{dataset_name}:{video}" not in sequence_to_index:
            sequence_to_index[f"{dataset_name}:{video}"] = len(sequence_to_index)
        mem_b10, mem_apcr = memories_b10[scope], memories_apcr[scope]
        snap_b10 = snapshot(candidates, mask, root, frame, mem_b10)
        snap_apcr = snapshot(candidates, mask, root, frame, mem_apcr)
        selected_b10 = int(np.argmax(snap_b10["b10_score"])) if mask.any() else NONE_INDEX
        selected_apcr_scores = apcr_scores(model, device, snap_apcr, mask, detector)
        selected_apcr = int(np.argmax(selected_apcr_scores)) if mask.any() else NONE_INDEX
        latest_tokens = [*mem_apcr["positive"], *mem_apcr["negative"]]
        latest = max((token.correction_id for token in latest_tokens), default=-1)
        cf_memory = latest_counterfactual(mem_apcr, latest)
        snap_cf = snapshot(candidates, mask, root, frame, cf_memory)
        cf_apcr_scores = apcr_scores(model, device, snap_cf, mask, detector) if latest >= 0 else selected_apcr_scores.copy()
        pair_valid = latest >= 0
        rejected = -1
        latest_neg = [token for token in mem_apcr["negative"] if token.correction_id == latest]
        if latest_neg and mask.any():
            rejected = int(np.argmax(candidates @ latest_neg[-1].embedding))
        b10_expected.append(selected_b10)
        b10_wrong = selected_b10 != target
        apcr_wrong = selected_apcr != target
        if b10_wrong:
            correction_ids["B10"] += 1
        if apcr_wrong:
            correction_ids["APCR"] += 1
        event_b10 = {"_policy_score": snap_b10["b10_score"]}
        event_apcr = {"_policy_score": selected_apcr_scores}
        b10_negative, b10_positive, b10_hard = update_memory(mem_b10, candidates, event_b10, selected_b10, target, frame, feedback_vector, correction_ids["B10"])
        apcr_negative, apcr_positive, apcr_hard = update_memory(mem_apcr, candidates, event_apcr, selected_apcr, target, frame, feedback_vector, correction_ids["APCR"])
        ledger_counts.update({"B10_negative": int(b10_negative), "B10_positive": int(b10_positive), "APCR_negative": int(apcr_negative), "APCR_positive": int(apcr_positive), "APCR_hard": int(apcr_hard >= 0)})
        b10_margin, b10_entropy = entropy_margin(snap_b10["b10_score"], mask)
        apcr_margin, apcr_entropy = entropy_margin(selected_apcr_scores, mask)
        valid_scores = selected_apcr_scores.copy()
        valid_scores[~mask] = -1e4
        soft = np.exp(valid_scores - valid_scores.max()) * mask
        soft /= max(float(soft.sum()), 1e-12)
        cf_valid = cf_apcr_scores.copy()
        cf_valid[~mask] = -1e4
        cf_soft = np.exp(cf_valid - cf_valid.max()) * mask
        cf_soft /= max(float(cf_soft.sum()), 1e-12)
        rows["candidate_mask"].append(np.pad(mask, (0, MAX_CANDIDATES - len(mask)), constant_values=False))
        rows["target"].append(target); rows["target_present"].append(target < MAX_CANDIDATES)
        rows["b10_score"].append(np.pad(snap_b10["b10_score"], (0, MAX_CANDIDATES - len(mask)), constant_values=-2.0))
        rows["apcr_score"].append(np.pad(selected_apcr_scores, (0, MAX_CANDIDATES - len(mask)), constant_values=-1e4))
        rows["cf_apcr_score"].append(np.pad(cf_apcr_scores, (0, MAX_CANDIDATES - len(mask)), constant_values=-1e4))
        rows["selected_b10"].append(selected_b10); rows["selected_apcr"].append(selected_apcr)
        rows["b10_correct"].append(selected_b10 == target); rows["apcr_correct"].append(selected_apcr == target)
        rows["b10_correction_event"].append(b10_wrong); rows["apcr_correction_event"].append(apcr_wrong)
        rows["pair_valid"].append(pair_valid); rows["rejected_index"].append(rejected)
        rows["target_probability"].append(float(soft[target]) if target < MAX_CANDIDATES else 0.0)
        rows["cf_target_probability"].append(float(cf_soft[target]) if target < MAX_CANDIDATES else 0.0)
        rows["b10_margin"].append(b10_margin); rows["b10_entropy"].append(b10_entropy)
        rows["apcr_margin"].append(apcr_margin); rows["apcr_entropy"].append(apcr_entropy)
        rows["max_root_similarity"].append(float(np.max(snap_apcr["root_similarity"][mask])) if mask.any() else -2.0)
        rows["candidate_count"].append(float(mask.sum()) / MAX_CANDIDATES); rows["detector_score"].append(np.pad(detector, (0, MAX_CANDIDATES - len(detector))))
        rows["dataset"].append(dataset_index); rows["sequence"].append(sequence_to_index[f"{dataset_name}:{video}"]); rows["fold"].append(fold); rows["frame"].append(frame); rows["identity"].append(identity_to_index[scope]); rows["candidate_source"].append(source_index)
        rows["correction_id_apcr"].append(correction_ids["APCR"] if apcr_wrong else -1)
        metadata.append({"parent_index": item_index, "event_key": event_key, "dataset": dataset_name, "video": video, "track_id": track_id, "frame": frame, "fold": fold, "target": target, "selected_b10": selected_b10, "selected_apcr": selected_apcr, "b10_correct": selected_b10 == target, "apcr_correct": selected_apcr == target, "b10_correction_event": b10_wrong, "apcr_correction_event": apcr_wrong, "pair_valid": pair_valid, "rejected_index": rejected, "current_feedback_used_by_current_prediction": False, "memory_scope": [dataset_name, video, track_id]})

    output: dict[str, np.ndarray] = {}
    bool_keys = {"candidate_mask", "target_present", "b10_correct", "apcr_correct", "b10_correction_event", "apcr_correction_event", "pair_valid"}
    int8_keys = {"target", "selected_b10", "selected_apcr", "rejected_index", "dataset", "fold", "candidate_source"}
    int16_keys = {"sequence"}
    int32_keys = {"frame", "identity", "correction_id_apcr"}
    float32_keys = {"b10_score", "apcr_score", "cf_apcr_score", "target_probability", "cf_target_probability", "b10_margin", "b10_entropy", "apcr_margin", "apcr_entropy", "max_root_similarity", "candidate_count", "detector_score"}
    for key, value in rows.items():
        if key in bool_keys: dtype = bool
        elif key in int8_keys: dtype = np.int8
        elif key in int16_keys: dtype = np.int16
        elif key in int32_keys: dtype = np.int32
        elif key in float32_keys: dtype = np.float32
        else: dtype = np.float16
        output[key] = np.asarray(value, dtype=dtype)
    summary = {"role": role, "parents": len(metadata), "b10_corrections": int(output["b10_correction_event"].sum()), "apcr_corrections": int(output["apcr_correction_event"].sum()), "b10_present_top1": float(output["b10_correct"][output["target_present"]].mean()), "apcr_present_top1": float(output["apcr_correct"][output["target_present"]].mean()), "pairs": int(output["pair_valid"].sum()), "ledger_counts": dict(ledger_counts), "b10_expected_rows": len(b10_expected), "val25_read": False}
    return output, metadata, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=OUT / "checkpoints/apcr_s_p2_best.pt")
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--include-cal10-ranking-only", action="store_true", help="historical ranking/correction diagnostic; never selects a threshold")
    args = parser.parse_args()
    started = time.monotonic()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = APCRS(APCRConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    store = FeatureStore()
    requests = [json.loads(line) for line in (DATA / "episode_requests.jsonl").open(encoding="utf-8") if line.strip()]
    heldout = [row for row in requests if row["role"] == "external_heldout"]
    external_output, external_metadata, external_summary = process_events(heldout, model, device, store, None, None, "external_heldout")
    with np.load(DATA / "external_heldout_b10_round0.npz", allow_pickle=False) as baseline:
        if not np.array_equal(external_output["selected_b10"], baseline["selected"]):
            mismatch = int(np.sum(external_output["selected_b10"] != baseline["selected"]))
            raise RuntimeError(f"dynamic B10 reproduction mismatch on external heldout: {mismatch}")
    external_path = DATA / "apcr_rollout_external_heldout.npz"
    atomic_npz(external_path, external_output)
    external_meta_path = OUT / "apcr_rollout_external_heldout_metadata.jsonl"
    temporary = external_meta_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in external_metadata: handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    os.replace(temporary, external_meta_path)

    dance_source = N26 / "round0_train30.npz"
    with np.load(dance_source, allow_pickle=False) as payload:
        dance_arrays = {key: payload[key].copy() for key in payload.files}
    dance_parents = [json.loads(line) for line in (N26 / "round0_train30_parents.jsonl").open(encoding="utf-8") if line.strip()]
    dance_output, dance_metadata, dance_summary = process_events(dance_parents, model, device, None, dance_arrays, dance_parents, "dance_train_real_p2")
    dance_path = DATA / "apcr_rollout_dance_train.npz"
    atomic_npz(dance_path, dance_output)
    dance_meta_path = OUT / "apcr_rollout_dance_train_metadata.jsonl"
    temporary = dance_meta_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in dance_metadata: handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    os.replace(temporary, dance_meta_path)
    summary = {"checkpoint": str(checkpoint_path.relative_to(ROOT)), "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), "external": external_summary, "dance": dance_summary, "external_npz": str(external_path.relative_to(ROOT)), "external_npz_sha256": hashlib.sha256(external_path.read_bytes()).hexdigest(), "dance_npz": str(dance_path.relative_to(ROOT)), "dance_npz_sha256": hashlib.sha256(dance_path.read_bytes()).hexdigest(), "val25_read": False}
    if args.include_cal10_ranking_only:
        cal_source = N26 / "round0_cal10.npz"
        with np.load(cal_source, allow_pickle=False) as payload:
            cal_arrays = {key: payload[key].copy() for key in payload.files}
        cal_parents = [json.loads(line) for line in (N26 / "round0_cal10_parents.jsonl").open(encoding="utf-8") if line.strip()]
        cal_output, cal_metadata, cal_summary = process_events(cal_parents, model, device, None, cal_arrays, cal_parents, "historical_cal10_ranking_only")
        cal_path = DATA / "apcr_rollout_cal10_ranking_only.npz"
        atomic_npz(cal_path, cal_output)
        cal_meta_path = OUT / "apcr_rollout_cal10_ranking_only_metadata.jsonl"
        temporary = cal_meta_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in cal_metadata:
                handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
        os.replace(temporary, cal_meta_path)
        expected = np.asarray([5 if int(row["round0_selected"]) < 0 else int(row["round0_selected"]) for row in cal_parents], dtype=np.int64)
        cal_b10_mismatch = int(np.sum(cal_output["selected_b10"] != expected))
        cal_summary.update({"b10_expected_mismatch": cal_b10_mismatch, "threshold_selection": "NOT_RUN", "safety_gate": "NOT_RUN_BECAUSE_SELECTION_POOL_PRIMARY_AND_CONFORMAL_SAFETY_FAILED", "npz": str(cal_path.relative_to(ROOT)), "npz_sha256": hashlib.sha256(cal_path.read_bytes()).hexdigest(), "metadata": str(cal_meta_path.relative_to(ROOT)), "metadata_sha256": hashlib.sha256(cal_meta_path.read_bytes()).hexdigest()})
        summary["historical_cal10_ranking_only"] = cal_summary
    summary["elapsed_seconds"] = time.monotonic() - started
    atomic_json(OUT / "apcr_dynamic_rollout_summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"checkpoint_sha256", "external_npz_sha256", "dance_npz_sha256"}}, indent=2, sort_keys=True), flush=True)
    print("N27_APCR_DYNAMIC_ROLLOUT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
