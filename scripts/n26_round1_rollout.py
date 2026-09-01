#!/usr/bin/env python3
"""Generate N26 Round-1 policy-conditioned replay without duplicating humans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(".")
OUT = ROOT / "outputs/n26/dense_dataset"
sys.path.insert(0, str(ROOT / "scripts"))
from n26_build_dense_dataset import (  # noqa: E402
    HARD_CAP, MAX_CANDIDATES, MAX_MEMORY, MEMORY_KIND, NEG_CAP, NONE_INDEX,
    POS_CAP, MemoryToken, memory_counts, memory_snapshot,
)
from n26_ccsam_model import CCSAM, CCSAMConfig  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


@torch.no_grad()
def predict(model: CCSAM, device: torch.device, arrays: dict[str, np.ndarray], state: int, parent_memory: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> tuple[int, np.ndarray]:
    memory_clip, memory_meta, memory_kind, memory_mask = parent_memory
    kwargs = {
        "candidate_clip": torch.from_numpy(arrays["candidate_clip"][state : state + 1]).to(device),
        "candidate_scalar": torch.from_numpy(arrays["candidate_scalar"][state : state + 1]).to(device),
        "candidate_mask": torch.from_numpy(arrays["candidate_mask"][state : state + 1]).to(device),
        "memory_clip": torch.from_numpy(memory_clip[None]).to(device),
        "memory_meta": torch.from_numpy(memory_meta[None]).to(device),
        "memory_kind": torch.from_numpy(memory_kind[None]).to(device),
        "memory_mask": torch.from_numpy(memory_mask[None]).to(device),
    }
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**kwargs, memory_mode="positive_negative")
    return int(output["logits"].argmax(1).item()), output["logits"][0].float().cpu().numpy()


def run(split: str, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    source_path = OUT / f"round0_{split}.npz"
    with np.load(source_path, allow_pickle=False) as z:
        arrays = {name: z[name].copy() for name in z.files}
    parent_rows = [json.loads(line) for line in (OUT / f"round0_{split}_parents.jsonl").open(encoding="utf-8") if line.strip()]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_name = "Round0 CC-SAM rollout" if str(checkpoint.get("stage", "")).startswith("round0") else "Final Round1 CC-SAM evaluation rollout"
    policy_version = "N26_ROUND1_CCSAM_ROLLOUT_V1" if str(checkpoint.get("stage", "")).startswith("round0") else "N26_FINAL_CCSAM_EVAL_ROLLOUT_V1"
    model = CCSAM(CCSAMConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    parents = len(parent_rows)
    new_memory_clip = np.zeros((parents, MAX_MEMORY, 1280), dtype=np.float16)
    new_memory_meta = np.zeros((parents, MAX_MEMORY, 10), dtype=np.float16)
    new_memory_kind = np.full((parents, MAX_MEMORY), -1, dtype=np.int8)
    new_memory_mask = np.zeros((parents, MAX_MEMORY), dtype=bool)
    new_memory_pre_mask = np.zeros((parents, MAX_MEMORY), dtype=bool)
    arrays["pair_valid"][:] = False
    arrays["rejected_index"][:] = -1
    memories: dict[tuple[str, int], dict[str, list[MemoryToken]]] = defaultdict(lambda: {"positive": [], "negative": [], "hard": []})
    ledger: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    correction_id = 0

    for parent, row in enumerate(parent_rows):
        sequence, frame, public_id = row["sequence"], int(row["frame"]), int(row["public_identity_id"])
        states = np.flatnonzero(arrays["parent"] == parent)
        if not len(states):
            raise RuntimeError(f"missing states for parent {parent}")
        root = arrays["memory_clip"][parent, 0].astype(np.float32)
        memory = memories[(sequence, public_id)]
        clip, meta, kind, pre_mask, latest = memory_snapshot(root, memory, frame)
        mask = kind >= 0
        new_memory_clip[parent], new_memory_meta[parent], new_memory_kind[parent] = clip, meta, kind
        new_memory_mask[parent], new_memory_pre_mask[parent] = mask, pre_mask
        values = np.asarray(memory_counts(memory, frame), dtype=np.float16)
        arrays["candidate_scalar"][states, :, 44:50] = values[None, None, :]
        if latest is not None:
            latest_negative = [token for token in memory["negative"] if token.correction_id == latest]
            if latest_negative:
                for state in states:
                    valid = np.flatnonzero(arrays["candidate_mask"][state])
                    if len(valid):
                        rejected = max(valid, key=lambda index: float(arrays["candidate_clip"][state, index].astype(np.float32) @ latest_negative[-1].embedding))
                        if int(arrays["target"][state]) == NONE_INDEX or int(rejected) != int(arrays["target"][state]):
                            arrays["rejected_index"][state] = rejected
                            arrays["pair_valid"][state] = True

        canonical = int(row["canonical_state_index"])
        selected, logits = predict(model, device, arrays, canonical, (clip, meta, kind, mask))
        target = int(arrays["target"][canonical])
        wrong = selected != target
        explicit_negative = False
        if wrong:
            correction_id += 1
            if selected < MAX_CANDIDATES and bool(arrays["candidate_mask"][canonical, selected]):
                embedding = arrays["candidate_clip"][canonical, selected].astype(np.float32)
                memory["negative"].append(MemoryToken(embedding, MEMORY_KIND["HUMAN_EXPLICIT_NEGATIVE"], frame, correction_id))
                memory["negative"] = memory["negative"][-NEG_CAP:]
                explicit_negative = True
                ledger.append({"split": split, "parent_event_id": parent, "event_key": row["event_key"], "sequence": sequence, "frame": frame, "public_identity_id": public_id, "candidate_rank": selected + 1, "memory_kind": "HUMAN_EXPLICIT_NEGATIVE", "source": "ROUND1_CCSAM_SELECTED_THEN_SIMULATED_HUMAN_REJECTED", "applies_from_next_parent_only": True})
            if target < MAX_CANDIDATES:
                embedding = arrays["candidate_clip"][canonical, target].astype(np.float32)
                memory["positive"].append(MemoryToken(embedding, MEMORY_KIND["HUMAN_EXPLICIT_POSITIVE"], frame, correction_id))
                memory["positive"] = memory["positive"][-POS_CAP:]
                ledger.append({"split": split, "parent_event_id": parent, "event_key": row["event_key"], "sequence": sequence, "frame": frame, "public_identity_id": public_id, "candidate_rank": target + 1, "memory_kind": "HUMAN_EXPLICIT_POSITIVE", "source": "SIMULATED_HUMAN_CORRECTED_TARGET_AFTER_ROUND1_ERROR", "applies_from_next_parent_only": True})
        valid_wrong = [index for index in np.flatnonzero(arrays["candidate_mask"][canonical]) if index != target and index != selected]
        if valid_wrong:
            hard = max(valid_wrong, key=lambda index: float(logits[index]))
            memory["hard"].append(MemoryToken(arrays["candidate_clip"][canonical, hard].astype(np.float32), MEMORY_KIND["MODEL_INDUCED_HARD_NEGATIVE"], frame, -1))
            memory["hard"] = memory["hard"][-HARD_CAP:]
        output_rows.append({
            **row, "source_parent_event_id": parent, "round1_selected": selected,
            "round1_selected_correct": not wrong, "correction_event": wrong,
            "explicit_negative_written": explicit_negative, "current_feedback_used_by_current_states": False,
            "policy_version": policy_version,
        })

    arrays["memory_clip"] = new_memory_clip
    arrays["memory_meta"] = new_memory_meta
    arrays["memory_kind"] = new_memory_kind
    arrays["memory_mask"] = new_memory_mask
    arrays["memory_pre_mask"] = new_memory_pre_mask
    destination = OUT / f"round1_{split}.npz"
    atomic_npz(destination, arrays)
    parent_path = OUT / f"round1_{split}_parents.jsonl"
    with parent_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    ledger_path = OUT / f"round1_{split}_memory_ledger.jsonl"
    with ledger_path.open("w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "split": split, "parents": parents, "states": len(arrays["target"]),
        "policy": policy_name, "policy_version": policy_version, "policy_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "policy_checkpoint_sha256": sha256(checkpoint_path), "correction_events": sum(row["correction_event"] for row in output_rows),
        "explicit_negative_writes": sum(row["explicit_negative_written"] for row in output_rows),
        "positive_writes": sum(row["memory_kind"] == "HUMAN_EXPLICIT_POSITIVE" for row in ledger),
        "pair_states": int(arrays["pair_valid"].sum()), "selected_none": sum(row["round1_selected"] == NONE_INDEX for row in output_rows),
        "parent_weight_sum": float(arrays["sample_weight"].sum()),
        "same_observation_parent_as_round0": True, "aggregate_cluster_key": "source_parent_event_id",
        "duplicate_parent_counted_as_independent_human": False, "current_feedback_used_by_current_state": False,
        "candidate_protocol": "frozen N25-R static GFN top-5; no union", "val25_read": False,
        "npz": str(destination.relative_to(ROOT)), "npz_sha256": sha256(destination),
        "parents_jsonl": str(parent_path.relative_to(ROOT)), "parents_sha256": sha256(parent_path),
        "memory_ledger": str(ledger_path.relative_to(ROOT)), "memory_ledger_sha256": sha256(ledger_path),
        "state_counts": dict(Counter(row["state_label"] for row in output_rows)),
    }
    (OUT / f"round1_{split}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train30", "cal10"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    summary = run(args.split, args.checkpoint, torch.device(args.device))
    print(json.dumps(summary, sort_keys=True), flush=True)
    print(f"N26_ROUND1_ROLLOUT_DONE split={args.split}", flush=True)


if __name__ == "__main__":
    main()
