#!/usr/bin/env python3
"""Pinned official TrackEval and paired sequence-cluster uncertainty."""
from __future__ import annotations
import argparse
from scripts.n72r20r4_common import *
from scripts.n72r20r3r2r3_pipeline import run_trackeval_many, parse_trackeval, trackeval_summary

METRICS = ("HOTA", "AssA", "DetA", "LocA", "IDF1", "MOTA", "IDSW", "FP", "FN")


def evaluate(group: str, names: list[str], sequences: list[str], *, split: str = "train") -> dict:
    check_storage(reserve_gib=0.01)
    if split not in {"train", "val"}:
        raise ValueError("test evaluation not authorized")
    tracker_root = ASSETS / group / "trackers"
    eval_root = ASSETS / group / "trackeval"
    seqmap = tracker_root / "seqmap.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n"+"\n".join(sequences)+"\n")
    result = run_trackeval_many(tracker_root, eval_root, names, seqmap, gt_split=split, gt_folder=DATASET / split)
    metrics = {name: trackeval_summary(parse_trackeval(eval_root, name, sequences)) for name in names}
    if any(value[k] is None for value in metrics.values() for k in METRICS):
        raise RuntimeError("TrackEval missing mandatory metrics")
    provenance = {"command": result["command"], "returncode": result["returncode"], "log_path": result["log"], "log_sha256": sha256(Path(result["log"])), "seqmap_sha256": sha256(seqmap), "gt_split": split, "sequences": sequences, "trackeval_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "third_party/MOTIP/TrackEval", text=True).strip(), "full_sequence": True, "all_variants_share_config": True, "runtime_gt_read": False}
    write_json(ASSETS / group / "EVALUATION.json", {"metrics": metrics, "provenance": provenance})
    return {"metrics": metrics, "provenance": provenance}


def paired_metrics(baseline: dict, treatment: dict, *, seed: int = 720401, samples: int = 2000) -> dict:
    keys = sorted(baseline["per_sequence"])
    if keys != sorted(treatment["per_sequence"]):
        raise ValueError("paired sequence axis mismatch")
    rng = np.random.default_rng(seed)
    bootstrap = {}
    for metric in METRICS:
        deltas = np.asarray([treatment["per_sequence"][s][metric] - baseline["per_sequence"][s][metric] for s in keys])
        draws = deltas[rng.integers(0, len(keys), size=(samples, len(keys)))].mean(axis=1)
        bootstrap[metric] = {"sequence_macro_delta": float(deltas.mean()), "interval_95": np.quantile(draws, [0.025,0.975]).tolist(), "paired_per_sequence": dict(zip(keys, deltas.tolist()))}
    return {"combined_delta": {k: treatment[k]-baseline[k] for k in METRICS}, "bootstrap": bootstrap, "method": "paired sequence-cluster bootstrap of sequence-macro deltas; combined TrackEval delta separately reported", "seed": seed, "resamples": samples, "HOTA_CI_lower_gt_zero": bootstrap["HOTA"]["interval_95"][0] > 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    args = parser.parse_args()
    result = evaluate(args.group, args.names, args.sequences)
    print(json.dumps(result["metrics"], sort_keys=True))
