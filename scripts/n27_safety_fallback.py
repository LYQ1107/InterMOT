#!/usr/bin/env python3
"""One predeclared group-conformal fallback after primary safety failure.

The fallback computes a per-sequence split-conformal upper quantile of the
OOF nonconformity scores of incorrect candidate commitments and deploys the
maximum group threshold.  It is deliberately conservative and may return no
usable coverage; it is never tuned against cal10.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from n27_train_safety import threshold_metrics


ROOT = Path(".")
OUT = ROOT / "outputs/n27"


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    if len(values) == 0:
        return 1.0
    ordered = np.sort(values)
    rank = int(math.ceil((len(ordered) + 1) * (1.0 - alpha))) - 1
    return float(ordered[min(max(rank, 0), len(ordered) - 1)])


def main() -> None:
    with np.load(OUT / "safety_oof.npz", allow_pickle=False) as payload:
        oof = {key: payload[key].copy() for key in payload.files}
    target = oof["target"].astype(np.int64)
    dataset = oof["dataset"].astype(np.int8)
    sequence = oof["sequence"].astype(np.int16)
    group = np.asarray([f"{int(d)}:{int(s)}" for d, s in zip(dataset, sequence)], dtype="U32")
    results = {}
    for method in ("b10", "apcr"):
        score = oof[f"{method}_selection_score"].astype(np.float32) if f"{method}_selection_score" in oof else None
        if score is None:
            raise RuntimeError("missing OOF score")
        selected = np.load(OUT / "data" / ("apcr_rollout_external_heldout.npz"), allow_pickle=False)[f"selected_{method}"] if method == "apcr" else None
        # The OOF file is intentionally score-only; selected labels are fixed
        # by the corresponding dynamic rollout and are concatenated here.
        with np.load(OUT / "data/apcr_rollout_external_heldout.npz", allow_pickle=False) as external, np.load(OUT / "data/apcr_rollout_dance_train.npz", allow_pickle=False) as dance:
            selected = np.concatenate([external[f"selected_{method}"], dance[f"selected_{method}"]]).astype(np.int64)
        group_thresholds = {}
        for item in np.unique(group):
            group_rows = group == item
            errors = group_rows & (selected < 5) & (selected != target)
            group_thresholds[str(item)] = conformal_quantile(score[errors], 0.10)
        threshold = max(group_thresholds.values()) if group_thresholds else math.inf
        metrics = threshold_metrics(score, selected, target, group, threshold, bootstrap=True) if math.isfinite(threshold) else {"status": "NO_FINITE_THRESHOLD"}
        passed = bool(math.isfinite(threshold) and metrics.get("coverage", 0.0) >= 0.05 and metrics.get("precision_lower_wilson_95", 0.0) >= 0.90 and metrics.get("absent_false_accept_upper_wilson_95", 1.0) <= 0.0726 and metrics.get("sequence_bootstrap_risk_upper_95", 1.0) <= 0.10 and metrics.get("sequence_count", 0) >= 5 and metrics.get("max_sequence_commit_fraction", 1.0) <= 0.5)
        results[method] = {"status": "FALLBACK_PASS" if passed else "FALLBACK_NO_USABLE_COVERAGE", "alpha": 0.10, "threshold": threshold, "group_threshold_count": len(group_thresholds), "metrics": metrics, "group_thresholds": group_thresholds, "cal10_read": False, "val25_read": False}
    artifact = {"phase": "N27", "primary_switch_used_once": True, "fallback": "GROUP_CONFORMAL_RISK_CONTROL_ONE_SWITCH_ONLY", "methods": results, "any_method_passed": any(item["status"] == "FALLBACK_PASS" for item in results.values()), "cal10_read": False, "val25_read": False}
    path = OUT / "safety_conformal_fallback.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    print(json.dumps(artifact, indent=2, sort_keys=True), flush=True)
    print("N27_SAFETY_CONFORMAL_FALLBACK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
