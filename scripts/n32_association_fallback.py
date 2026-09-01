#!/usr/bin/env python3
"""N32 fallback: bounded real multi-ID association challenger.

This is intentionally separate from the N32 spatial selector.  It uses only
the already completed N30 train-fold multi-ID artifact and never pretends that
an unseen candidate's future IoU is known when the artifact did not record it.
Consequently learned-association quality is reported with an explicit known
candidate coverage, while B10 retains the complete recorded baseline score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/n30/multi_identity_write_ablation.json"
OUT = ROOT / "outputs/n32/association_fallback_results.json"
POLICY = "M3_official_spatial_plus_b10"
FEATURE_DIM = 6

from sam3_intermot.adaptation.correction_rls import CRLSConfig, CorrectionRLS  # noqa: E402


def _default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix(case: Mapping[str, Any]) -> tuple[np.ndarray, int, np.ndarray, list[dict[str, Any]]]:
    branch = case["branches"][POLICY]
    assignment = branch["assignment"]
    matrix = np.asarray(assignment["matrix"], dtype=np.float32)
    candidate_count = int(assignment["candidate_count"])
    base_assignment = np.asarray(assignment["assignment"], dtype=int)
    rows = list(assignment["assignment_rows"])
    if matrix.ndim != 2 or matrix.shape[0] != len(rows) or len(base_assignment) != len(rows):
        raise ValueError(f"invalid N30 assignment shape in {case['case_id']}")
    if candidate_count <= 0 or candidate_count > matrix.shape[1]:
        raise ValueError(f"invalid candidate count in {case['case_id']}")
    return matrix, candidate_count, base_assignment, rows


def _features(matrix: np.ndarray, candidate_count: int, base_assignment: np.ndarray) -> np.ndarray:
    """Build correction-time score/geometry features for valid candidates."""
    output = np.zeros((matrix.shape[0], candidate_count, FEATURE_DIM), dtype=np.float32)
    for row_index in range(matrix.shape[0]):
        scores = matrix[row_index, :candidate_count]
        valid = np.isfinite(scores) & (scores > -1.0e8)
        finite = scores[valid]
        row_max = float(np.max(finite)) if len(finite) else 0.0
        row_mean = float(np.mean(finite)) if len(finite) else 0.0
        order = np.argsort(-scores, kind="stable")
        rank = {int(index): position for position, index in enumerate(order) if valid[index]}
        for candidate in range(candidate_count):
            if not valid[candidate]:
                continue
            output[row_index, candidate] = np.asarray([
                float(np.clip(scores[candidate], -1.0, 1.0)),
                float(rank.get(candidate, candidate_count) / max(1, candidate_count - 1)),
                row_max,
                row_mean,
                float(candidate / max(1, candidate_count - 1)),
                float(candidate == int(base_assignment[row_index])),
            ], dtype=np.float32)
    return output


def _samples(cases: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[float] = []
    for case in cases:
        matrix, candidate_count, base_assignment, rows = _matrix(case)
        tape = _features(matrix, candidate_count, base_assignment)
        for row_index, row in enumerate(rows):
            selected = int(base_assignment[row_index])
            if selected < 0 or selected >= candidate_count:
                continue
            # The only legal labels in this stored artifact are the actual
            # selected candidates.  Other candidate cells remain unlabeled.
            correct = bool(row.get("correct_identity_assignment", False))
            features.append(tape[row_index, selected])
            labels.append(1.0 if correct else -1.0)
    if not features:
        raise RuntimeError("N30 artifact contains no legal selected-candidate labels")
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def _apply_assignment(matrix: np.ndarray, deltas: np.ndarray, candidate_count: int) -> np.ndarray:
    adjusted = np.asarray(matrix, dtype=np.float32).copy()
    adjusted[:, :candidate_count] += np.asarray(deltas, dtype=np.float32)
    rows, cols = linear_sum_assignment(-adjusted)
    assignment = np.full(matrix.shape[0], -1, dtype=int)
    for row, column in zip(rows, cols):
        assignment[int(row)] = int(column) if int(column) < candidate_count else -1
    return assignment


def _evaluate_case(case: Mapping[str, Any], assignment: np.ndarray, method: str) -> dict[str, Any]:
    matrix, candidate_count, baseline_assignment, rows = _matrix(case)
    known = []
    selected_scores = []
    changed = 0
    for row_index, row in enumerate(rows):
        candidate = int(assignment[row_index])
        baseline = int(baseline_assignment[row_index])
        if candidate != baseline:
            changed += 1
        if candidate >= 0 and candidate < matrix.shape[1]:
            selected_scores.append(float(matrix[row_index, candidate]))
        if candidate == baseline:
            known.append(row)
    known_iou = [float(row["selected_box_iou_to_gt"]) for row in known if row.get("selected_box_iou_to_gt") is not None]
    known_correct = [bool(row.get("correct_identity_assignment", False)) for row in known]
    known_wrong = [bool(row.get("wrong_public_id_assignment", False)) for row in known]
    return {
        "case_id": str(case["case_id"]),
        "sequence": str(case["sequence"]),
        "method": method,
        "assignment": assignment.tolist(),
        "baseline_assignment": baseline_assignment.tolist(),
        "assignment_change_count": int(changed),
        "row_count": len(rows),
        "known_candidate_row_count": len(known),
        "known_candidate_coverage": float(len(known) / max(1, len(rows))),
        "known_mean_iou": float(np.mean(known_iou)) if known_iou else None,
        "known_id_assignment_accuracy": float(np.mean(known_correct)) if known_correct else None,
        "known_id_switch_rate": float(np.mean(known_wrong)) if known_wrong else None,
        "mean_selected_frozen_score": float(np.mean(selected_scores)) if selected_scores else None,
        "full_future_iou_claim": "NOT_CLAIMED_FOR_UNRECORDED_CANDIDATES" if len(known) < len(rows) else "RECORDED_BASELINE_ROWS_ONLY",
    }


class SmallAssociationMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


def _train_mlp(features: np.ndarray, labels: np.ndarray) -> SmallAssociationMLP:
    torch.manual_seed(3210)
    model = SmallAssociationMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    return model


def _rls_delta(rls: CorrectionRLS, case: Mapping[str, Any]) -> np.ndarray:
    matrix, candidate_count, baseline_assignment, _ = _matrix(case)
    tape = _features(matrix, candidate_count, baseline_assignment)
    return rls.delta("global_correction_writer", tape).astype(np.float32)


def _mlp_delta(model: SmallAssociationMLP, case: Mapping[str, Any]) -> np.ndarray:
    matrix, candidate_count, baseline_assignment, _ = _matrix(case)
    tape = _features(matrix, candidate_count, baseline_assignment)
    with torch.no_grad():
        values = model(torch.from_numpy(tape.reshape(-1, FEATURE_DIM))).reshape(tape.shape[0], tape.shape[1]).numpy()
    # Keep the challenger bounded relative to the frozen B10 score scale.
    return (0.10 * np.tanh(values)).astype(np.float32)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None
    return {
        "case_count": len(rows),
        "known_candidate_coverage": avg("known_candidate_coverage"),
        "known_mean_iou": avg("known_mean_iou"),
        "known_id_assignment_accuracy": avg("known_id_assignment_accuracy"),
        "known_id_switch_rate": avg("known_id_switch_rate"),
        "mean_selected_frozen_score": avg("mean_selected_frozen_score"),
        "assignment_change_rate": float(np.mean([row["assignment_change_count"] / max(1, row["row_count"]) for row in rows])) if rows else None,
        "full_future_iou_claim": "BOUNDED_KNOWN_CANDIDATE_ONLY",
    }


def run(*, source: Path = SOURCE, output: Path = OUT) -> dict[str, Any]:
    artifact = json.loads(source.read_text(encoding="utf-8"))
    if artifact.get("status") != "PASS" or artifact.get("val25_read") is not False or artifact.get("test_labels_used") is not False:
        raise RuntimeError("N30 multi-ID fallback source is not a blind PASS artifact")
    cases = sorted(list(artifact.get("case_results", [])), key=lambda row: (str(row["sequence"]), str(row["case_id"])))
    if len(cases) != 10:
        raise RuntimeError(f"expected 10 bounded N30 cases, got {len(cases)}")
    train_cases = cases[:6]
    selection_cases = cases[6:8]
    calibration_cases = cases[8:]
    train_features, train_labels = _samples(train_cases)
    rls = CorrectionRLS(CRLSConfig(feature_dim=FEATURE_DIM, ridge=1.0, residual_scale=0.10, forgetting=1.0))
    for feature, label in zip(train_features, train_labels):
        rls.update("global_correction_writer", feature[None, :], np.asarray([label], dtype=np.float64))
    mlp = _train_mlp(train_features, train_labels)

    method_case_rows: dict[str, list[dict[str, Any]]] = {"B10_fixed": [], "correction_supervised_RLS": [], "small_pairwise_MLP": []}
    all_evaluated = train_cases + selection_cases + calibration_cases
    for case in all_evaluated:
        matrix, candidate_count, baseline_assignment, _ = _matrix(case)
        deltas_rls = _rls_delta(rls, case)
        deltas_mlp = _mlp_delta(mlp, case)
        assignments = {
            "B10_fixed": baseline_assignment,
            "correction_supervised_RLS": _apply_assignment(matrix, deltas_rls, candidate_count),
            "small_pairwise_MLP": _apply_assignment(matrix, deltas_mlp, candidate_count),
        }
        for method, assignment in assignments.items():
            method_case_rows[method].append(_evaluate_case(case, assignment, method))

    def fold_rows(method: str, fold: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        ids = {str(case["case_id"]) for case in fold}
        return [row for row in method_case_rows[method] if row["case_id"] in ids]

    result = {
        "protocol": "N32-FALLBACK-BOUNDED-REAL-MULTI-ID-ASSOCIATION",
        "status": "PASS",
        "route": "association_fallback",
        "source_artifact": str(source),
        "source_sha256": _sha(source),
        "source_policy": POLICY,
        "case_count": len(cases),
        "sequence_disjoint_split": {"train": [str(case["sequence"]) for case in train_cases], "selection": [str(case["sequence"]) for case in selection_cases], "calibration": [str(case["sequence"]) for case in calibration_cases]},
        "train_sample_count": int(len(train_features)),
        "methods": {},
        "rls": {"feature_dim": FEATURE_DIM, "ridge": 1.0, "residual_scale": 0.10, "train_label_source": "legal N30 selected-candidate identity labels on train cases only", "state_norm": rls.state_norm(["global_correction_writer"])},
        "mlp": {"architecture": "Linear(6,16)->ReLU->Linear(16,1)", "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)", "epochs": 200, "seed": 3210, "train_label_source": "legal N30 selected-candidate identity labels on train cases only"},
        "note": "Alternative candidate future IoU is not reconstructed because N30 stored only the selected candidate's post-hoc IoU; learned methods therefore report known-candidate coverage and do not claim an unobserved MOT gain.",
        "no_more_spatial_selector_scaling": True,
        "future_gt_used_for_selection": False,
        "future_gt_used_for_training_labels": True,
        "val25_read": False,
        "test_labels_used": False,
    }
    for method, rows in method_case_rows.items():
        result["methods"][method] = {
            "all": _aggregate(rows),
            "train": _aggregate(fold_rows(method, train_cases)),
            "selection": _aggregate(fold_rows(method, selection_cases)),
            "calibration": _aggregate(fold_rows(method, calibration_cases)),
            "case_rows": rows,
        }
    result["deployment_authorized"] = False
    result["bounded_fallback_gate"] = {"execution_pass": True, "sequence_disjoint_training": True, "full_future_mot_gain_claim": False, "status": "DIAGNOSTIC_ONLY"}
    _write(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = run(source=args.source, output=args.output)
    print(json.dumps({"protocol": result["protocol"], "status": result["status"], "route": result["route"], "methods": {key: value["all"] for key, value in result["methods"].items()}}, indent=2, default=_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
