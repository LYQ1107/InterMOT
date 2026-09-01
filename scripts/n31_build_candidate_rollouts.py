#!/usr/bin/env python3
"""N31-D: real H20 rollouts for the frozen correction-state candidate library."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SAM3_ROOT = ROOT / "third_party" / "sam3"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from sam3_intermot.adaptation.correction_state_candidates import (  # noqa: E402
    BOX_PROMPTED_SAM_PSEUDO_MASK,
    BOX_RECTANGLE_MASKLET,
    BOX_SANITIZED_RECTANGLE_MASKLET,
    candidate_mask_features,
    interactive_box_candidates,
    rectangle_mask,
    write_target_mask,
)
from sam3_intermot.backend.sam3_state_snapshot import restore_continuation_state, snapshot_continuation_state  # noqa: E402
from scripts.n29_lit_online_replay import (  # noqa: E402
    DecoderCapture,
    _get_official_decoder,
    _image_files,
    _install_official_box_singleton,
    _make_backend,
    _read_gt,
    _session,
)
from scripts.n29r_paired_replay import _horizon_metrics, _load_manifest  # noqa: E402


CHECKPOINT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
HARD_MANIFEST = ROOT / "outputs/n29r/hard_episode_manifest.json"
OUT_DIR = ROOT / "outputs/n31"
CANDIDATES = (
    "S0_restore_old_state",
    "S1_exact_rectangle_masklet",
    "S2_sanitized_rectangle_masklet",
    "S3_interactive_sam_top1_predicted_iou",
    "S4_interactive_sam_second_token",
    "S5_interactive_sam_lowest_token",
)


def _default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    raise TypeError(type(value).__name__)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_default).encode("utf-8")).hexdigest()


def _add_refine_history(backend: Any, frame: int, public_id: int) -> None:
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    raw_id = int(backend._ext_to_sam.get(int(public_id), int(public_id)))
    add_history = getattr(backend._predictor.model, "add_action_history", None)
    if add_history is not None:
        add_history(state, action_type="refine", frame_idx=int(frame), obj_ids=[raw_id])


def _prefix_box(backend: Any, frame: int, public_id: int, fallback: list[float]) -> np.ndarray:
    for observation in backend._output_cache.get(int(frame), []):
        if int(getattr(observation, "sam_object_id", -1)) == int(public_id):
            return np.asarray(observation.box_xyxy, dtype=float)
    return np.asarray(fallback, dtype=float)


def _mask_features(*, candidate: str, box: list[float], state: Mapping[str, Any], token_index: int = -1, predicted_iou: float = 0.0, mask_area_ratio: float = 0.0) -> list[float]:
    base = candidate_mask_features({
        "box_xyxy": box,
        "token_index": int(token_index),
        "predicted_iou": float(predicted_iou),
        "video_width": int(state["orig_width"]),
        "video_height": int(state["orig_height"]),
        "mask_area_ratio": float(mask_area_ratio),
    })
    one_hot = [1.0 if candidate == name else 0.0 for name in CANDIDATES]
    return [float(value) for value in base] + one_hot


def _reward(metrics: Mapping[str, Any], protected_regression: bool) -> float:
    iou = float(metrics.get("mean_box_iou_visible") or 0.0)
    missing = float(metrics.get("missing_prediction_rate_visible") if metrics.get("missing_prediction_rate_visible") is not None else 1.0)
    drift = float(metrics.get("mask_area_drift") if metrics.get("mask_area_drift") is not None else 0.0)
    return float(iou - 0.5 * missing - 0.1 * float(bool(protected_regression)) - 0.05 * drift)


def _run_candidate(
    *,
    backend: Any,
    episode: Mapping[str, Any],
    sequence: Path,
    prefix_snapshot: Any,
    prefix_box: np.ndarray,
    candidate: str,
    interactive_cache: Optional[Mapping[str, Any]],
) -> tuple[dict[str, Any], Optional[Mapping[str, Any]]]:
    correction = int(episode["correction_frame"])
    query_end = int(episode["query_end"])
    public_id = int(episode["public_id"])
    restore_continuation_state(backend, prefix_snapshot)
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    writer: dict[str, Any] = {"status": "NOT_WRITTEN"}
    features: list[float]
    interactive_cache = interactive_cache
    try:
        if candidate == "S0_restore_old_state":
            writer = {"status": "NOT_WRITTEN", "method": "restore_old_state"}
            features = _mask_features(candidate=candidate, box=prefix_box.tolist(), state=state)
        elif candidate in ("S1_exact_rectangle_masklet", "S2_sanitized_rectangle_masklet"):
            box = np.asarray(episode["correction_box"], dtype=float)
            if candidate == "S2_sanitized_rectangle_masklet":
                box = backend._sanitize_box(box)
            mask = rectangle_mask(box, int(state["orig_height"]), int(state["orig_width"]), device=state["device"])
            provenance = BOX_RECTANGLE_MASKLET if candidate == "S1_exact_rectangle_masklet" else BOX_SANITIZED_RECTANGLE_MASKLET
            writer = write_target_mask(backend, frame_idx=correction, public_id=public_id, mask=mask, provenance=provenance)
            _add_refine_history(backend, correction, public_id)
            features = _mask_features(candidate=candidate, box=box.tolist(), state=state, mask_area_ratio=float((mask > 0.5).float().mean().item()))
        else:
            if interactive_cache is None:
                interactive_cache = interactive_box_candidates(backend, frame_idx=correction, box_xyxy=episode["correction_box"])
            order = list(interactive_cache["rank_order"])
            if candidate == "S3_interactive_sam_top1_predicted_iou":
                token = int(order[0])
            elif candidate == "S4_interactive_sam_second_token":
                token = int(order[1])
            else:
                token = int(order[-1])
            if token >= len(interactive_cache["masks"]):
                raise RuntimeError(f"candidate token {token} unavailable; token_count={len(interactive_cache['masks'])}")
            probabilities = interactive_cache["masks"][token]
            predicted_iou = float(interactive_cache["predicted_iou"][token])
            writer = write_target_mask(backend, frame_idx=correction, public_id=public_id, mask=probabilities, provenance=BOX_PROMPTED_SAM_PSEUDO_MASK)
            writer["token_index"] = int(token)
            writer["predicted_iou"] = predicted_iou
            writer["token_count"] = int(len(interactive_cache["masks"]))
            writer["rank_position"] = int(order.index(token)) if token in order else None
            _add_refine_history(backend, correction, public_id)
            features = _mask_features(candidate=candidate, box=list(episode["correction_box"]), state=state, token_index=token, predicted_iou=predicted_iou, mask_area_ratio=float((probabilities > 0.5).float().mean().item()))
        future = backend.propagate(correction + 1, query_end, start_frame_index=correction + 1)
        metrics = _horizon_metrics(future, _read_gt(sequence), episode)
        h20 = metrics.get("20", {})
        protected_regression = not bool(writer.get("protected_identity_namespace_unchanged", True))
        row = {
            "status": "PASS",
            "available": True,
            "episode_id": str(episode["episode_id"]),
            "sequence": str(episode["sequence"]),
            "split": str(episode["split"]),
            "candidate": candidate,
            "correction_frame": correction,
            "query_end": query_end,
            "writer": {key: value for key, value in writer.items() if key not in {"protected_state_before", "protected_state_after"}},
            "features": features,
            "metrics": metrics,
            "reward": _reward(h20, protected_regression),
            "protected_identity_regression": bool(protected_regression),
            "future_frame_count": len(future),
            "future_gt_used_for_selection": False,
            "future_gt_used_for_posthoc_evaluation": True,
        }
        return row, interactive_cache
    except Exception as exc:
        unavailable = candidate.startswith("S3_") or candidate.startswith("S4_") or candidate.startswith("S5_")
        return {
            "status": "NOT_AVAILABLE" if unavailable else "PARTIAL",
            "available": False,
            "episode_id": str(episode["episode_id"]),
            "sequence": str(episode["sequence"]),
            "split": str(episode["split"]),
            "candidate": candidate,
            "features": [],
            "failure": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(limit=16),
            "future_gt_used_for_selection": False,
            "future_gt_used_for_posthoc_evaluation": False,
        }, interactive_cache


def _cluster_ci(rows: list[Mapping[str, Any]], *, candidate: str, baseline: str, field: str = "mean_box_iou_visible") -> Optional[list[float]]:
    grouped: dict[str, list[float]] = {}
    by_episode: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        by_episode[(str(row["episode_id"]), str(row["candidate"]))] = row
    for episode_id, cand_row in [(key[0], value) for key, value in by_episode.items() if key[1] == candidate]:
        base = by_episode.get((episode_id, baseline))
        if base is None:
            continue
        a = cand_row.get("metrics", {}).get("20", {}).get(field)
        b = base.get("metrics", {}).get("20", {}).get(field)
        if a is None or b is None:
            continue
        sequence = str(cand_row["sequence"])
        grouped.setdefault(sequence, []).append(float(a) - float(b))
    if not grouped:
        return None
    values = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=float)
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(31031 + len(candidate))
    draws = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def run(*, manifest_path: Path, checkpoint: Path, output: Path, oracle_output: Path, limit: Optional[int], resume: bool) -> dict[str, Any]:
    manifest, manifest_sha = _load_manifest(manifest_path)
    episodes = list(manifest["episodes"])
    if len(episodes) != 50 and limit is None:
        raise ValueError(f"N31-D requires the frozen 50-episode hard source, got {len(episodes)}")
    if limit is not None:
        episodes = episodes[: int(limit)]
    partial = output.with_suffix(".partial.json")
    previous: dict[tuple[str, str], dict[str, Any]] = {}
    if resume and partial.is_file():
        try:
            old = json.loads(partial.read_text(encoding="utf-8"))
            previous = {(str(row["episode_id"]), str(row["candidate"])): row for row in old.get("rows", [])}
        except Exception:
            previous = {}
    backend = _make_backend(checkpoint)
    capture = None
    decoder = None
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for episode in episodes:
            episode_id = str(episode["episode_id"])
            sequence = Path(episode["sequence_path"])
            needed = [name for name in CANDIDATES if (episode_id, name) not in previous]
            if not needed:
                rows.extend(previous[(episode_id, name)] for name in CANDIDATES)
                continue
            _session(backend, sequence)
            if decoder is None:
                decoder = _get_official_decoder(backend)
                capture = DecoderCapture(decoder)
            gt = _read_gt(sequence)
            init = int(episode["initialization_frame"])
            correction = int(episode["correction_frame"])
            public_id = int(episode["public_id"])
            init_box = np.asarray(gt[init][int(episode["dataset_identity"])], dtype=float)
            backend.add_box(init, public_id, init_box)
            _install_official_box_singleton(backend, frame_idx=init, public_id=public_id, box_xyxy=init_box)
            if capture is not None:
                capture.reset(target_call=max(0, correction - init))
            backend.propagate(init, correction, start_frame_index=init)
            prefix_snapshot = snapshot_continuation_state(backend)
            prefix_box = _prefix_box(backend, correction, public_id, list(episode["correction_box"]))
            interactive_cache = None
            for candidate in CANDIDATES:
                if (episode_id, candidate) in previous:
                    rows.append(previous[(episode_id, candidate)])
                    continue
                row, interactive_cache = _run_candidate(
                    backend=backend,
                    episode=episode,
                    sequence=sequence,
                    prefix_snapshot=prefix_snapshot,
                    prefix_box=prefix_box,
                    candidate=candidate,
                    interactive_cache=interactive_cache,
                )
                rows.append(row)
                _write(partial, {"protocol": "N31-D-CANDIDATE-ROLLOUTS", "status": "PARTIAL", "rows": rows, "manifest_sha256": manifest_sha})
    finally:
        if capture is not None:
            capture.close()
        backend.close()
    oracle_rows = [row for row in rows if row.get("available")]
    summary: dict[str, Any] = {
        "protocol": "N31-D-CANDIDATE-ROLLOUT-SUMMARY",
        "status": "PASS" if oracle_rows else "NOT_RUN",
        "candidate_order": list(CANDIDATES),
        "episode_count": len(episodes),
        "row_count": len(rows),
        "available_row_count": len(oracle_rows),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "future_gt_used_for_selection": False,
        "future_gt_used_for_posthoc_evaluation": True,
        "candidate_metrics": {},
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    for candidate in CANDIDATES:
        candidate_rows = [row for row in rows if row.get("candidate") == candidate and row.get("available")]
        h20 = [row.get("metrics", {}).get("20", {}) for row in candidate_rows]
        summary["candidate_metrics"][candidate] = {
            "available_count": len(candidate_rows),
            "availability_rate": float(len(candidate_rows) / len(episodes)) if episodes else None,
            "mean_reward": None if not candidate_rows else float(np.mean([float(row.get("reward", 0.0)) for row in candidate_rows])),
            "mean_h20_iou": None if not h20 else float(np.mean([float(m["mean_box_iou_visible"]) for m in h20 if m.get("mean_box_iou_visible") is not None])) if any(m.get("mean_box_iou_visible") is not None for m in h20) else None,
            "mean_h20_success": None if not h20 else float(np.mean([float(m["success_at_0_5_visible"]) for m in h20 if m.get("success_at_0_5_visible") is not None])) if any(m.get("success_at_0_5_visible") is not None for m in h20) else None,
        }
    result = {
        "protocol": "N31-D-CANDIDATE-ROLLOUTS",
        "status": summary["status"],
        "rows": rows,
        "summary": summary,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "candidate_order": list(CANDIDATES),
        "checkpoint": str(checkpoint),
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selection": False,
        "elapsed_seconds": summary["elapsed_seconds"],
    }
    _write(output, result)
    baseline = "S0_restore_old_state"
    gains = {}
    for candidate in CANDIDATES[1:]:
        ci = _cluster_ci(rows, candidate=candidate, baseline=baseline)
        values = []
        by_id = {(str(row["episode_id"]), str(row["candidate"])): row for row in rows}
        for episode in episodes:
            left = by_id.get((str(episode["episode_id"]), candidate))
            right = by_id.get((str(episode["episode_id"]), baseline))
            if left and right and left.get("available") and right.get("available"):
                a = left.get("metrics", {}).get("20", {}).get("mean_box_iou_visible")
                b = right.get("metrics", {}).get("20", {}).get("mean_box_iou_visible")
                if a is not None and b is not None:
                    values.append(float(a) - float(b))
        gains[candidate] = {"mean_h20_iou_gain_vs_s0": None if not values else float(np.mean(values)), "sequence_cluster_ci95": ci, "sample_count": len(values)}
    available_candidates = [candidate for candidate in CANDIDATES if summary["candidate_metrics"][candidate]["available_count"] > 0]
    best = max(available_candidates, key=lambda name: summary["candidate_metrics"][name]["mean_reward"]) if available_candidates else None
    best_not_fixed = []
    by_episode = {(str(row["episode_id"]), str(row["candidate"])): row for row in rows}
    for episode in episodes:
        candidates_here = [row for (eid, _), row in by_episode.items() if eid == str(episode["episode_id"]) and row.get("available")]
        if not candidates_here:
            continue
        winner = max(candidates_here, key=lambda row: float(row.get("reward", -1e9)))
        best_not_fixed.append(str(winner["candidate"]) != "S2_sanitized_rectangle_masklet")
    oracle_pass = False
    if best is not None:
        gain = gains.get(best, {})
        ci = gain.get("sequence_cluster_ci95")
        oracle_pass = bool(
            gain.get("mean_h20_iou_gain_vs_s0") is not None
            and float(gain["mean_h20_iou_gain_vs_s0"]) >= 0.01
            and ci is not None
            and float(ci[0]) > 0.0
            and best_not_fixed
            and float(np.mean(best_not_fixed)) >= 0.30
        )
    oracle = {
        "protocol": "N31-D-ORACLE-GATE",
        "status": "PASS" if oracle_pass else "FAIL",
        "baseline": baseline,
        "best_candidate_by_reward": best,
        "gains_vs_s0": gains,
        "best_not_s2_rate": None if not best_not_fixed else float(np.mean(best_not_fixed)),
        "thresholds": {"h20_gain": 0.01, "sequence_cluster_ci_lower": 0.0, "best_candidate_not_fixed_rate": 0.30},
        "future_gt_used_for_selection": False,
        "future_gt_used_for_posthoc_oracle": True,
        "candidate_rollout_artifact": str(output),
    }
    _write(oracle_output, oracle)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HARD_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "candidate_rollout_index.json")
    parser.add_argument("--oracle-output", type=Path, default=OUT_DIR / "candidate_oracle_gate.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(manifest_path=args.manifest, checkpoint=args.checkpoint, output=args.output, oracle_output=args.oracle_output, limit=args.limit, resume=args.resume)
    print(json.dumps({key: result.get(key) for key in ("protocol", "status", "candidate_order", "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
