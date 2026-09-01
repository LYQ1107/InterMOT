#!/usr/bin/env python3
"""N32-C: resumable three-policy strict-future official SAM3 rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
try:
    import torch
except Exception:  # pragma: no cover - the official backend imports torch itself
    torch = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "third_party/sam3") not in sys.path:
    sys.path.insert(0, str(ROOT / "third_party/sam3"))

from sam3_intermot.adaptation.correction_application_policy import (  # noqa: E402
    CorrectionApplicationAction,
    CorrectionApplicationPolicy,
)
from sam3_intermot.adaptation.correction_selector_features import (  # noqa: E402
    FEATURE_NAMES,
    build_selector_features,
)
from sam3_intermot.backend.sam3_state_snapshot import snapshot_continuation_state  # noqa: E402
from sam3_intermot.backend.sam3_state_snapshot import restore_continuation_state  # noqa: E402
from scripts.n29_lit_online_replay import (  # noqa: E402
    _image_files,
    _install_official_box_singleton,
    _make_backend,
    _read_gt,
    _session,
)
from scripts.n29r_paired_replay import _ensure_public_singleton_binding, _horizon_metrics  # noqa: E402


CHECKPOINT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
MANIFEST = ROOT / "outputs/n31/episode_manifest.json"
OUT_DIR = ROOT / "outputs/n32/policy_rollouts"
POLICIES = (
    ("K0_KEEP_OLD", CorrectionApplicationAction.KEEP_OLD_STATE),
    ("K1_APPLY_ENSURE", CorrectionApplicationAction.APPLY_CURRENT_ENSURE),
    ("K2_PROMPT_THEN_RESTORE", CorrectionApplicationAction.PROMPT_THEN_RESTORE),
)


def _default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_expanded_manifest(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "PASS":
        raise ValueError("N31 expanded manifest is not a frozen PASS artifact")
    if payload.get("val25_read") is not False or payload.get("test_labels_used") is not False or payload.get("future_gt_used_for_selection") is not False:
        raise ValueError("N31 expanded manifest violates the blind/causal boundary")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 689:
        raise ValueError("N31 expanded manifest must contain exactly 689 episodes")
    for episode in episodes:
        sequence = str(episode.get("sequence", ""))
        split = str(episode.get("split", ""))
        if "val" in sequence.lower() or "test" in sequence.lower() or "val" in split.lower() or "test" in split.lower():
            raise ValueError(f"N32-C refused non-train episode {episode.get('episode_id')}")
    return payload, _sha256(path)


def _obs_summary(observations: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    for obs in observations:
        mask = np.asarray(getattr(obs, "mask", np.zeros((1, 1))), dtype=bool)
        rows.append({
            "sam_object_id": int(getattr(obs, "sam_object_id", -1)),
            "box_xyxy": np.asarray(getattr(obs, "box_xyxy", []), dtype=float).tolist(),
            "confidence": float(getattr(obs, "confidence", 0.0) or 0.0),
            "presence_score": None if getattr(obs, "presence_score", None) is None else float(obs.presence_score),
            "mask_shape": list(mask.shape),
            "mask_area": int(mask.sum()) if mask.ndim == 2 and mask.size > 1 else None,
            "source": str(getattr(obs, "source", "")),
            "is_human_verified": bool(getattr(obs, "is_human_verified", False)),
        })
    return sorted(rows, key=lambda row: (row["sam_object_id"], row["box_xyxy"]))


def _observation_digest(observations: Sequence[Any]) -> str:
    return hashlib.sha256(json.dumps(_obs_summary(observations), sort_keys=True).encode("utf-8")).hexdigest()


def _release_cuda_cache() -> None:
    if torch is None or not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()


def _has_three_policy_rows(row: Mapping[str, Any]) -> bool:
    policies = row.get("policies", {})
    policy_names = {name for name, _ in POLICIES}
    return (
        len(policies) == len(policy_names)
        and all(bool(policies.get(name, {}).get("available", False)) for name in policy_names)
    )


def _remove_partial_episode(path: Path, episode_id: str) -> None:
    """Atomically remove one retryable failure before appending its replacement."""
    if not path.is_file():
        return
    temporary = path.with_suffix(path.suffix + ".retry.tmp")
    with path.open("r", encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("episode_id", "")) != episode_id:
                target.write(line)
    temporary.replace(path)


def _temporal_feature_sequence(
    *,
    backend: Any,
    correction_frame: int,
    corrected_box: Sequence[float],
    prefix_outputs: Mapping[int, Sequence[Any]],
    public_id: int,
) -> list[list[float]]:
    """Store the last five causal feature snapshots for the optional GRU.

    Every snapshot is built from outputs through that frame only.  The
    correction box is the current human observation supplied to the selector;
    no future frame or future label is consulted.  The official state fields
    are deliberately retained as the state-at-correction context, while the
    temporal fallback is gated separately from the static result.
    """
    frames = sorted(int(value) for value in prefix_outputs if int(value) <= int(correction_frame))
    frames = frames[-5:] if frames else [int(correction_frame)]
    sequence: list[list[float]] = []
    for frame in frames:
        causal_outputs = {key: value for key, value in prefix_outputs.items() if int(key) <= frame}
        payload = build_selector_features(
            backend=backend,
            correction_frame=frame,
            corrected_box=corrected_box,
            prefix_outputs=causal_outputs,
            public_id=public_id,
        )
        sequence.append(list(payload["features"]))
    return sequence


def _policy_reward(metrics: Mapping[str, Any]) -> float:
    iou = float(metrics.get("mean_box_iou_visible") or 0.0)
    missing = float(metrics.get("missing_prediction_rate_visible") if metrics.get("missing_prediction_rate_visible") is not None else 1.0)
    drift = float(metrics.get("mask_area_drift") if metrics.get("mask_area_drift") is not None else 0.0)
    # Single-ID rollouts have no protected identity, so the explicitly
    # reported regression rate is zero rather than an invented identity score.
    return float(iou - 0.50 * missing - 0.05 * drift)


def _load_previous(path: Path) -> dict[str, dict[str, Any]]:
    previous: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return previous
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"malformed partial JSONL at {path}:{line_number}") from exc
            episode_id = str(row.get("episode_id", ""))
            if not episode_id or episode_id in previous:
                raise RuntimeError(f"duplicate/empty episode in partial JSONL: {path}:{line_number}")
            previous[episode_id] = row
    return previous


def _run_episode(
    backend: Any,
    episode: Mapping[str, Any],
    gt: Mapping[int, Mapping[int, np.ndarray]],
    policy_names: set[str] | None = None,
) -> dict[str, Any]:
    sequence = Path(str(episode["sequence_path"]))
    images = _image_files(sequence)
    init = int(episode["initialization_frame"])
    correction = int(episode["correction_frame"])
    query_end = min(int(episode["query_end"]), len(images) - 1)
    public_id = int(episode["public_id"])
    dataset_identity = int(episode["dataset_identity"])
    if query_end < correction + 20:
        raise ValueError(f"episode lacks H20: {episode['episode_id']}")
    init_box = np.asarray(gt[init][dataset_identity], dtype=float)
    corrected_box = np.asarray(episode["correction_box"], dtype=float)
    started = time.perf_counter()
    _session(backend, sequence)
    backend.add_box(init, public_id, init_box)
    _install_official_box_singleton(backend, frame_idx=init, public_id=public_id, box_xyxy=init_box)
    prefix_outputs = backend.propagate(init, correction, start_frame_index=init)
    raw_current = _obs_summary(prefix_outputs.get(correction, []))
    prefix_digest = _observation_digest(prefix_outputs.get(correction, []))
    raw_recorded = correction in prefix_outputs
    prefix_snapshot = snapshot_continuation_state(backend)
    feature_payload = build_selector_features(
        backend=backend,
        correction_frame=correction,
        corrected_box=corrected_box,
        prefix_outputs=prefix_outputs,
        public_id=public_id,
    )
    temporal_features = _temporal_feature_sequence(
        backend=backend,
        correction_frame=correction,
        corrected_box=corrected_box,
        prefix_outputs=prefix_outputs,
        public_id=public_id,
    )
    policies: dict[str, Any] = {}
    for policy_name, action in POLICIES:
        if policy_names is not None and policy_name not in policy_names:
            continue
        # Every policy must start from the identical post-prefix state.  The
        # previous policy's future propagation is not part of this episode's
        # information and must never contaminate the next branch.
        restore_continuation_state(backend, prefix_snapshot)
        action_started = time.perf_counter()
        ledger: list[dict[str, Any]] = []
        result = CorrectionApplicationPolicy(action).apply(
            backend,
            correction_frame=correction,
            public_id=public_id,
            corrected_box=corrected_box,
            pre_correction_snapshot=prefix_snapshot,
            ledger=ledger,
            raw_output_recorded=raw_recorded,
            ensure_binding=_ensure_public_singleton_binding,
        )
        future: dict[int, list[Any]] = {}
        metrics: dict[str, Any] = {}
        failure = result.failure
        try:
            future = backend.propagate(correction + 1, query_end, start_frame_index=correction + 1)
            metrics = _horizon_metrics(future, gt, {**episode, "query_end": query_end})
        except Exception as exc:
            failure = f"future_{type(exc).__name__}: {exc}"
        h20 = metrics.get("20", {})
        available = bool(future) and failure is None or bool(future) and result.status == "ROLLBACK" and failure is not None
        # Prompt failure followed by a legal rollback is a completed policy
        # outcome; preserve its reason instead of silently counting it as a
        # missing episode.
        policy_row = {
            "policy": policy_name,
            "action_value": int(action.value),
            "status": "PASS" if available else "NOT_RUN",
            "available": available,
            "action_trace": result.to_dict(),
            "ledger": ledger,
            "current_raw_output_recorded": raw_recorded,
            "current_raw_output_digest": prefix_digest,
            "current_delivered_box": corrected_box.tolist(),
            "current_delivered_source": "human_correction",
            "metrics": metrics,
            "reward": None if not metrics else _policy_reward(h20),
            "protected_identity_regression": False,
            "protected_identity_status": "NOT_APPLICABLE_SINGLE_ID",
            "future_frame_count": len(future),
            "prompt_success": bool(result.prompt_returned_target),
            "fallback_used": bool(result.fallback_used),
            "rollback_used": bool(result.rollback_used),
            "mapping_valid": bool(result.mapping_valid),
            "target_state_present": bool(result.target_state_present),
            "feature_vector": list(feature_payload["features"]),
            "feature_names": list(FEATURE_NAMES),
            "temporal_feature_sequence": temporal_features,
            "future_gt_used_for_selection": False,
            "future_gt_used_for_posthoc_evaluation": True,
            "failure": failure,
            "elapsed_seconds": float(time.perf_counter() - action_started),
        }
        policies[policy_name] = policy_row
        # Future observations can retain device-backed tensors until their
        # Python references are dropped.  Release each branch before the
        # next policy so branch isolation is not defeated by allocator state.
        del future
        del metrics
        _release_cuda_cache()
    return {
        "protocol": "N32-C-POLICY-ROLLOUT",
        "status": "PASS" if all(row["available"] for row in policies.values()) else "PARTIAL",
        "episode_id": str(episode["episode_id"]),
        "sequence": str(episode["sequence"]),
        "sequence_path": str(sequence),
        "learning_split": str(episode["learning_split"]),
        "split": str(episode["split"]),
        "correction_frame": correction,
        "query_start": correction + 1,
        "query_end": query_end,
        "public_id": public_id,
        "dataset_identity": dataset_identity,
        "raw_current_output": raw_current,
        "raw_current_output_recorded": raw_recorded,
        "raw_current_output_digest": prefix_digest,
        "current_correction_box": corrected_box.tolist(),
        "feature_audit": {
            "feature_names": list(FEATURE_NAMES),
            "feature_vector": list(feature_payload["features"]),
            "feature_sources": feature_payload["feature_sources"],
            "identity_features_available": feature_payload["identity_features_available"],
            "future_gt_used": False,
            "future_image_used": False,
            "public_id_emitted": False,
            "sequence_id_emitted": False,
            "temporal_feature_sequence_length": len(temporal_features),
        },
        "policies": policies,
        "gt_role": "current legal correction validation and post-hoc train-fold future evaluation only",
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selection": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def run(
    *,
    manifest_path: Path = MANIFEST,
    checkpoint: Path = CHECKPOINT,
    output_dir: Path = OUT_DIR,
    worker_index: int = 0,
    worker_count: int = 1,
    resume: bool = True,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    manifest, manifest_sha = _load_expanded_manifest(manifest_path)
    if manifest.get("status") != "PASS" or manifest.get("val25_read") is not False or manifest.get("test_labels_used") is not False:
        raise ValueError("N31 expanded manifest is not a blind PASS artifact")
    episodes = list(manifest.get("episodes", []))
    if len(episodes) != 689:
        raise ValueError(f"N32-C requires all 689 frozen episodes, got {len(episodes)}")
    sequences = sorted({str(episode["sequence"]) for episode in episodes})
    assigned = set(sequences[int(worker_index)::int(worker_count)])
    selected = [episode for episode in episodes if str(episode["sequence"]) in assigned]
    selected.sort(key=lambda item: (str(item["sequence"]), int(item["correction_frame"]), str(item["episode_id"])))
    if max_episodes is not None:
        selected = selected[: int(max_episodes)]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"worker_{int(worker_index):02d}.jsonl"
    previous = _load_previous(output_path) if resume else {}
    backend = _make_backend(checkpoint)
    gt_cache: dict[str, Mapping[int, Mapping[int, np.ndarray]]] = {}
    processed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for position, episode in enumerate(selected, 1):
            episode_id = str(episode["episode_id"])
            if episode_id in previous and _has_three_policy_rows(previous[episode_id]):
                processed.append(previous[episode_id])
                continue
            existing_row = previous.get(episode_id)
            existing_policies = existing_row.get("policies", {}) if existing_row else {}
            repair_policy_names = {
                name for name, _ in POLICIES
                if not bool(existing_policies.get(name, {}).get("available", False))
            }
            if episode_id in previous:
                _remove_partial_episode(output_path, episode_id)
                previous.pop(episode_id, None)
            sequence = str(episode["sequence"])
            if any(part.lower() in {"val", "test"} for part in Path(str(episode["sequence_path"])).parts) or any(token in sequence.lower() for token in ("val", "test")):
                raise ValueError(f"N32-C refused non-train episode {episode_id}")
            try:
                _release_cuda_cache()
                if sequence not in gt_cache:
                    gt_cache[sequence] = _read_gt(Path(str(episode["sequence_path"])))
                row = _run_episode(
                    backend,
                    episode,
                    gt_cache[sequence],
                    policy_names=repair_policy_names if existing_row else None,
                )
                if existing_row:
                    merged_policies = dict(existing_policies)
                    merged_policies.update(row.get("policies", {}))
                    row["policies"] = merged_policies
                    row["status"] = "PASS" if all(
                        bool(merged_policies.get(name, {}).get("available", False))
                        for name, _ in POLICIES
                    ) else "PARTIAL"
                    row["repair_policy_names"] = sorted(repair_policy_names)
            except Exception as exc:
                _release_cuda_cache()
                row = dict(existing_row) if existing_row else {
                    "protocol": "N32-C-POLICY-ROLLOUT",
                    "status": "NOT_RUN",
                    "episode_id": episode_id,
                    "sequence": sequence,
                    "learning_split": str(episode["learning_split"]),
                    "policies": {},
                    "failure": f"{type(exc).__name__}: {exc}",
                    "failure_traceback": traceback.format_exc(limit=20),
                    "val25_read": False,
                    "test_labels_used": False,
                    "future_gt_used_for_selection": False,
                }
                if existing_row:
                    row["status"] = "PARTIAL"
                    row["failure"] = f"{type(exc).__name__}: {exc}"
                    row["failure_traceback"] = traceback.format_exc(limit=20)
                    row["repair_policy_names"] = sorted(repair_policy_names)
                failures.append(row)
            processed.append(row)
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, default=_default) + "\n")
                handle.flush()
            print(f"N32_WORKER {worker_index} {position}/{len(selected)} {episode_id} {row['status']}", flush=True)
    finally:
        backend.close()
    summary = {
        "protocol": "N32-C-POLICY-ROLLOUT-WORKER",
        "status": "PASS" if processed and not failures and all(row.get("status") == "PASS" for row in processed) else "PARTIAL" if processed else "NOT_RUN",
        "worker_index": int(worker_index),
        "worker_count": int(worker_count),
        "assigned_sequence_count": len(assigned),
        "assigned_sequences": sorted(assigned),
        "requested_episode_count": len(selected),
        "processed_episode_count": len(processed),
        "failure_count": len(failures),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "output": str(output_path),
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selection": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _write(output_dir / f"worker_{int(worker_index):02d}.summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    result = run(
        manifest_path=args.manifest,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        resume=not args.no_resume,
        max_episodes=args.max_episodes,
    )
    print(json.dumps({key: result[key] for key in ("protocol", "status", "worker_index", "requested_episode_count", "processed_episode_count", "failure_count", "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
