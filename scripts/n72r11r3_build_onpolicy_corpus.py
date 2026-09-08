#!/usr/bin/env python3
"""Build a causal, model-on-policy V3 corpus from a sealed N72R11 corpus.

The source corpus contains frozen candidate observations and offline labels.  This
adapter never reads the labels or dataset GT while rolling the model state.  For
each event it runs the supplied V3 scorer in frame order, applies the shared
``TemporalIdentityState`` update using the scorer's own selection as the
development proxy, and writes a new immutable tensor/metadata pair.  The copied
labels remain offline-only supervision and are never inputs to the rollout.

This is an isolated research artifact.  It is deliberately marked
``simulated_from_gt`` because the upstream event tape is simulated, not human
evidence, and it cannot authorize production use.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from sam3_intermot.association.target_edge_interface import select_candidate_from_logits  # noqa: E402
from sam3_intermot.reacquisition.temporal_state_policy import (  # noqa: E402
    TEMPORAL_FEATURE_SCHEMA,
    build_temporal_features,
    initialize_temporal_state,
    state_audit,
    state_memory_arrays,
    update_temporal_state,
)
from scripts import n72r11_train_v3 as train_v3  # noqa: E402


DEFAULT_SOURCE_ROOT = ROOT / "outputs/N72R11R3/bootstrap_corpus"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/N72R11R3/onpolicy_corpus"
DEFAULT_STAGE_PATH = ROOT / "outputs/N72R11R3/stage_12_onpolicy_corpus.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, content: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        if isinstance(content, bytes):
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write(path, json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(temporary, **dict(arrays))
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def device_from(value: str) -> torch.device:
    device = torch.device(str(value))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    if device.type == "cuda" and device.index is not None and int(device.index) >= torch.cuda.device_count():
        raise RuntimeError(f"requested {device}, only {torch.cuda.device_count()} CUDA devices are visible")
    return device


def _row_model_input(
    model: torch.nn.Module,
    arrays: Mapping[str, np.ndarray],
    index: int,
    state: Any,
    temporal: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run one causal row without exposing labels to the model input."""

    recent, recent_mask, long_term, long_mask, distractors, distractor_mask = state_memory_arrays(state)
    tensors = (
        torch.as_tensor(arrays["candidate_features"][index : index + 1], dtype=torch.float32, device=device),
        torch.as_tensor(arrays["candidate_mask"][index : index + 1], dtype=torch.bool, device=device),
        torch.as_tensor(arrays["source_features"][index : index + 1], dtype=torch.float32, device=device),
        torch.as_tensor(arrays["human_anchor"][index : index + 1], dtype=torch.float32, device=device),
        torch.as_tensor(recent[None], dtype=torch.float32, device=device),
        torch.as_tensor(recent_mask[None], dtype=torch.bool, device=device),
        torch.as_tensor(long_term[None], dtype=torch.float32, device=device),
        torch.as_tensor(long_mask[None], dtype=torch.bool, device=device),
        torch.as_tensor(distractors[None], dtype=torch.float32, device=device),
        torch.as_tensor(distractor_mask[None], dtype=torch.bool, device=device),
        torch.as_tensor(arrays["neighbor_feature"][index : index + 1], dtype=torch.float32, device=device),
        torch.as_tensor(temporal[None], dtype=torch.float32, device=device),
    )
    with torch.no_grad():
        output = model(*tensors)[0].detach().float().cpu().numpy()
    if output.ndim != 1 or not np.isfinite(output).all():
        raise RuntimeError(f"non-finite on-policy logits at row {index}")
    return output


def roll_split(
    model: torch.nn.Module,
    arrays: Mapping[str, np.ndarray],
    metadata: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    count = int(arrays["labels"].shape[0])
    if len(metadata) != count:
        raise RuntimeError(f"metadata/array count mismatch: {len(metadata)} != {count}")
    output_arrays = {key: np.asarray(value).copy() for key, value in arrays.items()}
    # Only these state-derived inputs are replaced.  Labels, candidate features,
    # source features and offline score columns are copied verbatim.
    for key in (
        "recent_trusted_memory", "recent_trusted_mask", "long_term_trusted_memory",
        "long_term_trusted_mask", "distractor_memory", "distractor_mask", "temporal_features",
    ):
        output_arrays[key] = np.zeros_like(arrays[key])

    states: dict[str, Any] = {}
    output_metadata: list[dict[str, Any]] = []
    accepted = 0
    state_update_count = 0
    for index, source_row in enumerate(metadata):
        row = dict(source_row)
        event_id = str(row["event_id"])
        state = states.get(event_id)
        if state is None:
            state = initialize_temporal_state(
                anchor_feature=np.asarray(arrays["human_anchor"][index], dtype=np.float32),
                anchor_box=row.get("anchor_box", [0.0, 0.0, 1.0, 1.0]),
            )
            states[event_id] = state
        before = state_audit(state)
        recent, recent_mask, long_term, long_mask, distractors, distractor_mask = state_memory_arrays(state)
        output_arrays["recent_trusted_memory"][index] = recent
        output_arrays["recent_trusted_mask"][index] = recent_mask
        output_arrays["long_term_trusted_memory"][index] = long_term
        output_arrays["long_term_trusted_mask"][index] = long_mask
        output_arrays["distractor_memory"][index] = distractors
        output_arrays["distractor_mask"][index] = distractor_mask
        candidate_sources = [str(value) for value in row.get("candidate_sources", [])]
        temporal = build_temporal_features(
            state,
            frame_horizon=int(row["frame_horizon"]),
            causal_top_score=float(row.get("causal_top_score", 0.0)),
            causal_second_score=float(row.get("causal_second_score", 0.0)),
            causal_margin=float(row.get("causal_margin", 0.0)),
            has_future_requery=any(value == "FUTURE_FRAME_REQUERY" for value in candidate_sources),
        )
        output_arrays["temporal_features"][index] = temporal
        logits = _row_model_input(model, output_arrays, index, state, temporal, device)
        candidate_count = int(arrays["candidate_counts"][index])
        if candidate_count < 1 or candidate_count > len(row.get("candidate_uids", [])):
            raise RuntimeError(f"invalid candidate count at row {index}")
        candidate_logits = logits[:candidate_count]
        # The tensor path keeps the sealed padded candidate axis so the V3
        # architecture is unchanged; its NONE logit is therefore the final
        # column, not the first padded column after the active candidates.
        none_logit = float(logits[-1])
        uids = [str(value) for value in row["candidate_uids"][:candidate_count]]
        selection = select_candidate_from_logits(candidate_logits, none_logit, uids)
        accepted_now = bool(selection["accepted"])
        accepted += int(accepted_now)
        selected_index = int(selection["selected_index"])
        selected_uid = None if not accepted_now else uids[selected_index]
        candidates = []
        for candidate_index in range(candidate_count):
            # The sealed decoder tensor uses an all-zero 512-D prefix for an
            # absent feature and carries availability in the candidate token.
            # Keep that tensor unchanged, but expose absent features as None to
            # the shared state policy so they cannot enter trusted or
            # distractor memory as fabricated unit vectors.
            candidate_feature = np.asarray(
                arrays["candidate_features"][index, candidate_index, :512],
                dtype=np.float32,
            )
            if candidate_feature.size != 512 or not np.isfinite(candidate_feature).all():
                raise RuntimeError(f"invalid candidate feature at row {index}, candidate {candidate_index}")
            feature_available = bool(np.linalg.norm(candidate_feature) > 1.0e-6)
            candidates.append(
                {
                    "candidate_uid": uids[candidate_index],
                    "box_xyxy": row["candidate_boxes"][candidate_index],
                    "candidate_source": candidate_sources[candidate_index] if candidate_index < len(candidate_sources) else "UNKNOWN",
                    "feature": candidate_feature if feature_available else None,
                    "feature_available": feature_available,
                    "official_raw_sam_id": None,
                    "native_scope": None,
                }
            )
        margin = float(selection["best_minus_second_margin"])
        state_update = update_temporal_state(
            state,
            candidates=candidates,
            target_uid=selected_uid,
            selected_uid=selected_uid,
            selected_score=None if not accepted_now else float(selection["selected_score"]),
            selected_margin=margin,
            fused_target_scores=np.asarray(candidate_logits, dtype=np.float64),
            frame_horizon=int(row["frame_horizon"]),
            assigned_candidate=None if not accepted_now else candidates[selected_index],
            base_top_score=float(row.get("causal_top_score", 0.0)),
            base_second_score=float(row.get("causal_second_score", 0.0)),
        )
        state_update_count += 1
        row["on_policy_selection"] = {
            "selected_index": selected_index,
            "selected_uid": selected_uid,
            "accepted": accepted_now,
            "none_logit": none_logit,
            "top_candidate_logit": float(selection["best_score"]),
            "second_candidate_logit": None if selection["second_score"] is None else float(selection["second_score"]),
            "margin": margin,
        }
        row["on_policy_state_before"] = before
        row["on_policy_state_update"] = state_update
        row["on_policy_state_after"] = state_audit(state)
        row["on_policy_runtime_future_gt_used"] = False
        row["on_policy_gt_used_for_runtime_decision"] = False
        row["on_policy_temporal_feature_schema"] = list(TEMPORAL_FEATURE_SCHEMA)
        output_metadata.append(row)

    for key, value in output_arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise RuntimeError(f"on-policy output {key} is non-finite")
    summary = {
        "examples": count,
        "event_count": len({str(row["event_id"]) for row in output_metadata}),
        "sequence_count": len({str(row["sequence"]) for row in output_metadata}),
        "accepted_candidate_count": accepted,
        "none_count": count - accepted,
        "state_update_count": state_update_count,
    }
    return output_arrays, output_metadata, summary


def stage_base(stage: str) -> dict[str, Any]:
    return {
        "schema_version": "N72R11R3_STAGE_STATUS_V1",
        "stage": stage,
        "started_at_utc": now_utc(),
        "runtime_future_gt_used": False,
        "gt_used_for_runtime_decision": False,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
    }


def main() -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage-path", type=Path, default=DEFAULT_STAGE_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resource-censored", action="store_true")
    parser.add_argument("--split", choices=("all", "train", "validation"), default="all")
    args = parser.parse_args()
    source_root = args.source_root if args.source_root.is_absolute() else ROOT / args.source_root
    checkpoint = args.model_checkpoint if args.model_checkpoint.is_absolute() else ROOT / args.model_checkpoint
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    stage_path = args.stage_path if args.stage_path.is_absolute() else ROOT / args.stage_path
    status = stage_base("N72R11R3-12-ON-POLICY-CORPUS")
    status.update({
        "source_root": str(source_root),
        "model_checkpoint": str(checkpoint),
        "resource_censored_development": bool(args.resource_censored),
    })
    try:
        source_manifest = read_json(source_root / "corpus_manifest.json")
        if bool(source_manifest.get("resource_censored_development", False)) != bool(args.resource_censored):
            raise RuntimeError("source corpus censoring mode does not match --resource-censored")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        train_v3.CORPUS_ROOT = source_root
        device = device_from(str(args.device))
        model, payload = train_v3.load_checkpoint(checkpoint, device)
        model.eval()
        output_root.mkdir(parents=True, exist_ok=True)
        split_summaries: dict[str, Any] = {}
        splits_to_run = ("train", "validation") if args.split == "all" else (str(args.split),)
        for split in splits_to_run:
            arrays, metadata, source_summary = train_v3.load_split(split)
            rolled_arrays, rolled_metadata, rollout_summary = roll_split(model, arrays, metadata, device)
            npz_path = output_root / f"{split}.npz"
            metadata_path = output_root / f"{split}_metadata.jsonl"
            atomic_npz(npz_path, rolled_arrays)
            atomic_jsonl(metadata_path, rolled_metadata)
            split_summaries[split] = {
                "source": source_summary,
                "rollout": rollout_summary,
                "npz": str(npz_path),
                "npz_sha256": sha256_file(npz_path),
                "metadata": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
            }
            atomic_json(output_root / f"split_summary_{split}.json", split_summaries[split])

        # Split workers are intentionally independent.  Only publish the
        # canonical manifest after both immutable split artifacts are present;
        # a killed worker therefore cannot leave a false complete corpus.
        for split in ("train", "validation"):
            summary_path = output_root / f"split_summary_{split}.json"
            if summary_path.is_file():
                split_summaries[split] = read_json(summary_path)
        complete = all(split in split_summaries for split in ("train", "validation"))
        manifest = {
            "schema_version": "N72R11R3_ON_POLICY_V3_CORPUS_V1",
            "status": "PASS_N72R11R3_ON_POLICY_CORPUS" if complete else "PARTIAL_N72R11R3_ON_POLICY_CORPUS_SPLIT",
            "created_at_utc": now_utc(),
            "source_corpus_manifest": str(source_root / "corpus_manifest.json"),
            "source_corpus_manifest_sha256": sha256_file(source_root / "corpus_manifest.json"),
            "model_checkpoint": str(checkpoint),
            "model_checkpoint_sha256": sha256_file(checkpoint),
            "model_checkpoint_schema": payload.get("temporal_feature_schema"),
            "splits": split_summaries,
            "splits_completed_in_this_process": list(splits_to_run),
            "split_worker_mode": "one independent process per split; canonical manifest requires train+validation",
            "runtime_rollout": "V3 scorer own selections update shared TemporalIdentityState; labels never read",
            "state_policy": "sam3_intermot.reacquisition.temporal_state_policy",
            "temporal_feature_schema": list(TEMPORAL_FEATURE_SCHEMA),
            "runtime_future_gt_used": False,
            "gt_used_for_runtime_decision": False,
            "offline_labels_copied_not_read_for_rollout": True,
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
            "not_real_human_evidence": True,
            "production_authorized": False,
            "resource_censored_development": bool(args.resource_censored),
        }
        manifest_path = output_root / "corpus_manifest.json"
        if complete:
            atomic_json(manifest_path, manifest)
        status.update({
            "status": manifest["status"],
            "device": str(device),
            "corpus_manifest": str(manifest_path) if complete else None,
            "corpus_manifest_sha256": sha256_file(manifest_path) if complete else None,
            "model_checkpoint_sha256": sha256_file(checkpoint),
            "splits": split_summaries,
            "splits_completed_in_this_process": list(splits_to_run),
            "temporal_feature_schema": list(TEMPORAL_FEATURE_SCHEMA),
            "finished_at_utc": now_utc(),
        })
        atomic_json(stage_path, status)
        print(json.dumps(status, sort_keys=True))
        return 0
    except Exception as exc:
        failure = output_root / "attempts" / f"onpolicy_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        payload = {
            **status,
            "status": "FAIL_N72R11R3_ON_POLICY_CORPUS",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failure_artifact": str(failure),
            "finished_at_utc": now_utc(),
        }
        atomic_json(failure, payload)
        atomic_json(stage_path, payload)
        print(json.dumps({"status": payload["status"], "failure_artifact": str(failure), "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
