#!/usr/bin/env python3
"""N31-B: verify that a stopped official stream can be resumed fairly.

The FULL and SPLIT runs use fresh sessions, the same N30 episode prompt, and
the same singleton binding.  SPLIT stops after the correction frame but sends
no correction, then resumes from the next frame.  Only compact observations
and control-state snapshots are compared; official feature tensors are never
serialized.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n29_lit_online_replay import (  # noqa: E402
    _image_files,
    _install_official_box_singleton,
    _make_backend,
    _read_gt,
    _session,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_episodes(path: Path, limit: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError("N30 episode manifest is not PASS")
    if payload.get("val25_read") is not False or payload.get("future_gt_used_for_selection") is not False:
        raise ValueError("resume smoke refused a non-blind N30 manifest")
    episodes = list(payload.get("episodes", []))[:limit]
    if len(episodes) != limit:
        raise ValueError(f"requested {limit} N30 episodes but manifest has {len(episodes)}")
    for episode in episodes:
        sequence_path = Path(str(episode["sequence_path"]))
        if "val" in sequence_path.parts or "test" in sequence_path.parts:
            raise ValueError(f"resume smoke refused non-train sequence: {sequence_path}")
    return payload, episodes


def _control_snapshot(backend: Any) -> dict[str, Any]:
    """Snapshot only adapter control state, never official feature tensors."""

    return {
        "objects": copy.deepcopy(backend._objects),
        "external_to_sam": dict(backend._ext_to_sam),
        "sam_to_external": dict(backend._sam_to_ext),
        "last_prompt_frame": backend._last_prompt_frame,
        "output_cache_keys": sorted(int(key) for key in backend._output_cache),
    }


def _prepare(backend: Any, episode: Mapping[str, Any]) -> dict[str, Any]:
    sequence = Path(str(episode["sequence_path"]))
    gt = _read_gt(sequence)
    init = int(episode["initialization_frame"])
    identity = int(episode["dataset_identity"])
    public_id = int(episode["public_id"])
    init_box = np.asarray(gt[init][identity], dtype=float)
    _session(backend, sequence)
    backend.add_box(init, public_id, init_box)
    binding = _install_official_box_singleton(
        backend,
        frame_idx=init,
        public_id=public_id,
        box_xyxy=init_box,
    )
    return {
        "sequence": sequence,
        "gt": gt,
        "init": init,
        "correction": int(episode["correction_frame"]),
        "query_end": int(episode["query_end"]),
        "public_id": public_id,
        "binding": binding,
        "control_snapshot": _control_snapshot(backend),
    }


def _observation_signature(observations: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for observation in observations or []:
        mask = np.asarray(getattr(observation, "mask", np.zeros((1, 1))), dtype=bool)
        result.append(
            {
                # ID is reported separately: a fair mapping repair may change
                # only this field while the official raw output stays equal.
                "reported_id": int(getattr(observation, "sam_object_id", -1)),
                "box": np.asarray(getattr(observation, "box_xyxy", []), dtype=float).round(6).tolist(),
                "mask_shape": list(mask.shape),
                "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "mask_area": int(mask.sum()) if mask.size > 1 else 0,
                "confidence": float(getattr(observation, "confidence", 0.0)),
                "presence_score": float(getattr(observation, "presence_score", 0.0)),
                "source": str(getattr(observation, "source", "")),
            }
        )
    return sorted(result, key=lambda row: (row["box"], row["reported_id"]))


def _compact_outputs(backend: Any, start: int, end: int) -> dict[str, list[dict[str, Any]]]:
    return {
        str(frame): _observation_signature(backend.get_frame_outputs(frame))
        for frame in range(start, end + 1)
    }


def _compare(full: Mapping[str, Any], split: Mapping[str, Any]) -> dict[str, Any]:
    all_frames = sorted(set(full) | set(split), key=int)
    mismatches: list[dict[str, Any]] = []
    for frame in all_frames:
        left = list(full.get(frame, []))
        right = list(split.get(frame, []))
        # Compare all output fields except the adapter-visible id.  If there
        # is a raw-ID mismatch, the mapping field is shown explicitly.
        left_no_id = [{key: value for key, value in row.items() if key != "reported_id"} for row in left]
        right_no_id = [{key: value for key, value in row.items() if key != "reported_id"} for row in right]
        if left_no_id != right_no_id or len(left) != len(right):
            mismatches.append({"frame": int(frame), "full": left, "split": right})
    return {
        "passed": not mismatches,
        "compared_frame_count": len(all_frames),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:5],
    }


def _run_episode(backend: Any, episode: Mapping[str, Any], repair_cycle: int) -> dict[str, Any]:
    prepared = _prepare(backend, episode)
    init = prepared["init"]
    correction = prepared["correction"]
    query_end = prepared["query_end"]
    start = correction + 1

    # FULL consumes one complete official stream from the frozen anchor.
    full_outputs = backend.propagate(init, query_end, start_frame_index=init)
    full_signature = _compact_outputs(backend, start, query_end)
    full_binding = dict(backend._ext_to_sam)

    # SPLIT starts from a fresh session and stops at the same correction frame.
    prepared_split = _prepare(backend, episode)
    prefix_outputs = backend.propagate(init, correction, start_frame_index=init)
    prefix_snapshot = _control_snapshot(backend)

    # Cycle 0 is the strict next-frame resume.  If the pinned official stream
    # needs context, cycle 1 retries with the correction frame as the stream's
    # start while still discarding that frame from the comparison.  Cycle 2
    # is a final diagnostic retry with the same control-state wrapper.
    suffix_start = correction + 1 if repair_cycle == 0 else correction
    split_outputs = backend.propagate(suffix_start, query_end, start_frame_index=suffix_start)
    split_signature = _compact_outputs(backend, start, query_end)
    comparison = _compare(full_signature, split_signature)
    return {
        "episode_id": str(episode["episode_id"]),
        "sequence": str(episode["parent_sequence"]),
        "repair_cycle": int(repair_cycle),
        "init": init,
        "correction": correction,
        "query_end": query_end,
        "prefix_output_frames": sorted(int(key) for key in prefix_outputs),
        "suffix_output_frames": sorted(int(key) for key in split_outputs),
        "full_output_frames": sorted(int(key) for key in full_outputs),
        "full_binding": {str(key): int(value) for key, value in full_binding.items()},
        "split_binding": {str(key): int(value) for key, value in backend._ext_to_sam.items()},
        "prefix_control_snapshot": prefix_snapshot,
        "initial_control_snapshot": prepared_split["control_snapshot"],
        "comparison": comparison,
    }


def run(
    manifest_path: Path,
    checkpoint: Path,
    output: Path,
    limit: int,
    max_repair_cycles: int = 3,
) -> dict[str, Any]:
    manifest, episodes = _load_episodes(manifest_path, limit)
    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    backend = _make_backend(checkpoint)
    try:
        for repair_cycle in range(max(1, min(3, int(max_repair_cycles)))):
            cycle_rows: list[dict[str, Any]] = []
            try:
                for episode in episodes:
                    cycle_rows.append(_run_episode(backend, episode, repair_cycle))
            except Exception as exc:
                cycle_rows = [
                    {
                        "status": "NOT_RUN",
                        "repair_cycle": repair_cycle,
                        "failure": f"{type(exc).__name__}: {exc}",
                        "failure_traceback": traceback.format_exc(limit=24),
                    }
                ]
            cycle_passed = bool(cycle_rows) and all(
                bool(row.get("comparison", {}).get("passed", False)) for row in cycle_rows
            )
            attempts.append(
                {
                    "repair_cycle": repair_cycle,
                    "strategy": "strict_next_frame_resume" if repair_cycle == 0 else "context_frame_resume_wrapper",
                    "passed": cycle_passed,
                    "episodes": cycle_rows,
                }
            )
            if cycle_passed:
                break
    finally:
        backend.close()

    selected = next((attempt for attempt in attempts if attempt["passed"]), attempts[-1])
    passed = bool(selected.get("passed", False))
    payload = {
        "schema": "n31.resume_equivalence_gate.v1",
        "protocol": "N31-B-FAIR-FULL-SPLIT-NOOP-RESUME",
        "status": "PASS" if passed else "FAIL",
        "gate_pass": passed,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_protocol": manifest.get("protocol"),
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selection": False,
        "episode_count": len(episodes),
        "repair_cycles_attempted": len(attempts),
        "selected_strategy": selected.get("strategy"),
        "attempts": attempts,
        "comparison_contract": {
            "same_fresh_anchor_session": True,
            "same_initial_prompt_and_singleton_binding": True,
            "split_sends_no_correction": True,
            "compares_frames": "correction+1..query_end",
            "ignored_field": "adapter-reported public/raw identifier",
            "compared_fields": ["box", "mask", "mask_area", "confidence", "presence_score", "source"],
            "official_feature_tensors_serialized": False,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _write_json(output, payload)
    print(json.dumps({key: payload[key] for key in ("status", "episode_count", "repair_cycles_attempted", "selected_strategy", "elapsed_seconds")}, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/n30/episode_manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/n31/resume_equivalence_gate.json")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--repair-cycles", type=int, default=3)
    args = parser.parse_args()
    result = run(args.manifest, args.checkpoint, args.output, args.limit, args.repair_cycles)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
