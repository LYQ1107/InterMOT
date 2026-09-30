#!/usr/bin/env python3
"""Build the fresh, GT-blind N72R20R1 base-score tape.

The scorer is the existing frozen ``score_matrix_pairwise`` path with its
documented deterministic ReID weights and no learned memory.  This script
only converts the new SAM3/OSNet candidate tape into observations, evaluates
the current base scorer, delegates assignment to ``solve_effect_assignment``,
and causally refreshes the ordinary base identity prototypes.  It does not
open DanceTrack images or GT and does not implement another matcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from sam3_intermot.association.effect_assignment import solve_effect_assignment
from sam3_intermot.association.identity_state import IdentityState
from sam3_intermot.association.online_associator import score_matrix_pairwise


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R1_assets")
SEQUENCES = ("dancetrack0001", "dancetrack0002")
NONE_SCORE = 0.0
BASE_REID_WEIGHTS = {"sim": 1.5, "iou": 1.0, "native": 0.5, "gap": 0.1}
BASE_NATIVE_BONUS = 3.0
BASE_POSITIVE_BONUS = 5.0
BASE_EMA = 0.9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            np.asarray(value, dtype=np.float64).tolist(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_zstd_jsonl(path: Path) -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["zstd", "-q", "-d", "-c", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def unit(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (512,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} is not a finite 512-D vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-6:
        raise ValueError(f"{label} is zero")
    return vector / norm


def load_candidate_frames(asset_root: Path, sequence: str) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    sequence_root = asset_root / "candidates" / sequence
    index = read_json(sequence_root / "index.json")
    metadata = read_zstd_jsonl(sequence_root / "metadata.jsonl.zst")
    embedding_count = int(index["embedding_count"])
    embedding_path = sequence_root / "embeddings.f16"
    embeddings = np.memmap(embedding_path, mode="r", dtype=np.float16, shape=(embedding_count, 512))
    output: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    seen_uids: set[str] = set()
    for expected_frame, frame_row in enumerate(metadata):
        if int(frame_row.get("frame", -1)) != expected_frame:
            raise ValueError(f"{sequence}:{expected_frame}: non-contiguous candidate frame")
        rows: list[dict[str, Any]] = []
        for candidate_index, raw in enumerate(frame_row.get("candidates", [])):
            uid = str(raw.get("candidate_uid"))
            if uid in seen_uids or uid in {"", "None"}:
                raise ValueError(f"{sequence}:{expected_frame}: duplicate candidate UID {uid}")
            seen_uids.add(uid)
            offset = int(raw.get("embedding_offset", -1))
            if offset < 0 or offset >= embedding_count:
                raise ValueError(f"{sequence}:{expected_frame}: invalid feature offset")
            feature = unit(np.asarray(embeddings[offset], dtype=np.float32), f"{sequence}:{uid}")
            row = dict(raw)
            row.update(
                {
                    "candidate_index": int(candidate_index),
                    "feature": feature,
                    "feat": feature,
                    "box": np.asarray(raw["box_xyxy"], dtype=np.float64),
                    "native_tid": int(raw["native_tid"]),
                    "native_scope": None,
                    "has_feat": True,
                    "conf": float(raw.get("confidence", raw.get("presence", 0.0)) or 0.0),
                    "presence_score": float(raw.get("presence", 0.0) or 0.0),
                }
            )
            rows.append(row)
        output.append((frame_row, rows))
    return output


def public_assignment_map(solver: Mapping[str, Any]) -> dict[int, str | None]:
    result: dict[int, str | None] = {}
    for item in solver.get("public_assignments", []):
        public = int(item["public_id"])
        uid = item.get("candidate_uid")
        result[public] = None if uid in (None, "", "None") else str(uid)
    return result


def serializable_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in row.items()
        if key not in {"feature", "feat", "box"}
    }
    result["candidate_uid"] = str(row["candidate_uid"])
    result["candidate_index"] = int(row["candidate_index"])
    result["runtime_future_gt_used"] = False
    result["runtime_gt_read"] = False
    result["posthoc_gt_used"] = False
    return result


def assignable_states(bindings: list[Mapping[str, Any]], states: Mapping[int, IdentityState]) -> list[IdentityState]:
    ordered: list[IdentityState] = []
    for binding in bindings:
        public = int(binding["public_id"])
        state = states[public]
        ordered.append(state)
    return ordered


def initialize_states(event: Mapping[str, Any], frame_rows: list[dict[str, Any]]) -> dict[int, IdentityState]:
    by_uid = {str(row["candidate_uid"]): row for row in frame_rows}
    anchor = unit(event["human_anchor"], "human anchor")
    states: dict[int, IdentityState] = {}
    target_public = int(event["target_public_id"])
    for binding in event["state_bindings"]:
        public = int(binding["public_id"])
        uid = str(binding["initial_candidate_uid"])
        candidate = by_uid.get(uid)
        if candidate is None:
            raise ValueError(f"event initial candidate is absent from tape: {uid}")
        feature = anchor if public == target_public else unit(candidate["feature"], f"initial state {public}")
        box = event["target_box_xyxy"] if public == target_public else candidate["box_xyxy"]
        state = IdentityState(
            int(binding["association_state_id"]),
            feature,
            np.asarray(box, dtype=np.float64),
            0,
            int(candidate["native_tid"]),
            native_scope=None,
        )
        # The base scorer is an existing solver-local scorer.  The explicit
        # axes below are supplied separately to the public assignment wrapper.
        state.association_state_id = int(binding["association_state_id"])
        state.public_id = public
        states[public] = state
    if target_public not in states:
        raise ValueError("human target is absent from public axis")
    return states


def update_base_states(
    *,
    states: Mapping[int, IdentityState],
    solver: Mapping[str, Any],
    rows: list[dict[str, Any]],
    frame: int,
    target_public: int,
    event_frame: int,
) -> dict[str, Any]:
    by_uid = {str(row["candidate_uid"]): row for row in rows}
    assignments = public_assignment_map(solver)
    machine_updates: list[int] = []
    none_publics: list[int] = []
    for public, state in states.items():
        uid = assignments.get(int(public))
        if uid is None:
            none_publics.append(int(public))
            if state.state == IdentityState.ACTIVE:
                state.mark_lost(int(frame))
            else:
                state.advance_lost()
            continue
        candidate = by_uid.get(uid)
        if candidate is None:
            raise ValueError(f"base solver assigned unknown candidate {uid}")
        # The simulated human event is the sole target initialization.  The
        # same-frame base decision updates motion/native provenance but must
        # not overwrite the target's human prototype.
        preserve_target_anchor = int(frame) == int(event_frame) and int(public) == int(target_public)
        state.update_machine(
            np.asarray(candidate["feature"], dtype=np.float32),
            np.asarray(candidate["box_xyxy"], dtype=np.float64),
            int(frame),
            int(candidate["native_tid"]),
            BASE_EMA,
            update_prototype=not preserve_target_anchor,
            native_scope=None,
        )
        machine_updates.append(int(public))
    return {
        "frame": int(frame),
        "assigned_public_ids": sorted(int(value) for value in assignments if assignments[value] is not None),
        "none_public_ids": sorted(none_publics),
        "machine_update_public_ids": sorted(machine_updates),
        "target_anchor_preserved": bool(int(frame) == int(event_frame)),
        "runtime_future_gt_used": False,
    }


def build_sequence(*, asset_root: Path, sequence: str, event: Mapping[str, Any]) -> dict[str, Any]:
    tape = load_candidate_frames(asset_root, sequence)
    bindings = [dict(item) for item in event["state_bindings"]]
    states = initialize_states(event, tape[0][1])
    ordered_states = assignable_states(bindings, states)
    out_root = asset_root / "base_scores" / sequence
    out_root.mkdir(parents=True, exist_ok=True)
    final_tape = out_root / "base_scores.jsonl.zst"
    final_index = out_root / "index.json"
    final_done = out_root / "done.json"
    if any(path.exists() for path in (final_tape, final_index, final_done)):
        raise FileExistsError(f"refusing to overwrite existing base tape: {out_root}")

    rows_out: list[dict[str, Any]] = []
    total_candidates = 0
    assignment_rows = 0
    state_updates = 0
    for frame_row, candidate_rows in tape:
        frame = int(frame_row["frame"])
        score_audit: dict[str, Any] = {}
        matrix = np.asarray(
            score_matrix_pairwise(
                ordered_states,
                candidate_rows,
                frame,
                model=None,
                reid_weights=dict(BASE_REID_WEIGHTS),
                positive_bonus=BASE_POSITIVE_BONUS,
                native_bonus=BASE_NATIVE_BONUS,
                score_audit=score_audit,
            ),
            dtype=np.float64,
        )
        if matrix.shape != (len(candidate_rows), len(ordered_states)) or not np.isfinite(matrix).all():
            raise ValueError(f"{sequence}:{frame}: invalid current base score matrix {matrix.shape}")
        solver = solve_effect_assignment(
            candidate_rows=[serializable_candidate(row) for row in candidate_rows],
            persistent_states=ordered_states,
            fused_state_candidate_scores=matrix.T,
            source_run_id=f"n72r20r1:{sequence}:base:{frame}",
            session_id=f"n72r20r1:{sequence}",
            none_score=NONE_SCORE,
        )
        state_digest_before = hashlib.sha256(
            json.dumps(
                {str(public): state.to_dict() for public, state in sorted(states.items())},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        update = update_base_states(
            states=states,
            solver=solver,
            rows=candidate_rows,
            frame=frame,
            target_public=int(event["target_public_id"]),
            event_frame=int(event["event_frame"]),
        )
        state_digest_after = hashlib.sha256(
            json.dumps(
                {str(public): state.to_dict() for public, state in sorted(states.items())},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        serialized_rows = [serializable_candidate(row) for row in candidate_rows]
        row = {
            "stage": "N72R20R1",
            "sequence": sequence,
            "split": "train",
            "event_frame": int(event["event_frame"]),
            "frame": frame,
            "frame_horizon": frame - int(event["event_frame"]),
            "candidate_rows": serialized_rows,
            "candidate_uid_axis": [str(row["candidate_uid"]) for row in candidate_rows],
            "candidate_count": len(candidate_rows),
            "association_state_axis": [int(item["association_state_id"]) for item in bindings],
            "public_id_axis": [int(item["public_id"]) for item in bindings],
            "base_score_matrix": matrix.tolist(),
            "base_score_matrix_shape": [int(value) for value in matrix.shape],
            "base_score_matrix_sha256": matrix_sha256(matrix),
            "base_assignment": solver,
            "explicit_none_score": float(NONE_SCORE),
            "base_scorer": {
                "implementation": "sam3_intermot.association.online_associator.score_matrix_pairwise",
                "model": None,
                "reid_weights": dict(BASE_REID_WEIGHTS),
                "native_bonus": float(BASE_NATIVE_BONUS),
                "positive_bonus": float(BASE_POSITIVE_BONUS),
                "ema": float(BASE_EMA),
                "state_update": "exact-public-assignment-output-then-IdentityState.update_machine",
            },
            "state_update": update,
            "state_digest_before": state_digest_before,
            "state_digest_after": state_digest_after,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
        }
        rows_out.append(row)
        total_candidates += len(candidate_rows)
        assignment_rows += len(solver.get("assignment_rows", []))
        state_updates += len(update.get("machine_update_public_ids", []))

    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows_out).encode("utf-8")
    compressed = subprocess.run(
        ["zstd", "-q", "-T0", "-c"],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    final_tape.write_bytes(compressed.stdout)
    index = {
        "stage": "N72R20R1",
        "sequence": sequence,
        "split": "train",
        "event_frame": int(event["event_frame"]),
        "frame_count": len(rows_out),
        "candidate_count": total_candidates,
        "assignment_row_count": assignment_rows,
        "base_state_update_count": state_updates,
        "candidate_uid_axis_source": "shared_candidate_tape_per_frame",
        "association_state_axis": [int(item["association_state_id"]) for item in bindings],
        "public_id_axis": [int(item["public_id"]) for item in bindings],
        "base_score_orientation": "candidate_x_association_state",
        "base_scores": str(final_tape),
        "base_scores_sha256": sha256(final_tape),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
    }
    final_index.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    done = {
        "status": "PASS_N72R20R1_BASE_SCORE_TAPE_SEQUENCE",
        "sequence": sequence,
        "index": str(final_index),
        "index_sha256": sha256(final_index),
        "base_scores_sha256": sha256(final_tape),
        "frame_count": len(rows_out),
        "candidate_count": total_candidates,
        "runtime_future_gt_used": False,
        "gt_opened": False,
        "historical_n72r15_used": False,
    }
    final_done.write_text(json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return done


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    event_path = (args.events or (asset_root / "interaction_events.json")).resolve()
    event_payload = read_json(event_path)
    events = {str(item["sequence"]): item for item in event_payload.get("events", [])}
    results = []
    for sequence in args.sequences:
        if str(sequence) not in events:
            raise ValueError(f"missing interaction event for {sequence}")
        results.append(build_sequence(asset_root=asset_root, sequence=str(sequence), event=events[str(sequence)]))
    print(json.dumps({"status": "PASS_N72R20R1_BASE_SCORE_TAPE", "sequences": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
