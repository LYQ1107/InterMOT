#!/usr/bin/env python3
"""Build the frozen base-score tape on the fresh R2 candidate streams.

The implementation is the existing R1 scorer/solver/state-update path.  This
script only gives the new artifact an R2 lineage label and a fresh source-run
namespace; it does not add a matcher or alter assignment semantics.  It never
opens DanceTrack GT.
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
from sam3_intermot.association.online_associator import score_matrix_pairwise
from scripts.n72r20r1_build_base_score_tape import (
    BASE_EMA,
    BASE_NATIVE_BONUS,
    BASE_POSITIVE_BONUS,
    BASE_REID_WEIGHTS,
    NONE_SCORE,
    assignable_states,
    initialize_states,
    load_candidate_frames,
    matrix_sha256,
    read_json,
    serializable_candidate,
    sha256,
    update_base_states,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R2_assets")
DEV_SEQUENCES = (
    "dancetrack0001",
    "dancetrack0002",
    "dancetrack0023",
    "dancetrack0024",
    "dancetrack0039",
    "dancetrack0057",
    "dancetrack0062",
    "dancetrack0072",
)


def state_digest(states: Mapping[int, Any]) -> str:
    payload = json.dumps(
        {str(public): state.to_dict() for public, state in sorted(states.items())},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        raise FileExistsError(f"refusing to overwrite existing R2 base tape: {out_root}")

    rows_out: list[dict[str, Any]] = []
    total_candidates = 0
    assignment_rows = 0
    state_updates = 0
    for frame_payload, candidate_rows in tape:
        frame = int(frame_payload["frame"])
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
            raise ValueError(f"{sequence}:{frame}: invalid base score matrix {matrix.shape}")
        solver = solve_effect_assignment(
            candidate_rows=[serializable_candidate(row) for row in candidate_rows],
            persistent_states=ordered_states,
            fused_state_candidate_scores=matrix.T,
            source_run_id=f"n72r20r2:{sequence}:base:{frame}",
            session_id=f"n72r20r2:{sequence}",
            none_score=NONE_SCORE,
        )
        before = state_digest(states)
        update = update_base_states(
            states=states,
            solver=solver,
            rows=candidate_rows,
            frame=frame,
            target_public=int(event["target_public_id"]),
            event_frame=int(event["event_frame"]),
        )
        after = state_digest(states)
        serialized_rows = [serializable_candidate(row) for row in candidate_rows]
        rows_out.append(
            {
                "stage": "N72R20R2",
                "source_stage": "N72R20R1",
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
                "state_digest_before": before,
                "state_digest_after": after,
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
                "posthoc_gt_used": False,
            }
        )
        total_candidates += len(candidate_rows)
        assignment_rows += len(solver.get("assignment_rows", []))
        state_updates += len(update.get("machine_update_public_ids", []))

    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows_out).encode("utf-8")
    compressed = subprocess.run(["zstd", "-q", "-T0", "-c"], input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    final_tape.write_bytes(compressed.stdout)
    index = {
        "stage": "N72R20R2",
        "source_stage": "N72R20R1",
        "sequence": sequence,
        "split": "train",
        "event_frame": int(event["event_frame"]),
        "frame_count": len(rows_out),
        "candidate_count": total_candidates,
        "assignment_row_count": assignment_rows,
        "base_state_update_count": state_updates,
        "candidate_uid_axis_source": "shared_r2_candidate_tape_per_frame",
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
        "status": "PASS_N72R20R2_BASE_SCORE_TAPE_SEQUENCE",
        "sequence": sequence,
        "index": str(final_index),
        "index_sha256": sha256(final_index),
        "base_scores_sha256": sha256(final_tape),
        "frame_count": len(rows_out),
        "candidate_count": total_candidates,
        "runtime_future_gt_used": False,
        "gt_opened": False,
    }
    final_done.write_text(json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return done


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--sequences", nargs="+", default=list(DEV_SEQUENCES))
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    event_payload = read_json((args.events or (asset_root / "interaction_events.json")).resolve())
    events = {str(item["sequence"]): item for item in event_payload.get("events", [])}
    results = []
    for sequence in args.sequences:
        if str(sequence) not in events:
            raise ValueError(f"missing interaction event for {sequence}")
        results.append(build_sequence(asset_root=asset_root, sequence=str(sequence), event=events[str(sequence)]))
    output = ROOT / "outputs/N72R20R2/base_score_dev_run.json"
    output.write_text(json.dumps({"stage": "N72R20R2", "status": "PASS_N72R20R2_BASE_SCORE_TAPE", "sequences": results, "runtime_future_gt_used": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_N72R20R2_BASE_SCORE_TAPE", "sequences": len(results), "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
