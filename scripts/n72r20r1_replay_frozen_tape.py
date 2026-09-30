#!/usr/bin/env python3
"""Replay one sealed N72R20R1 candidate/base-score lineage.

E0, E1 and E2 all consume the same candidate rows and the same frozen base
matrix from the fresh tape.  E1 uses the existing trusted relative-state edge;
E2 uses the frozen N72R18 learned identity edge.  Neither branch owns a
solver or public authority, and both update memory only after assignment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from sam3_intermot.association.effect_assignment import solve_effect_assignment
from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank
from sam3_intermot.association.learned_identity_state_edge import build_learned_identity_state_edge_matrix
from sam3_intermot.association.relative_persistent_state_edge import (
    build_relative_persistent_state_edge_matrix,
    fuse_row_max_preserving,
)
from sam3_intermot.association.trusted_persistent_public_state import TrustedPersistentPublicAssociationBank
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames, read_json, read_zstd_jsonl, unit


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R1_assets")
CHECKPOINT = ROOT / "outputs/N72R18/checkpoints/identity_memory_gru.pt"
CHECKPOINT_SHA = "94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2"
ENCODER_SHA = "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154"
VARIANTS = (
    "FRESH_BASELINE_B0",
    "FRESH_TRUSTED_RELATIVE_STATE",
    "FRESH_LEARNED_IDENTITY_TRUSTED",
)
NONE_SCORE = 0.0


@dataclass(frozen=True)
class ExplicitState:
    association_state_id: int
    public_id: int


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


def write_zstd_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode("utf-8")
    completed = subprocess.run(
        ["zstd", "-q", "-T0", "-c"],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    path.write_bytes(completed.stdout)


def public_map(solver: Mapping[str, Any]) -> dict[int, str | None]:
    output: dict[int, str | None] = {}
    for item in solver.get("public_assignments", []):
        uid = item.get("candidate_uid")
        output[int(item["public_id"])] = None if uid in (None, "", "None") else str(uid)
    return output


def state_digest(bank: Any) -> str:
    if bank is None:
        return "NONE"
    return str(bank.digest())


def serializable_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"feature", "feat", "box"}
    }


def validate_common_row(
    *,
    base_row: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    sequence: str,
) -> np.ndarray:
    candidate_uids = [str(row["candidate_uid"]) for row in candidate_rows]
    if candidate_uids != [str(value) for value in base_row.get("candidate_uid_axis", [])]:
        raise ValueError(f"{sequence}:{base_row.get('frame')}: candidate tape/base tape UID mismatch")
    if base_row.get("runtime_future_gt_used") is not False or base_row.get("runtime_gt_read") is not False:
        raise ValueError(f"{sequence}:{base_row.get('frame')}: base tape runtime GT flag violation")
    matrix = np.asarray(base_row.get("base_score_matrix", []), dtype=np.float64)
    expected = (len(candidate_rows), len(base_row.get("association_state_axis", [])))
    if matrix.shape != expected or not np.isfinite(matrix).all():
        raise ValueError(f"{sequence}:{base_row.get('frame')}: base matrix shape/finite failure {matrix.shape} != {expected}")
    if matrix_sha256(matrix) != str(base_row.get("base_score_matrix_sha256")):
        raise ValueError(f"{sequence}:{base_row.get('frame')}: base matrix digest mismatch")
    return matrix


def explicit_states(event: Mapping[str, Any]) -> list[ExplicitState]:
    return [
        ExplicitState(int(item["association_state_id"]), int(item["public_id"]))
        for item in event["state_bindings"]
    ]


def assignment_delta(base_solver: Mapping[str, Any], treatment_solver: Mapping[str, Any]) -> dict[str, Any]:
    base = public_map(base_solver)
    treatment = public_map(treatment_solver)
    changed = sorted(public for public in set(base) | set(treatment) if base.get(public) != treatment.get(public))
    none_changes = sorted(
        public
        for public in changed
        if (base.get(public) is None) != (treatment.get(public) is None)
    )
    return {
        "changed_public_ids": changed,
        "changed_public_count": len(changed),
        "candidate_none_changes": none_changes,
        "candidate_none_change_count": len(none_changes),
        "assignment_changed": bool(changed),
    }


def initialize_variant(
    variant: str,
    event: Mapping[str, Any],
    first_candidates: Sequence[Mapping[str, Any]],
    first_base_solver: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    if variant == "FRESH_BASELINE_B0":
        return None, {"event_frame_memory_read": False, "initialization": None, "runtime_future_gt_used": False}
    if variant == "FRESH_TRUSTED_RELATIVE_STATE":
        bank = TrustedPersistentPublicAssociationBank()
        target_uid = str(event["target_candidate_uid"])
        target_candidate = next(row for row in first_candidates if str(row["candidate_uid"]) == target_uid)
        initialization = bank.initialize_from_event_frame(
            event_frame=int(event["event_frame"]),
            candidate_rows=first_candidates,
            solver=first_base_solver,
            association_state_axis=[int(value) for value in event["association_state_axis"]],
            public_id_axis=[int(value) for value in event["public_id_axis"]],
            target_public_id=int(event["target_public_id"]),
            target_anchor=unit(event["human_anchor"], "human anchor"),
            target_box=event["target_box_xyxy"],
            target_candidate=target_candidate,
            event_id=f"{event['sequence']}:N72R20R1:human-init",
        )
        return bank, {"event_frame_memory_read": False, "initialization": initialization, "runtime_future_gt_used": False}
    if variant == "FRESH_LEARNED_IDENTITY_TRUSTED":
        bank = LearnedIdentityMemoryBank.from_checkpoint(
            CHECKPOINT,
            encoder_sha256=ENCODER_SHA,
            expected_encoder_sha256=ENCODER_SHA,
            expected_checkpoint_sha256=CHECKPOINT_SHA,
        )
        initialization = bank.initialize_public_identity(
            public_id=int(event["target_public_id"]),
            association_state_id=int(event["target_association_state_id"]),
            frame=int(event["event_frame"]),
            human_anchor=unit(event["human_anchor"], "human anchor"),
            interaction_source="simulated_from_gt",
        )
        return bank, {"event_frame_memory_read": False, "initialization": initialization, "runtime_future_gt_used": False}
    raise ValueError(f"unknown variant {variant}")


def replay_sequence(*, asset_root: Path, output_root: Path, sequence: str, event: Mapping[str, Any]) -> dict[str, Any]:
    candidate_frames = load_candidate_frames(asset_root, sequence)
    base_root = asset_root / "base_scores" / sequence
    base_rows = read_zstd_jsonl(base_root / "base_scores.jsonl.zst")
    if len(candidate_frames) != len(base_rows):
        raise ValueError(f"{sequence}: candidate/base frame count mismatch")
    base_index = read_json(base_root / "index.json")
    if base_index.get("base_scores_sha256") != sha256(base_root / "base_scores.jsonl.zst"):
        raise ValueError(f"{sequence}: base tape SHA mismatch")
    bindings = [dict(item) for item in event["state_bindings"]]
    state_axis = [int(item["association_state_id"]) for item in bindings]
    public_axis = [int(item["public_id"]) for item in bindings]
    if state_axis != [int(value) for value in event["association_state_axis"]] or public_axis != [int(value) for value in event["public_id_axis"]]:
        raise ValueError(f"{sequence}: event axes mismatch")

    outputs: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    audits: dict[str, dict[str, Any]] = {
        variant: {
            "variant": variant,
            "sequence": sequence,
            "frames": 0,
            "candidate_count": 0,
            "score_changed_frames": 0,
            "score_changed_cells": 0,
            "solver_changed_frames": 0,
            "changed_public_id_count": 0,
            "candidate_none_change_count": 0,
            "row_max_preservation_failures": 0,
            "memory_read_frames": 0,
            "memory_write_count": 0,
            "consensus_public_count": 0,
            "disagreement_public_count": 0,
            "runtime_future_gt_used": False,
        }
        for variant in VARIANTS
    }
    memory_writes: list[dict[str, Any]] = []
    banks: dict[str, Any] = {}
    first_full_candidates = candidate_frames[0][1]
    first_base_solver = read_zstd_jsonl(base_root / "base_scores.jsonl.zst")[0]["base_assignment"]
    for variant in VARIANTS:
        banks[variant], init = initialize_variant(variant, event, first_full_candidates, first_base_solver)
        banks[f"{variant}:init"] = init

    for (candidate_frame_row, full_candidates), base_row in zip(candidate_frames, base_rows):
        frame = int(base_row["frame"])
        matrix = validate_common_row(base_row=base_row, candidate_rows=full_candidates, sequence=sequence)
        base_solver = dict(base_row["base_assignment"])
        if int(base_solver.get("none_score", NONE_SCORE)) != int(NONE_SCORE):
            raise ValueError(f"{sequence}:{frame}: NONE score changed")
        if [int(value) for value in base_solver.get("association_state_axis", [])] != state_axis:
            raise ValueError(f"{sequence}:{frame}: solver state axis changed")
        frame_init = frame == int(event["event_frame"])
        for variant in VARIANTS:
            bank = banks[variant]
            if variant == "FRESH_BASELINE_B0" or frame_init:
                treatment_solver = base_solver
                edge_audit = {
                    "score_changed": False,
                    "score_changed_cells": 0,
                    "score_changed_rows": 0,
                    "row_max_preserved": True,
                    "learned_signal_abs_max": 0.0,
                    "learned_signal_nonzero_cells": 0,
                }
                before = state_digest(bank)
                if frame_init:
                    init = banks[f"{variant}:init"]
                    state_update = dict(init.get("initialization") or {})
                    state_update["runtime_future_gt_used"] = False
                else:
                    state_update = {
                        "frame": frame,
                        "updates": [],
                        "runtime_future_gt_used": False,
                    }
                after = state_digest(bank)
            elif variant == "FRESH_TRUSTED_RELATIVE_STATE":
                states = bank.states_for_pairs(list(zip(state_axis, public_axis)))
                before = state_digest(bank)
                edge = build_relative_persistent_state_edge_matrix(
                    bank=bank,
                    candidate_rows=full_candidates,
                    public_id_axis=public_axis,
                    frame=frame,
                    target_public_id=int(event["target_public_id"]),
                    state_scope="TRUSTED_GLOBAL",
                )
                fusion = fuse_row_max_preserving(matrix, edge["relative_delta"], state_edge_scale=float(edge["state_edge_scale"]))
                fused = np.asarray(fusion["fused"], dtype=np.float64)
                treatment_solver = solve_effect_assignment(
                    candidate_rows=[serializable_row(row) for row in full_candidates],
                    persistent_states=states,
                    fused_state_candidate_scores=fused.T,
                    source_run_id=f"n72r20r1:{sequence}:{variant}:{frame}",
                    session_id=f"n72r20r1:{sequence}",
                    none_score=NONE_SCORE,
                )
                state_update = bank.update_from_consensus(
                    frame=frame,
                    candidate_rows=full_candidates,
                    base_solver=base_solver,
                    treatment_solver=treatment_solver,
                    target_public_id=int(event["target_public_id"]),
                    none_score=NONE_SCORE,
                )
                after = state_digest(bank)
                edge_audit = {
                    "score_changed": bool(np.any(np.abs(fused - matrix) > 1.0e-12)),
                    "score_changed_cells": int(np.sum(np.abs(fused - matrix) > 1.0e-12)),
                    "score_changed_rows": int(np.sum(np.any(np.abs(fused - matrix) > 1.0e-12, axis=1))) if fused.size else 0,
                    "row_max_preserved": bool(fusion["row_max_preserved"]),
                    "learned_signal_abs_max": 0.0,
                    "learned_signal_nonzero_cells": 0,
                }
            elif variant == "FRESH_LEARNED_IDENTITY_TRUSTED":
                before = state_digest(bank)
                edge = build_learned_identity_state_edge_matrix(
                    bank=bank,
                    candidate_rows=full_candidates,
                    public_id_axis=public_axis,
                    frame=frame,
                    state_scope="HUMAN_INITIALIZED_ONLY",
                    base_candidate_public_scores=matrix,
                )
                fused = np.asarray(edge["fused_candidate_public_scores"], dtype=np.float64)
                treatment_solver = solve_effect_assignment(
                    candidate_rows=[serializable_row(row) for row in full_candidates],
                    persistent_states=explicit_states(event),
                    fused_state_candidate_scores=fused.T,
                    source_run_id=f"n72r20r1:{sequence}:{variant}:{frame}",
                    session_id=f"n72r20r1:{sequence}",
                    none_score=NONE_SCORE,
                )
                state_update = bank.update_from_consensus(
                    frame=frame,
                    candidate_rows=full_candidates,
                    base_solver=base_solver,
                    treatment_solver=treatment_solver,
                )
                after = state_digest(bank)
                learned_signal = np.asarray(edge["relative_delta"], dtype=np.float64)
                edge_audit = {
                    "score_changed": bool(np.any(np.abs(fused - matrix) > 1.0e-12)),
                    "score_changed_cells": int(np.sum(np.abs(fused - matrix) > 1.0e-12)),
                    "score_changed_rows": int(np.sum(np.any(np.abs(fused - matrix) > 1.0e-12, axis=1))) if fused.size else 0,
                    "row_max_preserved": bool(edge["row_max_preserved"]),
                    "learned_signal_abs_max": float(np.max(np.abs(learned_signal))) if learned_signal.size else 0.0,
                    "learned_signal_nonzero_cells": int(np.sum(np.abs(learned_signal) > 1.0e-12)),
                }
            else:
                raise AssertionError(variant)

            delta = assignment_delta(base_solver, treatment_solver)
            update_items = list(state_update.get("updates", []))
            for update in update_items:
                memory_writes.append({"sequence": sequence, "variant": variant, **dict(update)})
            audit = audits[variant]
            audit["frames"] += 1
            audit["candidate_count"] += len(full_candidates)
            audit["score_changed_frames"] += int(edge_audit["score_changed"])
            audit["score_changed_cells"] += int(edge_audit["score_changed_cells"])
            audit["solver_changed_frames"] += int(delta["assignment_changed"])
            audit["changed_public_id_count"] += int(delta["changed_public_count"])
            audit["candidate_none_change_count"] += int(delta["candidate_none_change_count"])
            audit["row_max_preservation_failures"] += int(not edge_audit["row_max_preserved"])
            audit["memory_read_frames"] += int(variant != "FRESH_BASELINE_B0" and not frame_init)
            audit["memory_write_count"] += len(update_items)
            audit["consensus_public_count"] += int(state_update.get("consensus_public_count", 0))
            audit["disagreement_public_count"] += int(state_update.get("disagreement_public_count", 0))
            outputs[variant].append(
                {
                    "stage": "N72R20R1",
                    "sequence": sequence,
                    "variant": variant,
                    "event_frame": int(event["event_frame"]),
                    "frame": frame,
                    "frame_horizon": frame - int(event["event_frame"]),
                    "candidate_uid_axis": [str(row["candidate_uid"]) for row in full_candidates],
                    "association_state_axis": state_axis,
                    "public_id_axis": public_axis,
                    "base_score_matrix_sha256": matrix_sha256(matrix),
                    "base_solver": base_solver,
                    "treatment_solver": treatment_solver,
                    "assignment_delta": delta,
                    "edge_audit": edge_audit,
                    "memory_read": bool(variant != "FRESH_BASELINE_B0" and not frame_init),
                    "memory_write": bool(update_items),
                    "state_update": state_update,
                    "state_digest_before": before,
                    "state_digest_after": after,
                    "runtime_future_gt_used": False,
                    "runtime_gt_read": False,
                    "posthoc_gt_used": False,
                }
            )

    for variant in VARIANTS:
        variant_root = output_root / sequence / variant
        write_zstd_jsonl(variant_root / "runtime_frames.jsonl.zst", outputs[variant])
        manifest = dict(audits[variant])
        manifest.update(
            {
                "status": "PASS_N72R20R1_REPLAY_VARIANT",
                "sequence": sequence,
                "variant": variant,
                "runtime_frames": str(variant_root / "runtime_frames.jsonl.zst"),
                "runtime_frames_sha256": sha256(variant_root / "runtime_frames.jsonl.zst"),
                "candidate_tape_shared": True,
                "base_score_tape_shared": True,
                "exact_solver_shared": True,
                "public_authority_shared": True,
                "runtime_future_gt_used": False,
            }
        )
        (variant_root / "done.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "sequence": sequence,
        "variants": audits,
        "runtime_future_gt_used": False,
        "memory_writes": memory_writes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=ASSET_ROOT)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--sequences", nargs="+", default=["dancetrack0001", "dancetrack0002"])
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/N72R20R1/causal_replay/train_smoke")
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    event_path = (args.events or (asset_root / "interaction_events.json")).resolve()
    event_payload = read_json(event_path)
    events = {str(item["sequence"]): item for item in event_payload.get("events", [])}
    results = []
    all_writes: list[dict[str, Any]] = []
    for sequence in args.sequences:
        if str(sequence) not in events:
            raise ValueError(f"missing event for {sequence}")
        result = replay_sequence(asset_root=asset_root, output_root=args.output_root.resolve(), sequence=str(sequence), event=events[str(sequence)])
        results.append({"sequence": result["sequence"], "variants": result["variants"]})
        all_writes.extend(result["memory_writes"])
    write_path = ROOT / "outputs/N72R20R1/memory_write_audit.jsonl"
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text("".join(json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in all_writes), encoding="utf-8")
    payload = {
        "status": "PASS_N72R20R1_TRAIN_SMOKE_CAUSAL_REPLAY",
        "stage": "N72R20R1",
        "sequences": results,
        "variants": list(VARIANTS),
        "memory_write_audit": str(write_path),
        "candidate_tape_shared": True,
        "base_score_tape_shared": True,
        "runtime_future_gt_used": False,
        "historical_n72r15_used": False,
    }
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    (args.output_root.parent / "train_smoke_replay_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
