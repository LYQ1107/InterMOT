#!/usr/bin/env python3
"""N72R14 CPU interface audit and toy causal smoke.

This audit does not open dataset GT and does not run a scientific effect
experiment.  It verifies the frozen candidate/public axes, exact solver
control, causal human-memory visibility, and the new candidate-by-public state
edge on a small non-scientific fixture before the formal runner is allowed.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.appearance_memory import AppearanceMemory  # noqa: E402
from sam3_intermot.association.effect_assignment import solve_effect_assignment  # noqa: E402
from sam3_intermot.association.persistent_public_state import (  # noqa: E402
    PersistentPublicAssociationBank,
)
from sam3_intermot.association.persistent_state_edge import (  # noqa: E402
    STATE_EDGE_SCALE,
    build_persistent_state_edge_matrix,
)
from sam3_intermot.reacquisition.target_candidate_pool import build_candidate_pool  # noqa: E402
from scripts import n72r11_on_demand_replay as replay  # noqa: E402


PROTOCOL = ROOT / "outputs/N72R9/protocol.json"
OUTPUT_ROOT = ROOT / "outputs/N72R14"
AUDIT_PATH = OUTPUT_ROOT / "interface_audit.json"
STATUS_PATH = OUTPUT_ROOT / "stage_00_status.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            import os

            os.fsync(handle.fileno())
        import os

        os.replace(temporary, path)
    finally:
        import os

        if os.path.exists(temporary):
            os.unlink(temporary)


def _events() -> list[dict[str, Any]]:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    events = payload.get("source_event_selection", {}).get("events", [])
    if not isinstance(events, list) or not events:
        raise RuntimeError("frozen N72R9 protocol has no event list")
    return [dict(event) for event in events]


def _unit(index: int, dim: int = 512) -> np.ndarray:
    result = np.zeros(dim, dtype=np.float32)
    result[int(index) % dim] = 1.0
    return result


def _toy_checks() -> dict[str, Any]:
    """Focused non-scientific checks for the new bank and edge contract."""

    memory = AppearanceMemory()
    anchor = _unit(0)
    memory.update_from_human(1003, 10, anchor, write_event_id="toy")
    hidden_at_event = memory.score(1003, anchor, 10) == 0.0
    visible_next = memory.score(1003, anchor, 11) > 0.0
    bank = PersistentPublicAssociationBank(appearance_memory=memory)
    # Deliberately non-equal numeric axes prove that public authority is not
    # inferred from a solver-local state ID.
    bank.ensure_state(7, 1003, last_box=[0, 0, 10, 10], last_seen_frame=10, previous_raw_sam_id=4, previous_native_scope="s")
    bank.ensure_state(8, 1004, last_box=[20, 0, 30, 10], last_seen_frame=10, previous_raw_sam_id=5, previous_native_scope="s")
    candidates = [
        {
            "candidate_uid": "toy-target",
            "box_xyxy": [0, 0, 10, 10],
            "feature": anchor.tolist(),
            "official_raw_sam_id": 4,
            "native_scope": "s",
        },
        {
            "candidate_uid": "toy-other",
            "box_xyxy": [20, 0, 30, 10],
            "feature": _unit(1).tolist(),
            "official_raw_sam_id": 5,
            "native_scope": "s",
        },
    ]
    edge = build_persistent_state_edge_matrix(
        bank=bank,
        candidate_rows=candidates,
        public_id_axis=[1003, 1004],
        frame=11,
    )
    delta = np.asarray(edge["delta"], dtype=float)
    target_only = np.zeros_like(delta)
    target_only[:, 0] = delta[:, 0]
    non_target_nonzero = bool(np.any(np.abs(delta[:, 1]) > 1.0e-9))
    state_axis = [7, 8]
    public_axis = [1003, 1004]
    exact = solve_effect_assignment(
        candidate_rows=candidates,
        persistent_states=bank.states_for_pairs(list(zip(state_axis, public_axis))),
        fused_state_candidate_scores=np.zeros((2, 2), dtype=float),
        source_run_id="n72r14:toy",
        session_id="n72r14:toy",
        none_score=0.0,
    )
    return {
        "fixture": "toy_non_scientific",
        "event_frame_memory_read_false": bool(hidden_at_event),
        "first_memory_visible_event_plus_one": bool(visible_next),
        "public_state_axis_explicit_and_distinct": bool(state_axis != public_axis and public_axis == [1003, 1004]),
        "edge_shape": edge["shape"],
        "edge_scale": float(STATE_EDGE_SCALE),
        "edge_finite": bool(np.isfinite(delta).all()),
        "edge_has_non_target_edges": non_target_nonzero,
        "target_only_mask_is_not_global": bool(np.any(np.abs(delta - target_only) > 1.0e-9)),
        "exact_solver_runtime_future_gt_used": exact.get("runtime_future_gt_used") is False,
        "all_pass": bool(
            hidden_at_event
            and visible_next
            and non_target_nonzero
            and np.isfinite(delta).all()
            and exact.get("runtime_future_gt_used") is False
        ),
    }


def _semantic_assignment(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(item.get("candidate_uid")): item.get("solver_public_id", item.get("public_id"))
        for item in row.get("candidate_rows", [])
    }


def _e0_equivalence(event: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Compare one regenerated frozen B0 row to the corrected R5R1 E0."""

    frame = int(inputs["event_frame"]) + 1
    c0 = inputs["rows"]["c0_source"][frame]
    pool, _ = build_candidate_pool(
        list(c0.get("candidate_rows", [])),
        (),
        sequence=str(inputs["sequence"]),
        frame=frame,
        include_target_session=False,
        # E0 must use the same pre-solver positive-geometry policy as the
        # corrected N72R11R5R1 control; post-hoc filtering is not equivalent.
        require_positive_geometry=True,
    )
    pairs, vectors = replay.legacy._base_vectors(inputs, frame, pool)
    matrix = np.stack([vectors[str(item["candidate_uid"])] for item in pool], axis=0)
    solver = solve_effect_assignment(
        candidate_rows=pool,
        persistent_states=replay._state_objects(pairs),
        fused_state_candidate_scores=matrix.T,
        source_run_id=f"n72r14:e0:{inputs['event_id']}:{frame}",
        session_id=f"n72r14:e0:{inputs['event_id']}",
        none_score=0.0,
    )
    regenerated = replay._solver_rows(pool, solver)
    frozen = inputs["baseline_rows"][1]
    frozen_audit = frozen.get("score_audit", {})
    regenerated_assignment = _semantic_assignment({"candidate_rows": regenerated})
    frozen_assignment = _semantic_assignment(frozen)
    matrix_ref = np.asarray(frozen_audit.get("fused_score_matrix", []), dtype=float)
    matrix_equal = bool(matrix_ref.shape == matrix.shape and np.array_equal(matrix_ref, matrix))
    r5r1_path = ROOT / "outputs/N72R11R5R1/formal_e1b" / str(event["event_id"]) / "E0_BASELINE_B0/runtime_frames.jsonl"
    r5r1_exists = r5r1_path.is_file()
    r5r1_assignment_equal = None
    r5r1_frame_axis_equal = None
    if r5r1_exists:
        lines = [json.loads(line) for line in r5r1_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        r5r1_future = next((item for item in lines if int(item.get("frame", -1)) == frame), None)
        r5r1_assignment_equal = r5r1_future is not None and _semantic_assignment(r5r1_future) == regenerated_assignment
        r5r1_frame_axis_equal = [int(item.get("frame", -1)) for item in lines] == list(
            range(int(inputs["event_frame"]), int(inputs["event_frame"]) + 101)
        )
    return {
        "event_id": str(event["event_id"]),
        "checked_frame": frame,
        "candidate_axis_equal_to_frozen": [str(item["candidate_uid"]) for item in regenerated]
        == [str(item["candidate_uid"]) for item in frozen.get("candidate_rows", [])],
        "public_axis_equal_to_frozen": solver.get("public_id_axis") == frozen_audit.get("public_id_axis"),
        "assignment_equal_to_frozen": regenerated_assignment == frozen_assignment,
        "score_matrix_exact_equal_to_frozen": matrix_equal,
        "r5r1_e0_path": str(r5r1_path),
        "r5r1_e0_exists": r5r1_exists,
        "r5r1_assignment_equal": r5r1_assignment_equal,
        "r5r1_frame_axis_equal": r5r1_frame_axis_equal,
        "pass": bool(
            regenerated_assignment == frozen_assignment
            and matrix_equal
            and (not r5r1_exists or bool(r5r1_assignment_equal))
        ),
    }


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    events = _events()
    event_checks: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for event in events:
        try:
            inputs = replay._load_inputs(event, horizon=100)
            event_frame = int(inputs["event_frame"])
            for name, rows in inputs["rows"].items():
                if [int(rows[frame]["frame"]) for frame in sorted(rows)] != list(
                    range(event_frame, event_frame + 101)
                ):
                    raise RuntimeError(f"{event['event_id']} {name} frame axis mismatch")
                if any(row.get("runtime_future_gt_used") is not False for row in rows.values()):
                    raise RuntimeError(f"{event['event_id']} {name} runtime GT flag")
            c0_event = inputs["rows"]["c0_source"][event_frame]
            event_checks.append(
                {
                    "event_id": str(event["event_id"]),
                    "sequence": str(event["sequence"]),
                    "event_frame": event_frame,
                    "action_type": str(event["action_type"]),
                    "c0_public_axis": c0_event.get("public_id_axis", []),
                    "c0_association_state_axis": c0_event.get("association_state_axis", []),
                    "candidate_count_event_frame": len(c0_event.get("candidate_rows", [])),
                    "target_feature_available_event_frame": bool(
                        inputs["rows"]["target_stream_source"][event_frame].get("candidate_rows", [{}])[0].get("feature")
                        is not None
                    ),
                    "source_hashes": dict(inputs["source_hashes"]),
                    "e0_equivalence": None,
                    "runtime_future_gt_used": False,
                }
            )
        except Exception as exc:  # preserve all input failures in the audit
            failures.append({"event_id": str(event.get("event_id")), "error": f"{type(exc).__name__}: {exc}"})
    if event_checks and not failures:
        try:
            first = next(event for event in events if str(event["event_id"]) == event_checks[0]["event_id"])
            inputs = replay._load_inputs(first, horizon=100)
            event_checks[0]["e0_equivalence"] = _e0_equivalence(first, inputs)
            if not event_checks[0]["e0_equivalence"]["pass"]:
                failures.append({"event_id": event_checks[0]["event_id"], "error": "BLOCKED_E0_CONTROL_NOT_EQUIVALENT"})
        except Exception as exc:
            failures.append({"event_id": event_checks[0]["event_id"], "error": f"E0 audit {type(exc).__name__}: {exc}"})
    toy = _toy_checks()
    audit = {
        "schema_version": "N72R14_INTERFACE_AUDIT_V1",
        "created_at_utc": now_utc(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "event_count": len(events),
        "event_checks_completed": len(event_checks),
        "event_checks": event_checks,
        "failure_count": len(failures),
        "failures": failures,
        "toy_checks": toy,
        "historical_outputs_modified": False,
        "runtime_future_gt_used": False,
        "real_human_evidence": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "status": "PASS_N72R14_INTERFACE_AUDIT" if not failures and toy["all_pass"] else "BLOCKED_N72R14_INTERFACE_AUDIT",
    }
    atomic_json(AUDIT_PATH, audit)
    status = {
        "schema_version": "N72R14_STAGE_STATUS_V1",
        "stage": "N72R14-00-INTERFACE-AUDIT",
        "status": audit["status"],
        "audit": str(AUDIT_PATH),
        "event_count": len(events),
        "failure_count": len(failures),
        "toy_checks_pass": bool(toy["all_pass"]),
        "runtime_future_gt_used": False,
        "historical_outputs_modified": False,
        "created_at_utc": now_utc(),
    }
    atomic_json(STATUS_PATH, status)
    print(json.dumps(status, sort_keys=True))
    return 0 if audit["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
