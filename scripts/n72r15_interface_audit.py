#!/usr/bin/env python3
"""CPU interface and causal-contract audit for N72R15."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.effect_assignment import solve_effect_assignment  # noqa: E402
from sam3_intermot.association.relative_persistent_state_edge import (  # noqa: E402
    build_relative_persistent_state_edge_matrix,
    fuse_row_max_preserving,
)
from sam3_intermot.association.trusted_persistent_public_state import (  # noqa: E402
    TrustedPersistentPublicAssociationBank,
)
from sam3_intermot.reacquisition.target_candidate_pool import build_candidate_pool  # noqa: E402
from scripts import n72r11_on_demand_replay as replay  # noqa: E402
from scripts import n72r15_run_repaired_persistent_state_formal as formal  # noqa: E402


PROTOCOL = ROOT / "outputs/N72R9/protocol.json"
OUTPUT_ROOT = ROOT / "outputs/N72R15"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            import os

            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _events() -> list[dict[str, Any]]:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    events = payload.get("source_event_selection", {}).get("events", [])
    if not isinstance(events, list) or len(events) != 32:
        raise RuntimeError("N72R9 protocol does not contain exactly 32 events")
    return [dict(item) for item in events]


def _toy_checks() -> dict[str, Any]:
    feature0 = np.zeros(512, dtype=np.float32)
    feature0[0] = 1.0
    feature1 = np.zeros(512, dtype=np.float32)
    feature1[1] = 1.0
    bank = TrustedPersistentPublicAssociationBank()
    bank.ensure_state(7, 1003, last_box=[0, 0, 10, 10], last_seen_frame=10, previous_raw_sam_id=4, previous_native_scope="toy")
    bank.ensure_state(8, 1004, last_box=[20, 0, 30, 10], last_seen_frame=10, previous_raw_sam_id=5, previous_native_scope="toy")
    bank.appearance_memory.update_from_human(1003, 10, feature0, write_event_id="toy")
    candidates = [
        {"candidate_uid": "toy-target", "box_xyxy": [0, 0, 10, 10], "feature": feature0.tolist(), "official_raw_sam_id": 4, "native_scope": "toy"},
        {"candidate_uid": "toy-other", "box_xyxy": [20, 0, 30, 10], "feature": feature1.tolist(), "official_raw_sam_id": 5, "native_scope": "toy"},
    ]
    edge = build_relative_persistent_state_edge_matrix(
        bank=bank,
        candidate_rows=candidates,
        public_id_axis=[1003, 1004],
        frame=11,
        target_public_id=1003,
        state_scope="HUMAN_TARGET_ONLY",
    )
    base = np.asarray([[0.20, 0.10], [0.10, 0.20]], dtype=np.float64)
    fusion = fuse_row_max_preserving(base, np.asarray(edge["relative_delta"], dtype=np.float64))
    hidden = bank.appearance_memory.score(1003, feature0, 10) == 0.0
    visible = bank.appearance_memory.score(1003, feature0, 11) > 0.0
    base_matrix = np.asarray([[2.0, 0.0], [0.0, 2.0]])
    treatment_matrix = np.asarray([[0.0, 2.0], [2.0, 0.0]])
    state_pairs = [(7, 1003), (8, 1004)]
    base_solver = solve_effect_assignment(
        candidate_rows=candidates,
        persistent_states=bank.states_for_pairs(state_pairs),
        fused_state_candidate_scores=base_matrix.T,
        source_run_id="n72r15:toy:base",
        session_id="n72r15:toy",
    )
    treatment_solver = solve_effect_assignment(
        candidate_rows=candidates,
        persistent_states=bank.states_for_pairs(state_pairs),
        fused_state_candidate_scores=treatment_matrix.T,
        source_run_id="n72r15:toy:treatment",
        session_id="n72r15:toy",
    )
    before = bank.digest()
    consensus = bank.update_from_consensus(
        frame=11,
        candidate_rows=candidates,
        base_solver=base_solver,
        treatment_solver=treatment_solver,
        target_public_id=1003,
    )
    after = bank.digest()
    return {
        "fixture": "toy_non_scientific",
        "event_frame_memory_read_false": bool(hidden),
        "first_memory_visible_event_plus_one": bool(visible),
        "relative_edge_finite": bool(np.isfinite(np.asarray(edge["relative_delta"], dtype=float)).all()),
        "row_max_preserved": bool(fusion["row_max_preserved"]),
        "row_max_exact": bool(np.allclose(np.max(base, axis=1), np.max(fusion["fused"], axis=1), atol=1.0e-10, rtol=0.0)),
        "explicit_axes_distinct": bool(bank.state_to_public == {7: 1003, 8: 1004}),
        "consensus_disagreement_count": int(consensus["disagreement_public_count"]),
        "disagreement_machine_write_count": len(consensus["machine_memory_write_public_ids"]),
        "disagreement_state_changed_for_audit": bool(before != after),
        "all_pass": bool(
            hidden
            and visible
            and np.isfinite(np.asarray(edge["relative_delta"], dtype=float)).all()
            and fusion["row_max_preserved"]
            and consensus["disagreement_public_count"] == 2
            and not consensus["machine_memory_write_public_ids"]
        ),
    }


def _event_check(event: Mapping[str, Any]) -> dict[str, Any]:
    inputs = dict(replay._load_inputs(event, horizon=100))
    event_frame = int(inputs["event_frame"])
    frame_checks = 0
    target_session_excluded = True
    for key, rows in inputs["rows"].items():
        expected = list(range(event_frame, event_frame + 101))
        actual = [int(rows[frame]["frame"]) for frame in sorted(rows)]
        if actual != expected:
            raise RuntimeError(f"{event['event_id']}:{key}: frame axis mismatch")
        for frame in sorted(rows):
            if rows[frame].get("runtime_future_gt_used") is not False:
                raise RuntimeError(f"{event['event_id']}:{key}:{frame}: runtime future GT flag")
            frame_checks += 1
    future_main = inputs["rows"]["c0_source"][event_frame + 1]
    pool, audit = build_candidate_pool(
        list(future_main.get("candidate_rows", [])),
        (),
        sequence=str(event["sequence"]),
        frame=event_frame + 1,
        include_target_session=False,
        require_positive_geometry=True,
    )
    if any(item.get("candidate_source") != "MAIN_B0_CANDIDATE" for item in pool):
        raise RuntimeError(f"{event['event_id']}: non-MAIN candidate entered repaired pool")
    return {
        "event_id": str(event["event_id"]),
        "sequence": str(event["sequence"]),
        "event_frame": event_frame,
        "action_type": str(event["action_type"]),
        "frame_checks": frame_checks,
        "future_main_candidate_count": len(pool),
        "candidate_pool_policy": "MAIN_B0_ONLY",
        "target_session_candidate_in_solver": False,
        "geometry_audit": audit,
        "runtime_future_gt_used": False,
    }


def _e0_equivalence(event: Mapping[str, Any]) -> dict[str, Any]:
    inputs = dict(replay._load_inputs(event, horizon=100))
    inputs["protocol_event"] = dict(event)
    frame = int(inputs["event_frame"]) + 1
    c0 = inputs["rows"]["c0_source"][frame]
    pool, _ = build_candidate_pool(
        list(c0.get("candidate_rows", [])),
        (),
        sequence=str(event["sequence"]),
        frame=frame,
        include_target_session=False,
        require_positive_geometry=True,
    )
    scored = formal._exact_base(inputs, frame, pool, event_id=str(event["event_id"]), variant="E0_EQUIVALENCE")
    regenerated = replay._solver_rows(pool, scored["base_solver"])
    frozen = formal._corrected_e0_rows(str(event["event_id"]))[1]
    frozen_audit = frozen.get("score_audit", {})
    matrix = np.asarray(scored["base_matrix"], dtype=float)
    reference = np.asarray(frozen_audit.get("fused_score_matrix", []), dtype=float)
    return {
        "event_id": str(event["event_id"]),
        "checked_frame": frame,
        "candidate_axis_equal": [str(item["candidate_uid"]) for item in regenerated] == [str(item["candidate_uid"]) for item in frozen.get("candidate_rows", [])],
        "public_axis_equal": scored["base_solver"].get("public_id_axis") == frozen_audit.get("public_id_axis"),
        "score_matrix_exact_equal": bool(reference.shape == matrix.shape and np.array_equal(reference, matrix)),
        "semantic_assignment_equal": formal._semantic_assignment({"candidate_rows": regenerated}) == formal._semantic_assignment(frozen),
        "pass": bool(
            [str(item["candidate_uid"]) for item in regenerated] == [str(item["candidate_uid"]) for item in frozen.get("candidate_rows", [])]
            and scored["base_solver"].get("public_id_axis") == frozen_audit.get("public_id_axis")
            and reference.shape == matrix.shape
            and np.array_equal(reference, matrix)
            and formal._semantic_assignment({"candidate_rows": regenerated}) == formal._semantic_assignment(frozen)
        ),
    }


def main() -> int:
    try:
        events = _events()
        checks: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        for event in events:
            try:
                checks.append(_event_check(event))
            except Exception as exc:
                failures.append({"event_id": str(event.get("event_id")), "error": f"{type(exc).__name__}: {exc}"})
        e0 = None
        if not failures:
            recover = sorted(
                [item for item in events if str(item.get("action_type")) == "RECOVER_IDENTITY"],
                key=lambda item: (str(item["sequence"]), int(item["event_frame"]), str(item["event_id"])),
            )
            if not recover:
                failures.append({"event_id": "<none>", "error": "no RECOVER_IDENTITY event"})
            else:
                e0 = _e0_equivalence(recover[0])
                if not e0["pass"]:
                    failures.append({"event_id": str(recover[0]["event_id"]), "error": "BLOCKED_E0_CONTROL_NOT_EQUIVALENT"})
        toy = _toy_checks()
        audit = {
            "schema_version": "N72R15_INTERFACE_AUDIT_V1",
            "created_at_utc": now_utc(),
            "protocol": str(PROTOCOL),
            "protocol_sha256": sha256_file(PROTOCOL),
            "event_count": len(events),
            "event_checks_completed": len(checks),
            "event_checks": checks,
            "failure_count": len(failures),
            "failures": failures,
            "e0_equivalence": e0,
            "toy_checks": toy,
            "candidate_pool_policy": "MAIN_B0_ONLY",
            "target_session_candidate_in_solver": False,
            "historical_outputs_modified": False,
            "runtime_future_gt_used": False,
            "real_human_evidence": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "status": "PASS_N72R15_INTERFACE_AUDIT" if not failures and toy["all_pass"] else "BLOCKED_N72R15_INTERFACE_AUDIT",
        }
        atomic_json(OUTPUT_ROOT / "interface_audit.json", audit)
        stage_status = audit["status"]
        atomic_json(
            OUTPUT_ROOT / "stage_00_status.json",
            {
                "schema_version": "N72R15_STAGE_STATUS_V1",
                "stage": "N72R15-00-INTERFACE-AUDIT",
                "status": stage_status,
                "interface_audit": str(OUTPUT_ROOT / "interface_audit.json"),
                "event_count": len(events),
                "failure_count": len(failures),
                "toy_checks_pass": bool(toy["all_pass"]),
                "runtime_future_gt_used": False,
                "historical_outputs_modified": False,
                "created_at_utc": now_utc(),
            },
        )
        for name, stage, detail in (
            ("stage_01_status.json", "N72R15-01-MAIN-B0-CANDIDATE-AXIS", {"candidate_pool_policy": "MAIN_B0_ONLY", "target_session_candidate_in_solver": False}),
            ("stage_02_status.json", "N72R15-02-TRUSTED-PERSISTENT-STATE", {"implementation": "trusted_persistent_public_state.py", "status_detail": "READY"}),
            ("stage_03_status.json", "N72R15-03-RELATIVE-ROW-MAX-EDGE", {"implementation": "relative_persistent_state_edge.py", "row_max_preservation_contract": "atol=1e-10, rtol=0"}),
            ("stage_04_status.json", "N72R15-04-CONSENSUS-MACHINE-UPDATE", {"implementation": "TrustedPersistentPublicAssociationBank.update_from_consensus", "disagreement_trusted_write_required": 0}),
        ):
            atomic_json(
                OUTPUT_ROOT / name,
                {
                    "schema_version": "N72R15_STAGE_STATUS_V1",
                    "stage": stage,
                    "status": "PASS_IMPLEMENTATION_READY" if stage_status.startswith("PASS") else "BLOCKED_INTERFACE_AUDIT",
                    "interface_audit": str(OUTPUT_ROOT / "interface_audit.json"),
                    "event_count": len(events),
                    "runtime_future_gt_used": False,
                    "historical_outputs_modified": False,
                    **detail,
                    "created_at_utc": now_utc(),
                },
            )
        print(json.dumps({"status": audit["status"], "event_count": len(events), "failure_count": len(failures), "output": str(OUTPUT_ROOT / "interface_audit.json")}, sort_keys=True))
        return 0 if audit["status"].startswith("PASS") else 2
    except Exception as exc:
        failure = {
            "schema_version": "N72R15_FAILURE_V1",
            "status": "FAIL_N72R15_INTERFACE_CONTROLLER",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "created_at_utc": now_utc(),
        }
        atomic_json(OUTPUT_ROOT / "interface_audit_failure.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
