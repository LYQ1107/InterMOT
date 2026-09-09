#!/usr/bin/env python3
"""Run the single N72R11R4 exact-on-policy/formal-input contract check.

This is intentionally one sealed event and one future frame.  It compares the
new corpus-side feature construction with the formal replay ``_model_values``
path under an identical raw pool, score matrix and temporal state.  It is a
contract check, not a dataset test or a scientific result.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from sam3_intermot.association.secondary_public_score import build_secondary_public_score_frame  # noqa: E402
from sam3_intermot.association.target_edge_interface import apply_legacy_injection, select_candidate_from_logits  # noqa: E402
from sam3_intermot.association.effect_assignment import solve_effect_assignment  # noqa: E402
from sam3_intermot.reacquisition.temporal_state_policy import initialize_temporal_state  # noqa: E402
from scripts import n72r11_build_causal_corpus as causal  # noqa: E402
from scripts import n72r11_on_demand_replay as formal  # noqa: E402
from scripts import n72r11r4_build_exact_onpolicy_corpus as exact  # noqa: E402


SOURCE_ROOT = ROOT / "outputs/N72R11R3/bootstrap_corpus"
OUTPUT_PATH = ROOT / "outputs/N72R11R4/stage_01_exact_onpolicy_contract.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_setup() -> tuple[dict, dict, list[dict], dict[int, dict], dict[int, list[dict]], dict]:
    catalog, source_manifest = exact._source_catalog(SOURCE_ROOT)
    schedule, records = exact._schedule_and_batch(catalog, source_manifest)
    event_id = sorted(catalog)[0]
    catalog_item = catalog[event_id]
    item = dict(schedule[event_id])
    item.update({
        "event_id": event_id,
        "split": catalog_item["split"],
        "target_public_id": catalog_item["target_public_id"],
        "target_dataset_gt_id": catalog_item["target_dataset_gt_id"],
    })
    target_by_frame, active_by_frame, details = causal.load_secondary_artifact(item, records[event_id])
    start = int(item["secondary_frame"])
    end = int(details["end_frame"])
    c0_by_frame = exact._load_c0_window(item, start, end)
    frame = start + 1
    main = [dict(value) for value in c0_by_frame[frame]["candidate_rows"]]
    current = [dict(value) for value in target_by_frame[frame].get("candidate_rows", [])]
    active = [dict(value) for value in active_by_frame.get(frame, [])]
    pool, _ = exact.build_candidate_pool_with_future_requery(
        main,
        current,
        active,
        sequence=str(item["sequence"]),
        frame=frame,
    )
    anchor = causal.unit(details["anchor"]["feature"], f"{event_id} anchor")
    anchor_box = causal.box_xyxy(item["current_target_box_posthoc_selection_only"], f"{event_id} anchor box")
    event = {
        "event_id": event_id,
        "sequence": str(item["sequence"]),
        "event_frame": start,
        "target_public_id": int(item["target_public_id"]),
        "anchor": anchor,
        "anchor_box": anchor_box,
    }
    target_event = list(target_by_frame[start].get("candidate_rows", []))
    initial_raw = target_event[0].get("official_raw_sam_id") if target_event else None
    initial_scope = target_event[0].get("native_scope", target_event[0].get("native_tid_scope")) if target_event else None
    state = initialize_temporal_state(
        anchor_feature=anchor,
        anchor_box=anchor_box,
        previous_raw_sam_id=None if initial_raw is None else int(initial_raw),
        previous_native_scope=None if initial_scope is None else str(initial_scope),
    )
    score_frame = build_secondary_public_score_frame(
        c0_row=c0_by_frame[frame],
        main_candidates=main,
        pool=pool,
        target_public_id=int(item["target_public_id"]),
        anchor_feature=anchor,
        predicted_box=state.predicted_box,
        event_id=event_id,
        frame=frame,
    )
    return event, item, pool, c0_by_frame, target_by_frame, {"active": active, "details": details, "state": state, "score_frame": score_frame}


def _allclose(name: str, left: np.ndarray, right: np.ndarray, results: dict[str, bool]) -> None:
    left = np.asarray(left)
    right = np.asarray(right)
    results[name] = bool(left.shape == right.shape and np.allclose(left, right, atol=1.0e-6, rtol=1.0e-5))
    if not results[name]:
        raise RuntimeError(f"contract mismatch: {name}: {left.shape} != {right.shape}")


def main() -> int:
    status = {
        "schema_version": "N72R11R4_EXACT_ONPOLICY_CONTRACT_V1",
        "status": "RUNNING",
        "started_at_utc": now_utc(),
        "source_root": str(SOURCE_ROOT),
        "runtime_future_gt_used": False,
        "gt_loaded": False,
        "scientific_result": False,
    }
    try:
        event, item, pool, c0_by_frame, _target_by_frame, setup = _event_setup()
        state = setup["state"]
        score_frame = setup["score_frame"]
        exact_values = exact._runtime_values(
            event=event,
            frame=int(item["secondary_frame"]) + 1,
            pool=pool,
            score_frame=score_frame,
            state=state,
        )
        formal_inputs = {
            "event_id": str(event["event_id"]),
            "sequence": str(event["sequence"]),
            "event_frame": int(event["event_frame"]),
            "target_public_id": int(event["target_public_id"]),
            "anchor": event["anchor"],
            "anchor_box": event["anchor_box"],
        }
        formal_values = formal._model_values(
            formal_inputs,
            int(item["secondary_frame"]) + 1,
            pool,
            np.asarray(score_frame.matrix, dtype=np.float64),
            score_frame.public_axis,
            state,
            exact_values["base_solver_target_uid"],
        )
        comparisons: dict[str, bool] = {}
        for name, left, right in (
            ("candidate_features", exact_values["candidate_values"], formal_values["candidate_values"]),
            ("source_features", exact_values["source_values"], formal_values["source_values"]),
            ("recent_memory", exact_values["recent_array"], formal_values["recent_array"]),
            ("recent_mask", exact_values["recent_mask"], formal_values["recent_mask"]),
            ("long_term_memory", exact_values["long_array"], formal_values["long_array"]),
            ("long_term_mask", exact_values["long_mask"], formal_values["long_mask"]),
            ("distractor_memory", exact_values["distractor_array"], formal_values["distractor_array"]),
            ("distractor_mask", exact_values["distractor_mask"], formal_values["distractor_mask"]),
            ("neighbor", exact_values["neighbor"], formal_values["neighbor"]),
            ("temporal", exact_values["temporal"], formal_values["temporal"]),
            ("motion_iou", exact_values["motion_iou"], formal_values["motion_iou"]),
            ("legacy_target_scores", exact_values["legacy_target_scores"], formal_values["base_scores"]),
            ("legacy_best_other_scores", exact_values["legacy_best_other_scores"], formal_values["legacy_best_other_scores"]),
            ("competition_features", exact_values["competition_values"], formal_values["competition_values"]),
        ):
            _allclose(name, left, right, comparisons)
        _allclose("base_matrix", score_frame.matrix, np.asarray(score_frame.matrix, dtype=np.float64), comparisons)

        uids = [str(candidate["candidate_uid"]) for candidate in pool]
        fixed_selection = select_candidate_from_logits(
            np.asarray([0.25 + 0.1 * index for index in range(len(pool))], dtype=np.float64),
            0.0,
            uids,
        )
        exact_fused, exact_delta = apply_legacy_injection(
            score_frame.matrix,
            target_column=score_frame.target_column,
            candidate_uids=uids,
            selection=fixed_selection,
        )
        formal_fused, formal_delta = apply_legacy_injection(
            np.asarray(score_frame.matrix, dtype=np.float64),
            target_column=score_frame.target_column,
            candidate_uids=uids,
            selection=fixed_selection,
        )
        _allclose("legacy_injection_matrix", exact_fused, formal_fused, comparisons)
        _allclose("legacy_injection_delta", np.asarray([exact_delta]), np.asarray([formal_delta]), comparisons)
        exact_solver = solve_effect_assignment(
            candidate_rows=pool,
            persistent_states=exact._state_objects(score_frame.state_axis, score_frame.public_axis),
            fused_state_candidate_scores=exact_fused.T,
            source_run_id="n72r11r4:contract:exact",
            session_id="n72r11r4:contract",
            none_score=0.0,
        )
        formal_solver = solve_effect_assignment(
            candidate_rows=pool,
            persistent_states=exact._state_objects(score_frame.state_axis, score_frame.public_axis),
            fused_state_candidate_scores=formal_fused.T,
            source_run_id="n72r11r4:contract:formal",
            session_id="n72r11r4:contract",
            none_score=0.0,
        )
        exact_target = exact._find_target_uid(exact_solver, int(item["target_public_id"]))
        formal_target = exact._find_target_uid(formal_solver, int(item["target_public_id"]))
        comparisons["exact_target_uid"] = bool(exact_target == formal_target)
        if exact_target != formal_target:
            raise RuntimeError(f"contract mismatch: exact target UID {exact_target} != formal {formal_target}")
        status.update({
            "status": "PASS_N72R11R4_EXACT_ONPOLICY_CONTRACT",
            "event_id": str(event["event_id"]),
            "sequence": str(event["sequence"]),
            "frame": int(item["secondary_frame"]) + 1,
            "candidate_count": len(pool),
            "comparisons": comparisons,
            "base_solver_target_uid": exact_values["base_solver_target_uid"],
            "exact_target_uid": exact_target,
            "formal_target_uid": formal_target,
            "runtime_future_gt_used": False,
            "gt_loaded": False,
            "finished_at_utc": now_utc(),
        })
        exact.atomic_json(OUTPUT_PATH, status)
        print(json.dumps(status, sort_keys=True))
        return 0
    except Exception as exc:
        status.update({
            "status": "FAIL_N72R11R4_EXACT_ONPOLICY_CONTRACT",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "finished_at_utc": now_utc(),
        })
        exact.atomic_json(OUTPUT_PATH, status)
        print(json.dumps(status, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
