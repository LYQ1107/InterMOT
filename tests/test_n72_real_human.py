"""Focused contract tests for the N72 external real-human tape adapter.

All fixtures in this file are explicitly toy/non-scientific inputs.  They test
schema, provenance, mapping, and causal-boundary rejection only; none of them
is an experiment or a real human event.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sam3_intermot.interaction.n72_real_human import (
    N72RealHumanTapeRecorder,
    load_candidate_tape,
    validate_real_human_event,
)
from sam3_intermot.provenance.mapping import canonical_candidate_uid


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toy_input_files(root: Path, kind: str) -> dict[str, str]:
    (root / "ui").mkdir(parents=True, exist_ok=True)
    frame = root / "frame.bin"
    frame.write_bytes(b"N72 toy frame; not a scientific sample\n")
    payload = root / "ui" / "input.json"
    payload.write_text(json.dumps({"kind": kind, "toy": True}) + "\n", encoding="utf-8")
    result = {
        "frame_image_ref": "frame.bin",
        "frame_image_sha256": _digest(frame),
        "raw_payload_ref": "ui/input.json",
        "raw_payload_sha256": _digest(payload),
    }
    if kind == "CONFIRMED_MASK":
        mask = root / "ui" / "mask.bin"
        mask.write_bytes(b"lossless toy mask; not a scientific sample\n")
        result["mask_payload_ref"] = "ui/mask.bin"
        result["mask_payload_sha256"] = _digest(mask)
    return result


def _toy_candidate_tape(root: Path, frame_hash: str, event_frame: int = 2) -> dict:
    frames = list(range(event_frame, event_frame + 101))
    manifest = [{"frame_id": frame, "candidate_count": 1} for frame in frames]
    rows = []
    for frame in frames:
        raw_native_id = 17
        adapter_external_id = 9001
        segment_local_id = "chunk-0:17"
        sequence_global_id = "toy:g17"
        rows.append(
            {
                "sequence": "toy_sequence",
                "frame_id": frame,
                "frame_hash_sha256": frame_hash,
                "raw_native_id": raw_native_id,
                "adapter_external_id": adapter_external_id,
                "segment_local_id": segment_local_id,
                "sequence_global_id": sequence_global_id,
                "candidate_uid": canonical_candidate_uid(
                    sequence="toy_sequence",
                    frame=frame,
                    raw_native_id=raw_native_id,
                    adapter_external_id=adapter_external_id,
                    segment_local_id=segment_local_id,
                    sequence_global_id=sequence_global_id,
                ),
                "box_xyxy": [1.0, 1.0, 4.0, 4.0],
                "mask_sha256": "a" * 64,
                "feature": {"dim": 512, "finite": True, "norm": 1.0, "sha256": "b" * 64},
                "mapping": {"status": "PUBLIC_ASSIGNMENT_ABSENT", "public_id": None},
                "runtime_future_gt_used": False,
            }
        )
    return {
        "schema": "N72_CANDIDATE_TAPE_V1",
        "sequence": "toy_sequence",
        "frame_manifest": manifest,
        "rows": rows,
    }


def _toy_event(root: Path, kind: str, action: str) -> tuple[dict, dict]:
    files = _toy_input_files(root, kind)
    event_frame = 2
    human_input = {
        "kind": kind,
        "origin": "external_human_ui",
        "human_confirmed": True,
        "raw_payload_ref": files["raw_payload_ref"],
        "raw_payload_sha256": files["raw_payload_sha256"],
    }
    if kind == "BOX":
        human_input["box_xyxy"] = [1.0, 1.0, 4.0, 4.0]
        prompt_route = "native_box"
    elif kind == "CLICK":
        human_input["click_points"] = [{"x": 2.0, "y": 2.0, "label": "positive"}]
        prompt_route = "box_fallback_from_click"
    else:
        human_input["confirmed_mask"] = {
            "format": "lossless_PNG",
            "payload_ref": files["mask_payload_ref"],
            "sha256": files["mask_payload_sha256"],
            "frame_shape": [10, 10],
            "mask_origin": "human_ui_confirmed",
            "machine_candidate_mask": False,
        }
        prompt_route = "box_fallback_from_mask"

    event = {
        "event_id": f"toy-{kind.lower()}-{action.lower()}",
        "interaction_source": "real_human",
        "test_fixture": False,
        "sequence": "toy_sequence",
        "split": "train",
        "event_frame": event_frame,
        "public_id": 101,
        "public_id_source": "human_direct",
        "action_type": action,
        "runtime_future_gt_used": False,
        "current_public_id": 102,
        "other_public_id": 102,
        "atomic_transaction_confirmed": True,
        "new_identity_confirmed": True,
        "recovery_confirmed": True,
        "human_confirmed": True,
        "ui_version": "n72-toy-ui-v1",
        "session_id": "toy-session",
        "annotator_id_hash": "c" * 64,
        "event_start_timestamp": "2026-09-02T00:00:00+08:00",
        "event_end_timestamp": "2026-09-02T00:00:01+08:00",
        "frame_image_ref": files["frame_image_ref"],
        "frame_image_sha256": files["frame_image_sha256"],
        "candidate_tape_ref": "candidate.json",
        "human_input": human_input,
        "human_embedding": {
            "derived_from": kind,
            "source_kind": "human_roi_encoder",
            "feature_dim": 512,
            "finite": True,
            "norm": 1.0,
            "sha256": "d" * 64,
        },
        "prefix_range": [0, 1],
        "future_ranges": {"H20": [3, 22], "H50": [3, 52], "H100": [3, 102]},
        "spatial_correction": {
            "status": "PASS",
            "current_frame_output_frozen_before": True,
            "correction_before_memory_write": True,
            "backend_prompt_route": prompt_route,
        },
        "mapping_audit": {
            "status": "EXACT",
            "source": "direct_user_public_id",
            "public_id": 101,
            "raw_native_id": 17,
            "adapter_external_id": 9001,
            "segment_local_id": "chunk-0:17",
            "sequence_global_id": "toy:g17",
            "stable": True,
        },
        "memory_audit": {
            "event_frame_read": False,
            "current_frame_write_hidden": True,
            "first_visible_frame": 3,
            "write_after_spatial_correction": True,
        },
    }
    if action == "ADD_NEW_IDENTITY":
        event["new_identity_confirmed"] = True
    elif action == "RECOVER_IDENTITY":
        event["recovery_confirmed"] = True
    elif action == "ATOMIC_ID_SWAP":
        event["atomic_transaction_confirmed"] = True
    elif action == "DELETE":
        event["delete_confirmed"] = True
    return event, _toy_candidate_tape(root, files["frame_image_sha256"], event_frame)


@pytest.mark.parametrize("action", ["ADD_NEW_IDENTITY", "AUTHORITATIVE_REASSIGN", "ATOMIC_ID_SWAP", "RECOVER_IDENTITY"])
@pytest.mark.parametrize("kind", ["BOX", "CLICK", "CONFIRMED_MASK"])
def test_external_human_contract_accepts_four_actions_and_three_input_kinds(tmp_path: Path, action: str, kind: str) -> None:
    event, tape = _toy_event(tmp_path, kind, action)
    result = validate_real_human_event(event, candidate_tape=tape, raw_root=tmp_path)
    assert result["valid"], result["errors"]
    assert result["canonical_action_type"] == action
    assert result["runtime_future_gt_used"] is False
    assert result["candidate_tape_audit"]["required_frame_count"] == 101


def test_delete_is_supported_but_requires_confirmation(tmp_path: Path) -> None:
    event, tape = _toy_event(tmp_path, "BOX", "DELETE")
    event["delete_confirmed"] = False
    result = validate_real_human_event(event, candidate_tape=tape, raw_root=tmp_path)
    assert not result["valid"]
    assert any(item["code"] == "DELETE_CONFIRMATION_MISSING" for item in result["errors"])


def test_test_fixture_and_simulated_or_gt_input_are_rejected(tmp_path: Path) -> None:
    event, tape = _toy_event(tmp_path, "BOX", "ADD_NEW_IDENTITY")
    event["test_fixture"] = True
    event["interaction_source"] = "simulated_from_gt"
    event["gt"] = {"id": 101}
    result = validate_real_human_event(event, candidate_tape=tape, raw_root=tmp_path)
    codes = {item["code"] for item in result["errors"]}
    assert {"TEST_FIXTURE_NOT_REAL_HUMAN", "REAL_HUMAN_SOURCE_MISSING", "GT_DERIVED_FIELD_FORBIDDEN"} <= codes


def test_machine_candidate_mask_and_causal_or_mapping_breaks_are_rejected(tmp_path: Path) -> None:
    event, tape = _toy_event(tmp_path, "CONFIRMED_MASK", "ADD_NEW_IDENTITY")
    event["human_input"]["confirmed_mask"]["machine_candidate_mask"] = True
    event["memory_audit"]["event_frame_read"] = True
    event["mapping_audit"]["sequence_global_id"] = None
    tape["rows"] = tape["rows"][:-1]
    result = validate_real_human_event(event, candidate_tape=tape, raw_root=tmp_path)
    codes = {item["code"] for item in result["errors"]}
    assert {"MACHINE_MASK_NOT_ALLOWED", "CAUSAL_MEMORY_BOUNDARY_INVALID", "EVENT_MAPPING_AXIS_MISSING", "CANDIDATE_COUNT_MISMATCH"} <= codes


def test_recorder_appends_without_truncating(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    recorder = N72RealHumanTapeRecorder(path)
    recorder.append_record({"event_id": "e1", "test_fixture": True})
    first_size = path.stat().st_size
    recorder.append_record({"event_id": "e2", "test_fixture": True})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert path.stat().st_size > first_size
    assert json.loads(lines[0])["event_id"] == "e1"
    assert json.loads(lines[1])["event_id"] == "e2"


def test_json_candidate_loader_preserves_declared_container(tmp_path: Path) -> None:
    event, tape = _toy_event(tmp_path, "BOX", "ADD_NEW_IDENTITY")
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(tape), encoding="utf-8")
    loaded = load_candidate_tape(path)
    assert loaded["schema"] == "N72_CANDIDATE_TAPE_V1"
    assert len(loaded["rows"]) == 101
    assert event["candidate_tape_ref"] == "candidate.json"
