"""UI ingestion smoke tests use toy bytes and never claim real evidence."""

from __future__ import annotations

import json

from ui.n72r1_human_ui import N72R1IngestionService, render_form, schema_document


def _submission(action: str = "AUTHORITATIVE_CORRECT") -> dict:
    return {
        "event_id": "ui-toy-event",
        "sequence": "toy-seq",
        "split": "train",
        "event_frame": 10,
        "action_type": action,
        "session_id": "will-be-replaced",
        "timestamp": "2026-09-02T01:02:03+08:00",
        "frame_hash_sha256": "a" * 64,
        "candidate_tape_ref": "candidate/window-0.jsonl",
        "prefix_range": [0, 9],
        "future_ranges": {"H20": [11, 30], "H50": [11, 60], "H100": [11, 110]},
        "human_confirmed": True,
        "runtime_future_gt_used": False,
        "public_id": 101,
        "public_id_source": "human_selected_existing_public",
        "human_input": {
            "kind": "BOX", "origin": "external_human_ui", "human_confirmed": True,
            "box_xyxy": [1.0, 2.0, 8.0, 10.0],
        },
    }


def test_ui_schema_and_server_finalization_are_explicit(tmp_path) -> None:
    assert schema_document()["finalization_rule"].startswith("only finalize")
    assert "GT" in render_form()
    service = N72R1IngestionService(raw_root=tmp_path / "raw", event_root=tmp_path / "events", annotator_id="toy-annotator")
    submission = _submission()
    submission["session_id"] = service.session["session_id"]
    result = service.submit_json_bytes(json.dumps(submission, sort_keys=True).encode("utf-8"))
    assert result["status"] == "ACCEPTED_REAL_HUMAN_EVENT", result
    row = json.loads((tmp_path / "events" / "real_human_events.jsonl").read_text().splitlines()[0])
    assert row["interaction_source"] == "real_human"
    assert row["server_generated_real_human"] is True
    assert row["runtime_future_gt_used"] is False
    assert list((tmp_path / "raw" / "requests").glob("*.json"))


def test_ui_rejects_gt_without_writing_final_event(tmp_path) -> None:
    service = N72R1IngestionService(raw_root=tmp_path / "raw", event_root=tmp_path / "events", annotator_id="toy-annotator")
    submission = _submission()
    submission["session_id"] = service.session["session_id"]
    submission["gt_box"] = [0, 0, 1, 1]
    result = service.submit_json_bytes(json.dumps(submission).encode("utf-8"))
    assert result["status"] == "REJECTED"
    assert not (tmp_path / "events" / "real_human_events.jsonl").exists()
    assert list((tmp_path / "raw" / "requests").glob("*.json"))
