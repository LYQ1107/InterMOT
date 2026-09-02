"""Append-only/path safety toy tests."""

from __future__ import annotations

import json

import pytest

from sam3_intermot.provenance.append_only import AppendOnlyJSONL, AppendOnlyJSONLError


def test_append_only_jsonl_hash_chain_and_duplicate_rejection(tmp_path) -> None:
    store = AppendOnlyJSONL("events.jsonl", root=tmp_path, key_field="event_id")
    first = store.append({"event_id": "e1", "runtime_future_gt_used": False})
    second = store.append({"event_id": "e2", "runtime_future_gt_used": False})
    assert first["previous_event_sha256"] is None
    assert second["previous_event_sha256"] == first["event_sha256"]
    assert [row["event_id"] for row in store.rows()] == ["e1", "e2"]
    with pytest.raises(AppendOnlyJSONLError, match="duplicate"):
        store.append({"event_id": "e1"})


def test_append_only_jsonl_rejects_path_escape(tmp_path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        AppendOnlyJSONL("../outside.jsonl", root=tmp_path, key_field="event_id")
