import numpy as np
from pathlib import Path
import json

from sam3_intermot.association.appearance_memory import AppearanceMemory
from sam3_intermot.association.ccam_replay import paired_replay, validate_candidate_tape
from sam3_intermot.association.human_intervention import apply_intervention, human_evidence
from sam3_intermot.association.online_associator import score_matrix_pairwise
from sam3_intermot.association.state_manager import StateManager, StateManagerConfig
from sam3_intermot.association.identity_state import IdentityState
from scripts.run_n33_ccam_ablation import run as run_ablation


def feat(i, d=512):
    x = np.zeros(d, dtype=np.float32)
    x[i] = 1.0
    return x


def test_memory_capacity_decay_and_roundtrip():
    m = AppearanceMemory(anchor_cap=2, decay_frames=10)
    for f in range(3):
        assert m.update_from_human(1, f, feat(f), quality=1.0, write_event_id=f"e{f}")
    assert len(m.records[1].positive) == 2
    payload = m.serialize()
    r = AppearanceMemory.deserialize(payload)
    assert len(r.records[1].positive) == 2
    assert r.records[1].positive[-1].write_event_id == "e2"
    # Same-frame treatment is invisible; next frame is eligible.
    assert m.score(1, feat(2), 2) == 0.0
    assert m.score(1, feat(2), 3) > 0.0


def test_id_isolation_and_negative_bank():
    m = AppearanceMemory()
    m.update_from_human(1, 0, feat(1), competing_embeddings=[feat(2)])
    m.update_from_human(2, 0, feat(2))
    assert m.score(1, feat(1), 1) > m.score(1, feat(2), 1)
    assert m.score(2, feat(2), 1) > m.score(2, feat(1), 1)


def test_m0_disabled_path_matches_legacy_score():
    st = IdentityState(1, feat(1), np.array([0, 0, 10, 10]), 0, native_tid=3)
    obs = [{"feat": feat(1), "box": np.array([0, 0, 10, 10]), "native_tid": 3, "native_age": 1, "has_feat": 1.0}]
    a = score_matrix_pairwise([st], obs, 1, None)
    b = score_matrix_pairwise([st], obs, 1, None, appearance_memory=None)
    np.testing.assert_array_equal(a, b)
    manager = StateManager(StateManagerConfig(use_appearance_memory=False))
    assert manager.appearance_memory.records == {}


def test_future_only_score_boundary():
    m = AppearanceMemory()
    m.update_from_human(1, 5, feat(5))
    assert m.score(1, feat(5), 5) == 0.0
    assert m.score(1, feat(5), 6) > 0.0


def test_human_evidence_never_falls_back_to_native_feature():
    class FakeExtractor:
        feature_dim = 512

        def extract(self, seq_dir, frame, box):
            assert Path(seq_dir) == Path("/tmp/sequence")
            assert frame == 4
            np.testing.assert_array_equal(box, np.array([1, 2, 11, 22], dtype=float))
            return feat(7)

    result = human_evidence(
        {"gt_box": [1, 2, 11, 22], "event_id": "human-1"},
        4,
        FakeExtractor(),
        Path("/tmp/sequence"),
        fallback_obs={"native_tid": 23, "feat": feat(99)},
    )
    assert result["status"] == "PASS"
    assert result["native_tid"] == 23
    assert result["has_feat"]
    assert int(np.argmax(result["feat"])) == 7

    unavailable = human_evidence(
        {"gt_box": [1, 2, 11, 22], "event_id": "human-2"},
        4,
        None,
        None,
        fallback_obs={"native_tid": 23, "feat": feat(99)},
    )
    assert unavailable["status"] == "HUMAN_FEATURE_NOT_AVAILABLE"
    assert not unavailable["has_feat"]
    assert not np.any(unavailable["feat"])
    fallback_only = human_evidence(
        {"gt_box": None, "event_id": "human-3"},
        4,
        FakeExtractor(),
        Path("/tmp/sequence"),
        fallback_obs={"native_tid": 23, "box": [1, 2, 11, 22], "feat": feat(99)},
    )
    assert fallback_only["status"] == "HUMAN_FEATURE_NOT_AVAILABLE"
    assert not fallback_only["has_feat"]

    class MaskExtractor(FakeExtractor):
        def extract_mask(self, seq_dir, frame, box, mask):
            assert mask.shape == (3, 4)
            return feat(8)

    masked = human_evidence(
        {"gt_box": [1, 2, 5, 5], "mask": np.ones((3, 4), dtype=np.uint8)},
        4,
        MaskExtractor(),
        Path("/tmp/sequence"),
    )
    assert masked["status"] == "PASS"
    assert masked["source"] == "human_roi_mask"
    assert int(np.argmax(masked["feat"])) == 8


def test_machine_confidence_gate_does_not_refresh_memory():
    m = AppearanceMemory()
    assert not m.update_from_machine(1, 3, feat(3), confidence=0.49)
    assert m.records == {}
    assert m.update_from_human(1, 4, feat(4), write_event_id="h-4")
    before = m.records[1].write_count
    assert not m.update_from_machine(1, 5, feat(5), confidence=0.1)
    assert m.records[1].write_count == before
    assert m.records[1].last_human_frame == 4
    assert m.records[1].last_human_event_id == "h-4"
    assert m.records[1].positive[-1].source == "human"
    assert m.records[1].positive[-1].to_dict(frame=7)["age"] == 3


def test_spatial_commit_and_human_write_are_separate_and_audited():
    manager = StateManager(
        StateManagerConfig(use_appearance_memory=True, score_threshold=-100.0)
    )
    obs = {
        "obs_id": 4,
        "feat": feat(1),
        "box": np.array([0, 0, 10, 10], dtype=float),
        "native_tid": 8,
        "native_age": 0,
        "has_feat": 1.0,
        "conf": 1.0,
    }
    manager.rollout_frame(0, [obs])

    class FakeExtractor:
        feature_dim = 512

        def extract(self, seq_dir, frame, box):
            return feat(12)

    event = {
        "event_id": "reassign-0",
        "event_type": "M1_IDENTITY_MISMATCH",
        "action_type": "AUTHORITATIVE_REASSIGN",
        "canonical_public_id": 1,
        "current_public_id": 1,
        "gt_box": [0, 0, 10, 10],
    }
    record = apply_intervention(
        manager, event, 0, [obs], [(1, obs["box"].copy())], FakeExtractor(), Path("/tmp/sequence")
    )
    assert record["applied"]
    assert record["appearance_memory"][0]["status"] == "PASS"
    assert record["appearance_memory"][0]["pid"] == 1
    assert manager.appearance_memory.records[1].positive[-1].write_event_id == "reassign-0"
    assert int(np.argmax(manager.appearance_memory.records[1].positive[-1].feature)) == 12
    assert manager.candidate_log[-1]["human_events"] == []
    assert manager.annotate_human_event(0, event, record)
    assert manager.candidate_log[-1]["human_events"][0]["event_id"] == "reassign-0"


def test_candidate_audit_m0_and_paired_replay_are_deterministic():
    manager = StateManager(StateManagerConfig(use_appearance_memory=False))
    obs = {
        "obs_id": 0,
        "feat": feat(1),
        "box": np.array([0, 0, 10, 10], dtype=float),
        "native_tid": 2,
        "native_age": 0,
        "has_feat": 1.0,
    }
    manager.rollout_frame(0, [obs])
    audit = manager.candidate_log[-1]
    assert audit["candidate_complete"]
    assert audit["appearance_memory_enabled"] is False
    assert audit["base_scores_before_appearance"] == audit["fused_scores"]
    assert audit["human_events"] == []

    tape = {
        "candidate_complete": True,
        "prefix_state": [
            {
                "public_id": 1,
                "embedding": feat(1).tolist(),
                "box": [0, 0, 10, 10],
                "native_tid": 2,
            }
        ],
        "event": {
            "event_id": "h0",
            "frame": 0,
            "public_id": 1,
            "correction_box": [0, 0, 10, 10],
            "correction_embedding": feat(1).tolist(),
            "human_embedding": feat(1).tolist(),
            "quality": 1.0,
        },
        "frames": [
            {
                "frame": 1,
                "candidates": [
                    {
                        "obs_id": 0,
                        "embedding": feat(1).tolist(),
                        "box": [0, 0, 10, 10],
                        "native_tid": 2,
                        "confidence": 1.0,
                    }
                ],
            },
            {
                "frame": 2,
                "candidates": [
                    {
                        "obs_id": 0,
                        "embedding": feat(1).tolist(),
                        "box": [0, 0, 10, 10],
                        "native_tid": 2,
                        "confidence": 1.0,
                    }
                ],
            },
        ],
    }
    validation = validate_candidate_tape(tape)
    assert validation["valid"] and validation["candidate_complete"]
    first = paired_replay(tape)
    second = paired_replay(tape)
    assert first == second
    assert first["status"] == "PASS"
    assert first["identity_effect"] == "NOT_COMPUTABLE"
    assert first["comparison"][0]["max_abs_score_delta"] > 0.0


def test_incomplete_candidate_tape_is_not_available():
    tape = {
        "prefix_state": [
            {"public_id": 1, "embedding": feat(1).tolist(), "box": [0, 0, 1, 1]}
        ],
        "event": {
            "frame": 0,
            "public_id": 1,
            "correction_box": [0, 0, 1, 1],
            "correction_embedding": feat(1).tolist(),
            "human_embedding": feat(1).tolist(),
        },
        "frames": [
            {"frame": 1, "candidates": [{"box": [0, 0, 1, 1], "native_tid": 1}]}
        ],
    }
    validation = validate_candidate_tape(tape)
    assert validation["valid"]
    assert not validation["candidate_complete"]
    replay = paired_replay(tape)
    assert replay["status"] == "NOT_AVAILABLE"


def test_ablation_driver_runs_all_variants_without_inventing_metrics(tmp_path):
    vector = feat(1).tolist()
    tape = {
        "candidate_complete": True,
        "prefix_state": [
            {"public_id": 1, "embedding": vector, "box": [0, 0, 10, 10], "native_tid": 2}
        ],
        "event": {
            "frame": 0,
            "public_id": 1,
            "correction_box": [0, 0, 10, 10],
            "correction_embedding": vector,
            "human_embedding": vector,
            "event_id": "h0",
        },
        "frames": [
            {
                "frame": 1,
                "candidates": [
                    {"embedding": vector, "box": [0, 0, 10, 10], "native_tid": 2}
                ],
            }
        ],
    }
    input_path = tmp_path / "complete.json"
    output_path = tmp_path / "ablation.json"
    input_path.write_text(json.dumps(tape), encoding="utf-8")
    artifact = run_ablation(input_path, output_path)
    assert artifact["status"] == "PASS"
    assert set(artifact["variants"]) == {"M0", "M1", "M2", "M3", "M4"}
    assert all(row["status"] == "PASS" for row in artifact["variants"].values())
    assert all(
        row["paired_replay"]["metrics"]["future_h20_iou"] is None
        for row in artifact["variants"].values()
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["identity_effect"] == "NOT_COMPUTABLE_NO_POSTHOC_IDENTITY_LABELS"
