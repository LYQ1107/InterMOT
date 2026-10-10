import numpy as np
from scripts.n72r21r2_event_context_v2 import sealed_click


def test_tracker_click_keeps_exact_raw_vector_even_when_identity_actor_normalizes():
    raw = np.array([.123456, .987654], np.float32)
    raw /= np.linalg.norm(raw)
    raw *= np.float32(1 - 2e-7)
    event = {"frame": 0, "clicked_candidate_uid": "p", "box_xyxy": [0, 0, 20, 60]}
    click = sealed_click(event, raw)
    assert np.array_equal(raw, click["human_anchor"])
    assert not np.array_equal(raw / np.linalg.norm(raw), click["human_anchor"])
    assert not np.shares_memory(raw, click["human_anchor"])
    assert click["target_candidate_uid"] == "p"
