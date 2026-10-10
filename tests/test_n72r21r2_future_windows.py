import pytest
from sam3_intermot.evaluation.causal_identity_windows import future_window


def test_future_h1_is_next_original_frame_not_action_frame():
    rows = [{"frame": f} for f in range(106)]
    selected, complete = future_window(rows, 5, 1)
    assert selected == [{"frame": 6}] and complete
    selected, complete = future_window(rows, 5, 100)
    assert selected[0]["frame"] == 6 and selected[-1]["frame"] == 105 and complete


def test_incomplete_window_not_imputed_negative_and_gaps_rejected():
    rows = [{"frame": f} for f in range(7)]
    selected, complete = future_window(rows, 5, 100)
    assert len(selected) == 1 and not complete
    with pytest.raises(ValueError):
        future_window([{"frame": 6}, {"frame": 8}], 5, 5)
