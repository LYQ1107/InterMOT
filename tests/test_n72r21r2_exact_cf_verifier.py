from copy import deepcopy
from scripts.n72r21r2_exact_cf_verifier import direct_repair_opportunities


def row():
    return {"episode_uid": "click", "frame": 5, "branch": "RAW_IDENTITY_TOP", "action": {"candidate_uid": "real"},
            "offline_supervision_labels": {"current_t": {"N01": 1, "outcome": "TARGET"},
              "future": {"H100": {"complete": True, "benefit_label": True, "N01": 50, "N10": 0}},
              "effective_direct_action_frames": [5], "H100_any_harm_including_current_t_label": False}}


def test_actual_current_and_future_safe_repair_required_not_delayed_or_unknown():
    good = row()
    duplicate = deepcopy(good)
    delayed = {**row(), "branch": "DELAYED_CHALLENGER"}
    harmful = row()
    harmful["offline_supervision_labels"]["H100_any_harm_including_current_t_label"] = True
    unknown = row()
    unknown["offline_supervision_labels"]["current_t"]["outcome"] = "UNKNOWN"
    direct, future = direct_repair_opportunities([good, duplicate, delayed, harmful, unknown])
    assert dict(direct) == {("click", 5): {"real"}}
    assert dict(future) == {("click", 5): {"real"}}


def test_future_improvement_is_not_mislabeled_as_direct_current_correction():
    later = row()
    later["offline_supervision_labels"]["current_t"]["N01"] = 0
    direct, future = direct_repair_opportunities([later])
    assert not direct
    assert future
