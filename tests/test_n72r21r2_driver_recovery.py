import pytest
from scripts.n72r21r2_joint_state_tensor_v2 import reference_checks, route, RoutedOutput
from scripts.n72r21r2_main_fit_driver_v2 import original_owner_alive


def test_null_before_uses_previous_exact_after_and_still_checks_current_after():
    previous = {"full_tracker_state_after_sha256": "before"}
    current = {"full_tracker_state_before_sha256": None, "full_tracker_state_after_sha256": "after"}
    checked = reference_checks(current, previous, "before", "after")
    assert checked["previous_after_before_checked"] and checked["after_checked"]
    assert not checked["direct_before_checked"]
    with pytest.raises(AssertionError, match="previous sealed"):
        reference_checks(current, previous, "wrong", "after")
    with pytest.raises(AssertionError, match="poststate"):
        reference_checks(current, previous, "before", "wrong")
    pilot = reference_checks({}, None, "whatever", "whatever")
    assert not pilot["before_available"] and not pilot["after_checked"]


def test_direct_before_not_weakened_and_original_plans_unchanged():
    with pytest.raises(AssertionError, match="direct sealed"):
        reference_checks({"full_tracker_state_before_sha256": "actual"}, None, "wrong", "after")
    assert route("on_policy/joint_state_v1/plans/v.json") == "on_policy/joint_state_v1/plans/v.json"
    assert str(RoutedOutput("/example") / "on_policy/joint_state_v1/runtime_sequences" / "v.json") == "/example/on_policy/joint_state_tensor_v2/runtime_sequences/v.json"
    assert route("on_policy/joint_state_v1/supervision/v.json") == "on_policy/joint_state_tensor_v2/supervision/v.json"
    assert not original_owner_alive(999999999)
