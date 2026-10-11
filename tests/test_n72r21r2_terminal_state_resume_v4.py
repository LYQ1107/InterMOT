import pytest
from scripts import n72r21r2_joint_state_terminal_resume_v4 as resume


def test_remaining_original_nineteen_order_not_best_scene_or_partial_adoption():
    registered = [f"s{i}" for i in range(19)]
    complete = registered[:7]
    assert resume.partition(registered, complete) == registered[7:]
    for planned, proven in ((registered[:-1], complete), (registered, complete + ["other"]), (registered, complete + [complete[0]]), (registered[:-1] + [registered[0]], complete)):
        with pytest.raises(ValueError): resume.partition(planned, proven)


def test_only_resume_namespace_changes_and_originals_restored_on_error():
    original = resume.original
    saved = original.PREFIX, original.ATTEMPT_ASSETS, original.write_json
    protocol = original.PROTOCOL
    with pytest.raises(RuntimeError):
        with resume.routed():
            assert original.PREFIX == resume.PREFIX
            assert original.PROTOCOL == protocol
            assert original.ATTEMPT_ASSETS == resume.ASSETS / "joint_state_terminal_resume_v4"
            raise RuntimeError("forced diagnostic failure")
    assert (original.PREFIX, original.ATTEMPT_ASSETS, original.write_json) == saved
