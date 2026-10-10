from scripts import n72r21r2_joint_state_fit_driver as driver
from scripts.n72r21r2_train_joint_state import past_actual_treatment


def test_treatment_phase_needs_own_action_strictly_in_the_past():
    row = {"source_case": "DELAYED_CHALLENGER", "frame": 3, "actual_source_effective_action_frames": [3]}
    assert not past_actual_treatment(row)
    row["frame"] = 4
    assert past_actual_treatment(row)
    row["source_case"] = "KEEP"
    assert not past_actual_treatment(row)


def test_all24_receipts_required_not_an_available_subset(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "OUT", tmp_path)
    directory = tmp_path / "on_policy/joint_state_v1/supervision"
    directory.mkdir(parents=True)
    required = ["video" + str(i) for i in range(24)]
    for sequence in required[:-1]:
        (directory / (sequence + ".json")).write_text("{}")
    assert not driver.all_sources_ready(required)
    (directory / (required[-1] + ".json")).write_text("{}")
    assert driver.all_sources_ready(required)
