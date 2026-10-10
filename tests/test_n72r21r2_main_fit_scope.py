from scripts import n72r21r2_main_fit_driver as driver


def test_no_main_fit_with_missing_new_scene_or_actual_trajectory_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "OUT", tmp_path)
    sequences = ["fresh_a", "fresh_b"]
    for sequence in sequences:
        for folder in driver.PREREQUISITES:
            path = tmp_path / folder / (sequence + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    assert driver.all_required_ready(sequences)
    (tmp_path / "events/trajectory_utility_audit/fresh_b.json").unlink()
    assert not driver.all_required_ready(sequences)
    assert driver.all_required_ready(["fresh_a"])
