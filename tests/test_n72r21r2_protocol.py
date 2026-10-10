"""Before-effects protocol and isolation contracts (not a scientific gate)."""
import json
from pathlib import Path
import pytest

from scripts.n72r21r2_common import ROOT, OUT, GOAL, preregistration, development_sequence, write_json


def test_goal_frozen_and_consistent():
    goal = json.loads((OUT / "FINAL_GOAL.json").read_text())
    protocol = preregistration()
    status = json.loads((OUT / "stage_status.json").read_text())
    assert goal["goal"] == protocol["goal"] == status["goal"] == GOAL
    assert goal["goal_frozen"] and protocol["frozen"]
    assert goal["full_global_assignment_required"] and goal["runtime_future_gt_forbidden"]


def test_frozen_40_split_disjoint_and_inherited():
    protocol = preregistration()
    original = ROOT.parent / "InterMOT_N72R21R1_worktree/outputs/N72R21R1/data/NEW_SEQUENCE_SPLIT.json"
    assert protocol["split"] == json.loads(original.read_text())["split"]
    assert protocol["fresh_candidate_pilot"] == protocol["split"]["fit"][:3]


@pytest.mark.parametrize("sequence", ["dancetrack0015", "dancetrack0001", "not_a_sequence"])
def test_development_rejects_confirmation_and_historical(sequence):
    with pytest.raises(ValueError):
        development_sequence(sequence)


def test_output_traversal_forbidden():
    with pytest.raises(ValueError):
        write_json("../../not_stage.json", {})


def test_nonvacuity_and_actual_joint_metrics():
    p = preregistration()
    assert p["gates"]["G1"]["independent_beneficial_correction_onsets_min"] == 10
    assert p["gates"]["G1"]["supporting_sequences_min"] == 3
    assert len(p["trackeval"]["metrics"]) == 9
    assert p["gates"]["G4"]["accepted_writes_min"] > 0
    assert p["gates"]["G4"]["correct_observation_retention_min"] == .6
    assert p["hyperparameters"]["seeds"] == [730101, 730102, 730103]


def test_fresh_preparation_not_blocked_by_r1_science():
    goal = json.loads((OUT / "FINAL_GOAL.json").read_text())
    assert goal["fresh_fit_inner_preparation_independent_of_old_development_gate"]
    assert goal["sot_deferred"] and goal["one_click_only"]
