from copy import deepcopy
import pytest
from sam3_intermot.evaluation.development_closure import ONSET_KEYS, summarize_complete_group, development_gates
from sam3_intermot.evaluation.learned_policy_evidence import METRICS

SEEDS = [1, 2, 3]
GATES = dict(G1=dict(independent_beneficial_correction_onsets_min=10, supporting_sequences_min=3,
                    severe_non_target_harm_onsets_max=0),
             G2=dict(delta_HOTA_min=.005, delta_AssA_gt=0., delta_DetA_min=-.005, delta_IDF1_min=0.,
                     IDSW_growth_max_fraction=.1, wrong_person_takeover_growth_max=0))


def fixture(sequences=("a", "b", "c"), actions=0, delta=0.):
    baselines, initializations, densities, cells = {}, {}, {}, []
    for s in sequences:
        initializations[s] = dict(inputs=[dict(episode_uid=s + "__click0", slot=0, initialization_failure=s == "failed")])
        densities[s] = "MEDIUM"
        base = {k: .3 if k not in ("IDSW", "FP", "FN") else 0. for k in METRICS}
        baselines[s] = dict(actual_nine_metrics={"CLICK_C0__click0": base})
        for seed in SEEDS:
            uid = "MAIN__unit__seed" + str(seed)
            current = {**base, "HOTA": base["HOTA"] + delta, "AssA": base["AssA"] + delta}
            labels = []
            for frame in range(actions):
                rows = [dict(frame=frame+i, N10=0, non_target_damage=0, verified_OTHER_writes=0, UNKNOWN_writes=0) for i in range(101)]
                labels.append(dict(frame=frame, labels=dict(raw_frame_components=rows,
                    future=dict(H100=dict(complete=True, observed_future_frames=100, benefit_label=True, severe_non_target_harm_public_ids=[])))))
            count = dict(N01_frames=actions, N10_frames=0, TARGET_frames=20+actions, C0_TARGET_frames=20,
                         positive_available_frames=40, physically_visible_frames=80, VERIFIED_OTHER_frames=0, C0_VERIFIED_OTHER_frames=0)
            summary = {k: actions if k in ("effective_decisions", "complete_H100_decisions", "beneficial_complete_H100_decisions") else 0 for k in ONSET_KEYS}
            cell = dict(sequence=s, seed=seed, experiment_uid=uid, density=dict(whole_video_density="MEDIUM"),
                        valid_initializations=1, all_nine_metrics={uid+"__click0":current},
                        paired_deltas_vs_C0={uid+"__click0":{k:current[k]-base[k] for k in METRICS}},
                        per_episode_target_components={s+"__click0":count}, own_onset_summary=summary,
                        effective_direct_decisions_NOT_independent_onsets=actions,
                        own_prefix_one_shot_labels=[dict(episode_uid=s+"__click0", onsets=labels)])
            if s == "failed":
                cell.update(valid_initializations=0, all_nine_metrics={}, paired_deltas_vs_C0={},
                    per_episode_target_components={}, own_prefix_one_shot_labels=[],
                    effective_direct_decisions_NOT_independent_onsets=0, own_onset_summary=dict.fromkeys(ONSET_KEYS, 0))
            cells.append(cell)
    return cells, baselines, initializations, densities


def run(args, sequences):
    return summarize_complete_group(*args, seeds=SEEDS, sequences=sequences)


def test_all_video_seed_grid_and_original_failed_clicks_are_retained():
    args = fixture(("a", "failed"))
    summary = run(args, ["a", "failed"])
    assert summary["actual_full_video"]["overall"]["usable_video_clusters"] == 1
    assert summary["actual_full_video"]["missing_or_failed_initialization_videos_retained"] == ["failed"]
    assert summary["seed_mean_full_video_components_NOT_event_counts"]["positive_available_frames"] == 40
    with pytest.raises(ValueError, match="all registered"):
        run((args[0][:-1], *args[1:]), ["a", "failed"])
    with pytest.raises(ValueError, match="all registered"):
        run((args[0] + [args[0][0]], *args[1:]), ["a", "failed"])


def test_positive_numeric_mot_or_many_seed_windows_never_prove_independent_roots():
    args = fixture(actions=2, delta=.01)
    summary = run(args, ["a", "b", "c"])
    gates = development_gates(summary, GATES)
    assert gates["G2"]["status"].startswith("PASS_NUMERIC")
    assert gates["G1"]["all_seed_click_effective_actions_UPPER_BOUND_ONLY"] == 18
    assert gates["G1"]["status"] == "UNPROVEN_INDEPENDENT_CAUSAL_ROOTS_REQUIRED"
    assert not gates["confirmation_authorized"] and not gates["next_association_stage_authorized"]
    assert summary["independent_G1_event_precision_recall"] is None


def test_zero_actions_never_qualify_and_zero_baseline_idsw_permits_no_increase():
    args = fixture()
    summary = run(args, ["a", "b", "c"])
    gates = development_gates(summary, GATES)
    assert gates["G1"]["status"].startswith("FAIL")
    assert gates["G2"]["status"] == "FAIL_NUMERIC_MOT_TARGETS"
    positive = run(fixture(actions=2, delta=.01), ["a", "b", "c"])
    positive["actual_full_video"]["overall"]["macro_deltas"]["IDSW"] = .01
    assert not development_gates(positive, GATES)["G2"]["checks"]["IDSW"]


def test_exact_pairing_density_and_every_actual_onset_are_checked():
    args = fixture(actions=1)
    bad = deepcopy(args); bad[0][0]["paired_deltas_vs_C0"][bad[0][0]["experiment_uid"]+"__click0"]["HOTA"] = .1
    with pytest.raises(ValueError, match="same exact C0"):
        run(bad, ["a", "b", "c"])
    bad = deepcopy(args); bad[0][0]["density"]["whole_video_density"] = "CROWDED"
    with pytest.raises(ValueError, match="frozen density"):
        run(bad, ["a", "b", "c"])
    bad = deepcopy(args); bad[0][0]["own_prefix_one_shot_labels"][0]["onsets"] = []
    with pytest.raises(ValueError, match="counters"):
        run(bad, ["a", "b", "c"])


def test_incomplete_h100_is_not_persistent_correction_or_severe_harm_hidden():
    args = fixture(actions=1)
    onset = args[0][0]["own_prefix_one_shot_labels"][0]["onsets"][0]
    onset["labels"]["raw_frame_components"] = onset["labels"]["raw_frame_components"][:11]
    onset["labels"]["future"]["H100"].update(complete=False, observed_future_frames=10, benefit_label=None)
    args[0][0]["own_onset_summary"].update(complete_H100_decisions=0, incomplete_H100_decisions=1, beneficial_complete_H100_decisions=0)
    summary = run(args, ["a", "b", "c"])
    assert summary["seed_mean_one_shot_decisions_NOT_independent_roots"]["incomplete_H100_decisions"] == pytest.approx(1/3)
    for c in args[0]:
        c["own_prefix_one_shot_labels"][0]["onsets"][0]["labels"]["future"]["H100"]["severe_non_target_harm_public_ids"] = [9]
        c["own_onset_summary"]["severe_non_target_harm_decisions"] = 1
    gates = development_gates(run(args, ["a", "b", "c"]), GATES)
    assert "OBSERVED_SEVERE_OTHER_PERSON_HARM" in gates["G1"]["negative_evidence"]


def test_no_unknown_as_verified_takeover_or_fraction_as_component_count():
    args = fixture(actions=2, delta=.01)
    for c in args[0]: c["per_episode_target_components"][c["sequence"]+"__click0"]["UNKNOWN_frames"] = 50
    gates = development_gates(run(args, ["a", "b", "c"]), GATES)
    assert gates["G2"]["checks"]["wrong_takeover"]
    args[0][0]["per_episode_target_components"]["a__click0"]["UNKNOWN_frames"] = .5
    with pytest.raises(ValueError, match="integer exposure"):
        run(args, ["a", "b", "c"])


def test_actual_pinned_tracker_metadata_is_separate_from_nine_metrics_and_units():
    args = fixture(actions=2, delta=.01)
    for c in args[0]:
        values = c["all_nine_metrics"][c["experiment_uid"]+"__click0"]
        values.update(tracker=c["experiment_uid"], per_sequence={c["sequence"]: {"IDF1": .3}})
    summary = run(args, ["a", "b", "c"])
    assert set(summary["actual_full_video"]["overall"]["macro_metrics"]) == set(METRICS)
    bad = deepcopy(args); bad[0][0]["all_nine_metrics"][bad[0][0]["experiment_uid"]+"__click0"]["HOTA"] = 30.
    with pytest.raises(ValueError, match="not percentages"):
        run(bad, ["a", "b", "c"])
    bad = deepcopy(args); bad[0][0]["all_nine_metrics"][bad[0][0]["experiment_uid"]+"__click0"]["FN"] = .5
    with pytest.raises(ValueError, match="integer exposure"):
        run(bad, ["a", "b", "c"])
