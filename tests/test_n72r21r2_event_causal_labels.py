from sam3_intermot.evaluation.event_causal_labels import label_actual_branch


def row(frame, uid="positive", other_uid="other"):
    outputs = [{"public_id": 1, "candidate_uid": uid}, {"public_id": 2, "candidate_uid": other_uid}]
    if uid == "other":
        outputs[1]["candidate_uid"] = "positive"
    if uid is None:
        outputs = [{"public_id": 3, "candidate_uid": "positive"}, {"public_id": 2, "candidate_uid": "other"}]
    if uid == "unknown":
        outputs = [{"public_id": 1, "candidate_uid": "unknown"}, {"public_id": 3, "candidate_uid": "positive"}, {"public_id": 2, "candidate_uid": "other"}]
    return {"frame": frame, "target_public_id": 1, "outputs": outputs, "identity_memory_write": False,
            "identity_memory_write_UID": None, "births": [], "deaths": [], "native_after": 4,
            "prototype_anchor_cosine": 1., "state_after": uid, "actor_state": {"last_box": [0, 0, 1, 1]}}


def test_hundred_propagated_errors_not_hundred_onsets():
    keep = [row(f) for f in range(101)]
    actual = [row(f, "other") for f in range(101)]
    matching = {f: {"positive": 10, "other": 20} for f in range(101)}
    labels = label_actual_branch(actual, keep, matching, 10, {1: 10, 2: 20}, {f: True for f in range(101)})
    assert labels["current_t"]["N10"] == 1
    assert labels["future"]["H100"]["N10"] == 100
    assert labels["first_global_ownership_divergent_frame"] == 0
    assert len(labels["N10_propagated_intervals_not_action_onsets"]) == 1
    assert labels["future"]["H100"]["severe_non_target_harm_public_ids"] == [2]


def test_incomplete_future_not_imputed_as_negative_label():
    keep = [row(f) for f in range(4)]
    matching = {f: {"positive": 10, "other": 20} for f in range(4)}
    labels = label_actual_branch(keep, keep, matching, 10, {1: 10, 2: 20}, {})
    assert labels["future"]["H100"]["benefit_label"] is None
    assert labels["future"]["H100"]["risk_label"] is None
    assert labels["future"]["H1"]["complete"]


def test_target_fragment_not_other_person_harm_and_none_not_unknown():
    keep = [row(0)]
    actual = [row(0, None)]
    matching = {0: {"positive": 10, "other": 10}}
    labels = label_actual_branch(actual, keep, matching, 10, {1: 10, 2: 10}, {})
    assert labels["current_t"]["NONE"] == 1
    assert labels["current_t"]["UNKNOWN"] == 0
    assert labels["current_t"]["non_target_damage"] == 0


def test_unknown_selection_is_not_verified_other():
    actual = [row(0, "unknown")]
    keep = [row(0, "unknown")]
    matching = {0: {"positive": 10, "other": 20, "unknown": None}}
    labels = label_actual_branch(actual, keep, matching, 10, {}, {})
    assert labels["current_t"]["UNKNOWN"] == 1
    assert labels["current_t"]["verified_OTHER"] == 0
