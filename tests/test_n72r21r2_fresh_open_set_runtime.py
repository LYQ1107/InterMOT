import pytest
import torch
from sam3_intermot.one_click.fresh_open_set_runtime import FreshOpenSetPredictor
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES, feature_vector
from sam3_intermot.one_click.open_set_verifier import OpenSetVerifierHead


def checkpoint(tmp_path, **extra):
    saved = dict(schema="N72R21R2_CURRENT_AXIS_OPEN_SET_V1", feature_names=list(FEATURE_NAMES), head_family="LOGISTIC",
                 model=OpenSetVerifierHead("LOGISTIC").state_dict(), FIT_mean=[0.] * 32, FIT_std=[1.] * 32,
                 selection=dict(status="INNER_CALIBRATED_CURRENT_IDENTITY_ONLY", temperature=1.), association_authority=False)
    saved.update(extra)
    path = tmp_path / "current_identity.pt"
    torch.save(saved, path)
    return path


def test_old_schema_uncalibrated_or_authority_checkpoint_are_rejected(tmp_path):
    for extra in [dict(schema="N72R21R1_CURRENT_AXIS_OPEN_SET_V1"), dict(selection=dict(status="UNCALIBRATED_EPOCH_NOT_DEPLOYABLE")), dict(association_authority=True)]:
        with pytest.raises(ValueError): FreshOpenSetPredictor(checkpoint(tmp_path, **extra))


def test_predictor_reads_only_registered_runtime_inputs_not_truth_or_future(tmp_path):
    predictor = FreshOpenSetPredictor(checkpoint(tmp_path))
    features = {k: 0. for k in FEATURE_NAMES}
    axis = [dict(candidate_uid="p", features=features, feature_vector=feature_vector(features).tolist()),
            dict(candidate_uid=None, features=features, feature_vector=feature_vector(features).tolist())]
    clean = predictor.predict_axis(axis)
    dirty = [{**r, "current_outcome": "FAKE_TARGET", "future_truth": 999} for r in axis]
    assert predictor.predict_axis(dirty) == clean
    assert len(clean) == 2 and all(not p.requires_grad for p in predictor.model.parameters())
    axis[0]["feature_vector"][0] = 99.
    with pytest.raises(ValueError): predictor.predict_axis(axis)
