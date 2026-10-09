import numpy as np
import pytest
import torch
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES
from sam3_intermot.one_click.learned_authority import FAMILIES, TrajectoryAuthorityHead, TrajectoryAuthorityPredictor, eligible_prediction
from scripts.n72r21r1_train_authority import calibrate, supervision


@pytest.mark.parametrize('family', FAMILIES)
def test_small_heads_have_actual_trainable_gradients_and_exact_input_axis(family):
    torch.manual_seed(72111); head = TrajectoryAuthorityHead(family)
    x = torch.randn(8, 4, len(FEATURE_NAMES)); result = head(x)
    assert result.shape == (8, 4)
    result.square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    with pytest.raises(ValueError): head(x[:, :3])


def test_strict_checkpoint_and_history_have_no_mutable_predictor_state(tmp_path):
    head = TrajectoryAuthorityHead('CAUSAL_TEMPORAL'); checkpoint = tmp_path/'head.pt'
    torch.save({'schema': 'N72R21R1_TRAJECTORY_AUTHORITY_V1', 'family': 'CAUSAL_TEMPORAL', 'model': head.state_dict(),
        'feature_names': list(FEATURE_NAMES), 'FIT_mean': [0.]*32, 'FIT_std': [1.]*32,
        'selection': {'status': 'CALIBRATION_ABSTAIN'}}, checkpoint)
    predictor = TrajectoryAuthorityPredictor(checkpoint); current = np.zeros(32, np.float32)
    first = predictor.predict(current); predictor.predict(current, np.ones((3,32),np.float32))
    assert first == predictor.predict(current)
    assert not eligible_prediction(first, predictor.selection)
    with pytest.raises(ValueError): predictor.predict(np.zeros(33))


def test_calibration_requires_nonzero_verified_safe_benefit():
    protocol = {'INNER_calibration_grid': {'benefit_min': [.5], 'harm_max': [.02], 'value_min': [0.]}}
    rows = [{'group': ('episode', 'source', i), 'class': 2, 'action': {'family': 'GLOBAL_SWAP', 'candidate_uid': 'q'}} for i in range(8)]
    pred = [{'beneficial': .99, 'harmful': .001, 'abstain': .009, 'value': 1.}]*8
    chosen, points = calibrate(rows, pred, protocol)
    assert chosen['status'] == 'CALIBRATION_ABSTAIN' and points[0]['harm_rate'] == 1.
    for row in rows: row['class'] = 1
    chosen, _ = calibrate(rows, pred, protocol)
    assert chosen['status'] == 'CALIBRATED_DIAGNOSTIC' and chosen['accepted_events'] == 8


def test_short_future_window_is_not_imputed_for_training():
    row = {'raw_trajectory_labels': {'future': {'H100': {'complete': False}}}}
    assert supervision(row, 'H100_GLOBAL_RISK') is None
