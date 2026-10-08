import numpy as np
import pytest
import torch
from sam3_intermot.identity_verification.cross_scene_adapter import CrossSceneIdentityAdapter
from sam3_intermot.association.identity_authority import AdapterEnsemble, AuthorityConfig, calibrated_residual, AuthorityController


def checkpoint(tmp_path):
    model = CrossSceneIdentityAdapter()
    p = tmp_path / "adapter.pt"
    torch.save({"architecture": "CrossSceneIdentityAdapter", "feature_dim": 512, "bottleneck_dim": 128, "trainable_parameters": 265472, "parameter_fit_sequences": ["fit"], "seed": 720321, "state_dict": model.state_dict()}, p)
    return p


def test_actual_tower_inference_and_axis(tmp_path):
    model = AdapterEnsemble([checkpoint(tmp_path)])
    generator = np.random.default_rng(3)
    x = generator.normal(size=(3, 512)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q = x[0]
    scores = model.scores(q, x)
    assert model.calls == 1
    with torch.inference_mode():
        expected = model.models[0](torch.tensor(q[None]), torch.tensor(x[None]))["scores"].numpy()[0]
    np.testing.assert_allclose(scores, expected, atol=1e-6)
    np.testing.assert_allclose(model.scores(q, x[::-1].copy()), scores[::-1], atol=1e-6)
    with pytest.raises(ValueError, match="axis"):
        model.scores(q, x, [np.zeros((2, 512))])


def test_hash_and_supervision_checks(tmp_path):
    p = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="SHA"):
        AdapterEnsemble([p], expected_shas=["wrong"])
    with pytest.raises(ValueError, match="supervision"):
        AdapterEnsemble([p], forbidden_sequences=["fit"])


@pytest.mark.parametrize("mode", ["scalar", "logistic", "mlp", "structured"])
def test_small_causal_controllers(mode):
    model = AuthorityController(mode)
    assert sum(p.numel() for p in model.parameters()) < 50000
    scores = model(torch.ones((4, 10)))
    assert ((scores >= 0) & (scores <= 1)).all()


def test_calibration_is_explicit_training_scale():
    c = AuthorityConfig(identity_mean=0.4, identity_std=0.2, base_scale=2)
    np.testing.assert_allclose(calibrated_residual(np.array([0.4, 0.6]), c), [0, 2])
