import torch
from torch import nn

from sam3.sam.transformer import TwoWayTransformer
from sam3_intermot.adaptation.corrected_mask_teacher import BOX_DERIVED_PSEUDO_MASK
from sam3_intermot.adaptation.decoder_update_transaction import (
    DecoderCorrectionEvent,
    DecoderUpdateConfig,
    DecoderUpdateTransaction,
)
from sam3_intermot.adaptation.sam3_decoder_lit import (
    DecoderLITConfig,
    SAM3DecoderLITAdapter,
)


def _fixture(seed=2901):
    torch.manual_seed(seed)
    decoder = nn.Module()
    decoder.transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=256,
        mlp_dim=256,
        num_heads=8,
    )
    decoder.head = nn.Linear(256, 64)
    for parameter in decoder.head.parameters():
        parameter.requires_grad = False
    adapter = SAM3DecoderLITAdapter(
        decoder,
        DecoderLITConfig(rank=4, alpha=4.0, dropout=0.0),
    )
    state = adapter.new_state("fixture", seed, device="cpu")
    support = (
        torch.randn(1, 256, 2, 2),
        torch.randn(1, 256, 2, 2),
        torch.randn(1, 1, 256),
    )
    future = (
        torch.randn(1, 256, 2, 2),
        torch.randn(1, 256, 2, 2),
        torch.randn(1, 1, 256),
    )

    def forward(inputs):
        image, pe, points = inputs
        token, _ = decoder.transformer(image, pe, points)
        return decoder.head(token[:, 0]).view(1, 1, 8, 8)

    event = DecoderCorrectionEvent(
        video_id="fixture",
        public_id=seed,
        frame_idx=0,
        provenance=BOX_DERIVED_PSEUDO_MASK,
        box_xyxy=(1.0, 1.0, 7.0, 7.0),
        image_size=(8, 8),
    )
    return decoder, adapter, state, support, future, forward, event


def test_zero_update_is_bitwise_noop_for_parameters_and_future_logits():
    decoder, adapter, state, support, future, forward, event = _fixture()
    before = {name: value.detach().clone() for name, value in state.named_parameters()}
    with torch.no_grad():
        base = forward(future)
    update = DecoderUpdateTransaction(
        adapter,
        DecoderUpdateConfig(inner_steps=5, learning_rate=0.01, optimizer_enabled=False),
    ).apply(
        event,
        state,
        forward_fn=lambda _supervision, _step: forward(support),
        deterministic_forward_fn=lambda _supervision: forward(support).detach(),
    )
    with adapter.activate(state), torch.no_grad():
        adapted = forward(future)
    assert update.committed
    assert all(torch.equal(before[name], value) for name, value in state.named_parameters())
    assert torch.equal(base, adapted)
    assert update.optimization_diagnostic["parameter_delta_l2_total"] == 0.0
    assert update.optimization_diagnostic["logit_delta_linf"] == 0.0


def test_validator_rollback_restores_adapter_state():
    decoder, adapter, state, support, _future, forward, event = _fixture(seed=2902)
    before = {name: value.detach().clone() for name, value in state.named_parameters()}
    update = DecoderUpdateTransaction(
        adapter,
        DecoderUpdateConfig(inner_steps=2, learning_rate=0.01),
    ).apply(
        event,
        state,
        forward_fn=lambda _supervision, _step: forward(support),
        validator_fn=lambda *_args: (False, "test_validator_rejected"),
    )
    assert update.status == "ROLLBACK"
    assert not update.committed
    assert update.rollback_reason == "test_validator_rejected"
    assert state.adapter_version == 0
    assert all(torch.equal(before[name], value) for name, value in state.named_parameters())


def test_committed_adapter_changes_strict_future_continuous_logits():
    decoder, adapter, state, support, future, forward, event = _fixture(seed=2903)
    with torch.no_grad():
        base = forward(future)
    update = DecoderUpdateTransaction(
        adapter,
        DecoderUpdateConfig(inner_steps=5, learning_rate=0.01),
    ).apply(
        event,
        state,
        forward_fn=lambda _supervision, _step: forward(support),
    )
    with adapter.activate(state), torch.no_grad():
        adapted = forward(future)
    assert update.committed
    assert state.adapter_version == 1
    assert float((adapted - base).abs().max()) > 0.0
