#!/usr/bin/env python3
"""Focused N72R11 contract smoke without SAM3 or scientific data.

The fake future session is deliberately marked as a toy fixture.  It tests
the controller lifecycle and the tensor contracts only; its output is never
used as an experiment result.
"""

from __future__ import annotations

from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import sys
import tempfile
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sam3_intermot.reacquisition.live_requery_controller as controller_module  # noqa: E402
from sam3_intermot.association.target_edge_bridge import (  # noqa: E402
    BRIDGE_INPUT_DIM,
    TargetEdgeBridge,
    build_target_edge_feature,
    fit_residual_scale,
)
from sam3_intermot.reacquisition.live_requery_controller import LiveFutureRequeryController  # noqa: E402
from sam3_intermot.reacquisition.models.n72r11_temporal_v3 import (  # noqa: E402
    CANDIDATE_FEATURE_DIM,
    DISTRACTOR_SLOTS,
    LONG_TERM_TRUSTED_SLOTS,
    N72R11TemporalIdentityModel,
    RECENT_TRUSTED_SLOTS,
    SOURCE_FEATURE_DIM,
    TEMPORAL_FEATURE_DIM,
    n72r11_loss,
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def toy_rows(event_id: str, frame: int) -> list[dict[str, object]]:
    return [
        {
            "candidate_uid": f"{event_id}:toy:{frame}:{index}",
            "candidate_kind": "FUTURE_FRAME_REQUERY_CANDIDATE",
            "candidate_source": "FUTURE_FRAME_REQUERY",
            "frame": int(frame),
            "requery_name": f"TOY_{index}",
            "box_xyxy": [1.0 + index, 2.0, 11.0 + index, 22.0],
            "official_raw_sam_id": 100 + index,
            "public_id": None,
            "public_id_inference": False,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
        }
        for index in range(4)
    ]


def controller_smoke() -> dict[str, object]:
    original = controller_module.FutureFrameRequerySession
    created: list[object] = []

    class FakeSession:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self._closed = False
            self.trigger_frame = None
            self.end_frame = None
            self._rows: list[dict[str, object]] = []
            created.append(self)

        def start_from_frame(self, trigger_frame: int, predicted_box: list[float], causal_state: dict[str, object], *, end_frame: int, main_y_pre_frozen: bool) -> dict[str, object]:
            self.trigger_frame = int(trigger_frame)
            self.end_frame = int(end_frame)
            return {"status": "PASS_TOY_STARTED", "runtime_future_gt_used": False}

        def query_current_frame(self) -> list[dict[str, object]]:
            assert self.trigger_frame is not None
            self._rows = toy_rows("toy-event", int(self.trigger_frame))
            return list(self._rows)

        def propagate_if_selected(self, *, selected_query_name: str | None = None, selected_candidate_uid: str | None = None, selection_audit: dict[str, object] | None = None, none_score: float | None = None, margin: float | None = None) -> list[dict[str, object]]:
            if selected_candidate_uid is None:
                self._closed = True
                return []
            assert self.trigger_frame is not None
            return self._rows[:1] + toy_rows("toy-event", int(self.trigger_frame) + 1)[:1]

        def audit(self) -> dict[str, object]:
            return {"closed": self._closed, "runtime_future_gt_used": False}

        def close(self) -> None:
            self._closed = True

    controller_module.FutureFrameRequerySession = FakeSession  # type: ignore[assignment]
    try:
        controller = LiveFutureRequeryController(
            backend_factory=lambda: object(),
            sequence="toy-sequence",
            event_id="toy-event",
            event_frame=0,
            target_public_id=1007,
            frame_paths={1: __file__, 2: __file__, 3: __file__},
            feature_fn=None,
            end_frame=3,
        )
        session, probe = controller.probe(
            frame=1,
            predicted_box=[1.0, 2.0, 11.0, 22.0],
            causal_state={"previous_raw_sam_id": 7, "runtime_future_gt_used": False},
        )
        selected = controller.commit(
            session=session,
            selected_candidate_uid=str(probe[0]["candidate_uid"]),
            selection_audit={"selector": "toy", "runtime_future_gt_used": False},
            none_score=0.0,
            margin=0.5,
        )
        if not getattr(session, "_closed", False):
            raise AssertionError("selected session was not released after materialization")
        if len(selected) != 2 or not controller.active_candidates(2):
            raise AssertionError("toy selected source was not retained")
        session2, _ = controller.probe(
            frame=2,
            predicted_box=[1.0, 2.0, 11.0, 22.0],
            causal_state={"previous_raw_sam_id": 7, "runtime_future_gt_used": False},
        )
        controller.commit(
            session=session2,
            selected_candidate_uid=None,
            selection_audit={"selector": "toy_none", "runtime_future_gt_used": False},
            none_score=0.0,
            margin=0.1,
        )
        controller.close()
        probe_parameters = list(inspect.signature(LiveFutureRequeryController.probe).parameters.values())
        commit_parameters = list(inspect.signature(LiveFutureRequeryController.commit).parameters.values())
        if not all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in probe_parameters[1:]):
            raise AssertionError("probe must expose keyword-only public API")
        if not all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in commit_parameters[1:]):
            raise AssertionError("probe/commit must expose keyword-only public APIs")
        return {
            "status": "PASS_TOY_CONTROLLER_CONTRACT",
            "created_session_count": len(created),
            "selected_row_count": len(selected),
            "controller_audit": controller.audit(),
        }
    finally:
        controller_module.FutureFrameRequerySession = original  # type: ignore[assignment]


def tensor_smoke() -> dict[str, object]:
    candidate = {
        "candidate_source": "FUTURE_FRAME_REQUERY",
        "incumbent_public_id_if_any": None,
        "confidence": 0.8,
        "presence_score": 0.8,
        "motion_iou": 0.4,
    }
    features = build_target_edge_feature(
        candidate,
        candidate_logit=0.4,
        none_logit=0.1,
        legacy_target_score=0.2,
        legacy_public_scores={1007: 0.2, 1008: 0.3},
        target_public_id=1007,
    )
    if len(features) != BRIDGE_INPUT_DIM or not np.isfinite(np.asarray(features)).all():
        raise AssertionError("target-edge bridge feature contract failed")
    scale = fit_residual_scale([0.0, 2.0], [0.0, 0.0])
    expected_scale = float(max(np.percentile(np.abs(np.asarray([0.0, 2.0]) - np.asarray([0.0, 0.0])), 95.0), 1.0))
    if not scale >= 1.0 or not np.isclose(scale, expected_scale):
        raise AssertionError("bridge residual scale did not follow train-only p95/floor protocol")
    bridge = TargetEdgeBridge(residual_scale=scale)
    delta, calibrated = bridge(torch.as_tensor([features]), torch.as_tensor([0.2]))
    if delta.shape != (1,) or calibrated.shape != (1,) or not torch.isfinite(calibrated).all():
        raise AssertionError("target-edge bridge forward contract failed")
    model = N72R11TemporalIdentityModel()
    batch, count = 2, 3
    tensors = (
        torch.randn(batch, count, CANDIDATE_FEATURE_DIM),
        torch.ones(batch, count, dtype=torch.bool),
        torch.nn.functional.one_hot(torch.tensor([[0, 1, 3], [0, 0, 1]]), SOURCE_FEATURE_DIM).float(),
        torch.randn(batch, 512),
        torch.randn(batch, RECENT_TRUSTED_SLOTS, 512),
        torch.ones(batch, RECENT_TRUSTED_SLOTS, dtype=torch.bool),
        torch.randn(batch, LONG_TERM_TRUSTED_SLOTS, 512),
        torch.ones(batch, LONG_TERM_TRUSTED_SLOTS, dtype=torch.bool),
        torch.randn(batch, DISTRACTOR_SLOTS, 512),
        torch.ones(batch, DISTRACTOR_SLOTS, dtype=torch.bool),
        torch.randn(batch, 512),
        torch.randn(batch, TEMPORAL_FEATURE_DIM),
    )
    logits = model(*tensors)
    loss, loss_parts = n72r11_loss(logits, torch.tensor([0, count]), torch.ones(batch, count, dtype=torch.bool))
    if logits.shape != (batch, count + 1) or not torch.isfinite(loss):
        raise AssertionError("N72R11 model/loss contract failed")
    loss.backward()
    return {
        "status": "PASS_TOY_TENSOR_CONTRACT",
        "bridge_input_dim": BRIDGE_INPUT_DIM,
        "bridge_scale": scale,
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "loss_parts": {key: float(value) for key, value in loss_parts.items()},
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    output = ROOT / "outputs/N72R11/attempts" / f"n72r11_focused_smoke_attempt_{int(args.attempt):02d}.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing smoke artifact: {output}")
    try:
        result = {
            "schema_version": "N72R11_FOCUSED_TOY_SMOKE_V1",
            "status": "PASS_TOY_CONTRACT_SMOKE",
            "created_at_utc": now_utc(),
            "scientific_result": None,
            "controller": controller_smoke(),
            "tensors": tensor_smoke(),
            "toy_fixture_only": True,
            "runtime_future_gt_used": False,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
        print(json.dumps({"status": result["status"], "output": str(output)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = output.with_suffix(".failure.json")
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(
            json.dumps(
                {
                    "schema_version": "N72R11_FOCUSED_TOY_SMOKE_FAILURE_V1",
                    "status": "FAIL_TOY_CONTRACT_SMOKE",
                    "attempt": int(args.attempt),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "scientific_result": None,
                    "toy_fixture_only": True,
                    "runtime_future_gt_used": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "FAIL_TOY_CONTRACT_SMOKE", "failure": str(failure), "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
