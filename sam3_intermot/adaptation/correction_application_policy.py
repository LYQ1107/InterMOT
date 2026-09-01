"""Atomic strategy-level application of a human box correction.

The selector chooses only how the correction changes the official SAM spatial
continuation.  Current-frame delivery, the human ledger, and identity evidence
are kept outside that choice.  This module is deliberately adapter-level and
does not patch the pinned SAM3 checkout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Mapping, MutableSequence, Optional

import numpy as np

from sam3_intermot.backend.sam3_state_snapshot import restore_continuation_state
from sam3_intermot.backend.output_types import PromptObjectObservation


class CorrectionApplicationAction(Enum):
    KEEP_OLD_STATE = 0
    APPLY_CURRENT_ENSURE = 1
    PROMPT_THEN_RESTORE = 2


@dataclass
class CorrectionApplicationResult:
    action: CorrectionApplicationAction
    correction_frame: int
    public_id: int
    current_output_corrected: bool
    future_state_changed: bool
    prompt_attempted: bool
    prompt_returned_target: bool
    fallback_used: bool
    rollback_used: bool
    mapping_valid: bool
    target_state_present: bool
    provenance: str
    status: str = "PASS"
    failure: Optional[str] = None
    binding: Optional[dict[str, Any]] = None
    identity_memory_update: str = "NOT_APPLICABLE_SINGLE_ID"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action"] = self.action.name
        return value


def _tracker_ids(backend: Any) -> list[int]:
    predictor = getattr(backend, "_predictor", None)
    session_id = getattr(backend, "_session_id", None)
    entry = getattr(predictor, "_all_inference_states", {}).get(session_id) if predictor is not None else None
    state = entry.get("state") if isinstance(entry, Mapping) else None
    result: list[int] = []
    for tracker_state in (state.get("sam2_inference_states", []) if isinstance(state, Mapping) else []):
        result.extend(int(value) for value in np.asarray(tracker_state.get("obj_ids", [])).reshape(-1))
    return result


def _mapping_valid(backend: Any, public_id: int) -> bool:
    public_id = int(public_id)
    mapped = getattr(backend, "_ext_to_sam", {}).get(public_id)
    inverse = getattr(backend, "_sam_to_ext", {})
    return mapped is not None and inverse.get(int(mapped)) == public_id and int(mapped) in _tracker_ids(backend)


def _deliver_current_correction(
    backend: Any,
    *,
    frame: int,
    public_id: int,
    box: np.ndarray,
    ledger: MutableSequence[dict[str, Any]],
    raw_output_recorded: bool,
) -> bool:
    """Expose the human box in the current cache without writing SAM state."""

    observation = PromptObjectObservation(
        frame_idx=int(frame),
        sam_object_id=int(public_id),
        mask=np.zeros((1, 1), dtype=bool),
        box_xyxy=np.asarray(box, dtype=float).copy(),
        confidence=1.0,
        presence_score=1.0,
        source="human_correction",
        is_human_verified=True,
    )
    cache = list(getattr(backend, "_output_cache", {}).get(int(frame), []))
    cache = [item for item in cache if int(getattr(item, "sam_object_id", -1)) != int(public_id)]
    cache.append(observation)
    backend._output_cache[int(frame)] = cache
    ledger.append({
        "event_type": "HUMAN_BOX_CORRECTION",
        "frame": int(frame),
        "public_id": int(public_id),
        "box_xyxy": np.asarray(box, dtype=float).tolist(),
        "raw_output_recorded_before_correction": bool(raw_output_recorded),
        "current_output_source": "human_correction",
    })
    return True


def _result_state(backend: Any, public_id: int) -> tuple[bool, bool]:
    public_id = int(public_id)
    mapped = getattr(backend, "_ext_to_sam", {}).get(public_id)
    ids = _tracker_ids(backend)
    present = bool(mapped is not None and int(mapped) in ids) or (mapped is None and public_id in ids)
    return present, _mapping_valid(backend, public_id)


class CorrectionApplicationPolicy:
    """Apply one of the three frozen spatial correction strategies."""

    def __init__(self, action: CorrectionApplicationAction) -> None:
        self.action = CorrectionApplicationAction(action)

    def apply(
        self,
        backend: Any,
        *,
        correction_frame: int,
        public_id: int,
        corrected_box: np.ndarray,
        pre_correction_snapshot: Any,
        ledger: MutableSequence[dict[str, Any]],
        raw_output_recorded: bool,
        ensure_binding: Optional[Callable[..., Mapping[str, Any]]] = None,
    ) -> CorrectionApplicationResult:
        frame = int(correction_frame)
        public_id = int(public_id)
        box = np.asarray(corrected_box, dtype=float).reshape(4)
        prompt_attempted = self.action is not CorrectionApplicationAction.KEEP_OLD_STATE
        prompt_returned_target = False
        fallback_used = False
        rollback_used = False
        binding: Optional[dict[str, Any]] = None
        failure: Optional[str] = None
        committed = False
        fallback_log_before = len(getattr(backend, "_prompt_fallback_log", []))

        if self.action is CorrectionApplicationAction.KEEP_OLD_STATE:
            # K0 is not a no-op event: the official spatial state is restored,
            # while public delivery and the human ledger remain visible.
            restore_continuation_state(backend, pre_correction_snapshot)
            current_ok = _deliver_current_correction(
                backend, frame=frame, public_id=public_id, box=box,
                ledger=ledger, raw_output_recorded=raw_output_recorded,
            )
            present, mapping = _result_state(backend, public_id)
            return CorrectionApplicationResult(
                action=self.action, correction_frame=frame, public_id=public_id,
                current_output_corrected=current_ok, future_state_changed=False,
                prompt_attempted=False, prompt_returned_target=False,
                fallback_used=False, rollback_used=False, mapping_valid=mapping,
                target_state_present=present, provenance="K0_KEEP_OLD_STATE",
            )

        try:
            if self.action is CorrectionApplicationAction.APPLY_CURRENT_ENSURE:
                backend.correct_object(frame, public_id, box_xyxy=box)
                prompt_returned_target, _ = _result_state(backend, public_id)
                if ensure_binding is None:
                    raise RuntimeError("K1 requires the existing N31 ensure-binding adapter")
                binding = dict(ensure_binding(backend=backend, frame=frame, public_id=public_id, box=box))
                fallback_used = bool(binding.get("fallback_used", False)) or len(getattr(backend, "_prompt_fallback_log", [])) > fallback_log_before
                present, mapping = _result_state(backend, public_id)
                if not present or not mapping:
                    raise RuntimeError(f"K1 target admission/mapping invalid: present={present}, mapping={mapping}")
                committed = True
            else:
                # K2 explicitly forbids the low-level rectangle fallback.  A
                # prompt-produced official state is committed only when its
                # target remains present and the public/raw mapping is valid.
                backend.correct_object(
                    frame, public_id, box_xyxy=box, allow_prompt_fallback=False
                )
                present, mapping = _result_state(backend, public_id)
                prompt_returned_target = bool(present and mapping)
                if not prompt_returned_target:
                    raise RuntimeError(f"K2 official prompt did not return a valid target: present={present}, mapping={mapping}")
                committed = True
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            restore_continuation_state(backend, pre_correction_snapshot)
            rollback_used = True
            committed = False

        current_ok = _deliver_current_correction(
            backend, frame=frame, public_id=public_id, box=box,
            ledger=ledger, raw_output_recorded=raw_output_recorded,
        )
        present, mapping = _result_state(backend, public_id)
        return CorrectionApplicationResult(
            action=self.action, correction_frame=frame, public_id=public_id,
            current_output_corrected=current_ok,
            future_state_changed=bool(committed),
            prompt_attempted=prompt_attempted,
            prompt_returned_target=prompt_returned_target,
            fallback_used=fallback_used,
            rollback_used=rollback_used,
            mapping_valid=mapping,
            target_state_present=present,
            provenance=("K1_APPLY_CURRENT_ENSURE" if self.action is CorrectionApplicationAction.APPLY_CURRENT_ENSURE else "K2_PROMPT_THEN_RESTORE"),
            status="PASS" if failure is None else "ROLLBACK",
            failure=failure,
            binding=binding,
        )


def apply_correction_policy(action: CorrectionApplicationAction, backend: Any, **kwargs: Any) -> CorrectionApplicationResult:
    """Functional convenience wrapper used by replay scripts and tests."""

    return CorrectionApplicationPolicy(action).apply(backend, **kwargs)


__all__ = [
    "CorrectionApplicationAction",
    "CorrectionApplicationPolicy",
    "CorrectionApplicationResult",
    "apply_correction_policy",
]
