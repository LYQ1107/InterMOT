"""Frozen N72R18 learned identity memory for N72R20 integration.

This module is deliberately a runtime adapter, not a new identity model.  It
loads the N72R18 GRU strictly, keeps state under an explicit public/state
authority axis, scores before updating, and accepts a machine observation only
after the frozen base and treatment assignments reach consensus.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..identity_memory.encoder import FEATURE_DIMENSION, normalize
from ..identity_memory.updater import VARIANT_GRU, build_updater


FEATURE_DIM = FEATURE_DIMENSION
SCHEMA_VERSION = "N72R20_LEARNED_IDENTITY_MEMORY_BANK_V1"
UPDATE_SCHEMA_VERSION = "N72R20_LEARNED_IDENTITY_UPDATE_V1"


def _unit(value: Any, label: str = "feature") -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != FEATURE_DIM or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite {FEATURE_DIM}-D feature")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-8:
        raise ValueError(f"{label} must have a non-zero norm")
    # Preserve already-normalized float32 states exactly.  Re-normalizing a
    # serialized state can change a few low bits and break transaction digest
    # equality even though the mathematical vector is unchanged.
    if abs(norm - 1.0) <= 1.0e-5:
        return array.copy()
    return (array / norm).astype(np.float32)


def _hash_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_uid(row: Mapping[str, Any]) -> str:
    uid = row.get("candidate_uid")
    if uid in (None, ""):
        raise ValueError("every candidate row needs a candidate_uid")
    return str(uid)


def _feature_from_candidate(row: Mapping[str, Any]) -> np.ndarray | None:
    value = row.get("feature", row.get("embedding"))
    if value is None:
        return None
    return _unit(value, "candidate feature")


def _assert_no_runtime_gt(rows: Sequence[Mapping[str, Any]]) -> None:
    if any(bool(row.get("runtime_future_gt_used", False)) for row in rows):
        raise ValueError("runtime_future_gt_used must be false")


def _assignment_by_public(solver: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    if solver.get("runtime_future_gt_used") is True:
        raise ValueError("runtime_future_gt_used must be false")
    assignments = solver.get("public_assignments")
    if not isinstance(assignments, list):
        raise ValueError("solver artifact lacks public_assignments")
    result: dict[int, dict[str, Any]] = {}
    for item in assignments:
        if not isinstance(item, Mapping) or item.get("public_id") is None:
            continue
        public_id = int(item["public_id"])
        if public_id in result:
            raise ValueError(f"duplicate public assignment {public_id}")
        result[public_id] = dict(item)
    return result


@dataclass
class LearnedIdentityRecord:
    """State owned by one immutable public/state authority binding."""

    public_id: int
    association_state_id: int
    _human_anchor: np.ndarray
    _current_state: np.ndarray
    initialized_frame: int
    last_update_frame: int
    machine_update_count: int
    human_update_count: int
    checkpoint_sha256: str
    encoder_sha256: str
    interaction_source: str

    def __post_init__(self) -> None:
        self.public_id = int(self.public_id)
        self.association_state_id = int(self.association_state_id)
        if self.public_id <= 0 or self.association_state_id <= 0:
            raise ValueError("public_id and association_state_id must be positive")
        self._human_anchor = _unit(self._human_anchor, "human anchor")
        self._current_state = _unit(self._current_state, "learned state")
        self.initialized_frame = int(self.initialized_frame)
        self.last_update_frame = int(self.last_update_frame)
        self.machine_update_count = max(0, int(self.machine_update_count))
        self.human_update_count = max(1, int(self.human_update_count))
        self.checkpoint_sha256 = str(self.checkpoint_sha256)
        self.encoder_sha256 = str(self.encoder_sha256)
        self.interaction_source = str(self.interaction_source)

    @property
    def human_anchor(self) -> np.ndarray:
        """Return a copy so the human-confirmed anchor cannot be mutated in place."""

        return self._human_anchor.copy()

    @property
    def current_state(self) -> np.ndarray:
        return self._current_state.copy()

    def state_hash(self) -> str:
        return _hash_bytes(self._current_state)

    def anchor_hash(self) -> str:
        return _hash_bytes(self._human_anchor)

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_id": self.public_id,
            "association_state_id": self.association_state_id,
            "human_anchor": self._human_anchor.tolist(),
            "current_state": self._current_state.tolist(),
            "initialized_frame": self.initialized_frame,
            "last_update_frame": self.last_update_frame,
            "machine_update_count": self.machine_update_count,
            "human_update_count": self.human_update_count,
            "checkpoint_sha256": self.checkpoint_sha256,
            "encoder_sha256": self.encoder_sha256,
            "interaction_source": self.interaction_source,
            "human_anchor_hash": self.anchor_hash(),
            "current_state_hash": self.state_hash(),
        }


class LearnedIdentityMemoryBank:
    """Causal, public-ID-scoped runtime for the frozen N72R18 GRU."""

    schema_version = SCHEMA_VERSION

    def __init__(
        self,
        updater: torch.nn.Module,
        *,
        checkpoint_sha256: str,
        encoder_sha256: str,
        device: str | torch.device = "cpu",
    ) -> None:
        if not isinstance(updater, torch.nn.Module):
            raise TypeError("updater must be a torch.nn.Module")
        self.device = torch.device(device)
        self.updater = updater.to(self.device)
        self.updater.eval()
        for parameter in self.updater.parameters():
            parameter.requires_grad_(False)
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.encoder_sha256 = str(encoder_sha256)
        self.records: dict[int, LearnedIdentityRecord] = {}
        self._update_keys: set[tuple[int, int, str]] = set()
        self.update_audit: list[dict[str, Any]] = []

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        encoder_sha256: str,
        expected_encoder_sha256: str | None = None,
        expected_checkpoint_sha256: str | None = None,
        device: str | torch.device = "cpu",
    ) -> "LearnedIdentityMemoryBank":
        """Strictly load the frozen N72R18 GRU and validate feature lineage."""

        checkpoint_path = Path(checkpoint_path)
        actual_checkpoint_sha256 = sha256_file(checkpoint_path)
        if expected_checkpoint_sha256 is not None and actual_checkpoint_sha256 != str(expected_checkpoint_sha256):
            raise ValueError(
                "N72R18 checkpoint SHA mismatch: "
                f"expected {expected_checkpoint_sha256}, got {actual_checkpoint_sha256}"
            )
        if expected_encoder_sha256 is not None and str(encoder_sha256) != str(expected_encoder_sha256):
            raise ValueError(
                "feature-lineage encoder SHA mismatch: "
                f"expected {expected_encoder_sha256}, got {encoder_sha256}"
            )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("stage") != "N72R18":
            raise ValueError("checkpoint is not an N72R18 checkpoint")
        if checkpoint.get("variant") != VARIANT_GRU:
            raise ValueError(
                f"N72R20 requires frozen {VARIANT_GRU}, got {checkpoint.get('variant')}"
            )
        config = dict(checkpoint.get("config", {}))
        feature_dim = int(config.get("feature_dimension", FEATURE_DIM))
        if feature_dim != FEATURE_DIM:
            raise ValueError(f"N72R18 feature dimension {feature_dim} != {FEATURE_DIM}")
        updater = build_updater(
            VARIANT_GRU,
            feature_dim=feature_dim,
            gate_hidden_dim=int(config.get("gate_hidden_dim", 256)),
        )
        # strict=True is intentional: a partially loaded or altered memory is
        # not a valid N72R20 treatment.
        updater.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return cls(
            updater,
            checkpoint_sha256=actual_checkpoint_sha256,
            encoder_sha256=str(encoder_sha256),
            device=device,
        )

    @classmethod
    def from_updater(
        cls,
        updater: torch.nn.Module,
        *,
        checkpoint_sha256: str = "TEST_OR_INJECTED_UPDATER",
        encoder_sha256: str = "TEST_OR_INJECTED_ENCODER",
        device: str | torch.device = "cpu",
    ) -> "LearnedIdentityMemoryBank":
        """Construct an explicitly injected frozen updater for unit tests."""

        return cls(
            updater,
            checkpoint_sha256=checkpoint_sha256,
            encoder_sha256=encoder_sha256,
            device=device,
        )

    def initialize_public_identity(
        self,
        *,
        public_id: int,
        association_state_id: int,
        frame: int,
        human_anchor: Sequence[float] | np.ndarray,
        interaction_source: str,
    ) -> dict[str, Any]:
        """Initialize once from the explicitly confirmed human observation."""

        public_id = int(public_id)
        association_state_id = int(association_state_id)
        if public_id in self.records:
            raise ValueError(f"public identity {public_id} is already initialized")
        if any(record.association_state_id == association_state_id for record in self.records.values()):
            raise ValueError(f"association state {association_state_id} is already bound")
        anchor = _unit(human_anchor, "human anchor")
        record = LearnedIdentityRecord(
            public_id=public_id,
            association_state_id=association_state_id,
            _human_anchor=anchor,
            _current_state=anchor,
            initialized_frame=int(frame),
            last_update_frame=int(frame),
            machine_update_count=0,
            human_update_count=1,
            checkpoint_sha256=self.checkpoint_sha256,
            encoder_sha256=self.encoder_sha256,
            interaction_source=str(interaction_source),
        )
        self.records[public_id] = record
        return {
            "public_id": public_id,
            "association_state_id": association_state_id,
            "frame": int(frame),
            "interaction_source": str(interaction_source),
            "human_anchor_hash": record.anchor_hash(),
            "current_state_hash": record.state_hash(),
            "runtime_future_gt_used": False,
        }

    def _ordered_records(self, public_id_axis: Sequence[int]) -> list[LearnedIdentityRecord | None]:
        publics = [int(value) for value in public_id_axis]
        if len(publics) != len(set(publics)):
            raise ValueError("public_id_axis contains duplicates")
        return [self.records.get(public) for public in publics]

    def score_matrix(
        self,
        *,
        candidate_rows: Sequence[Mapping[str, Any]],
        public_id_axis: Sequence[int],
        frame: int,
    ) -> dict[str, Any]:
        """Score current states against candidates before any update."""

        _assert_no_runtime_gt(candidate_rows)
        rows = [dict(row) for row in candidate_rows]
        uids = [_candidate_uid(row) for row in rows]
        if len(uids) != len(set(uids)):
            raise ValueError("candidate_uid collision")
        frame = int(frame)
        records = self._ordered_records(public_id_axis)
        matrix = np.zeros((len(records), len(rows)), dtype=np.float64)
        feature_available = np.zeros(len(rows), dtype=bool)
        for column, row in enumerate(rows):
            feature = _feature_from_candidate(row)
            if feature is None:
                continue
            feature_available[column] = True
            for row_index, record in enumerate(records):
                if record is None or frame <= record.initialized_frame:
                    continue
                matrix[row_index, column] = float(np.dot(record._current_state, feature))
        if not np.isfinite(matrix).all():
            raise RuntimeError("learned identity score matrix contains non-finite values")
        return {
            "schema_version": "N72R20_LEARNED_IDENTITY_SCORE_V1",
            "frame": frame,
            "candidate_uids": uids,
            "public_id_axis": [int(value) for value in public_id_axis],
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "state_candidate_scores": matrix,
            "feature_available_by_candidate": feature_available,
            "state_available_by_public": [record is not None for record in records],
            "runtime_future_gt_used": False,
        }

    def update_from_consensus(
        self,
        *,
        frame: int,
        candidate_rows: Sequence[Mapping[str, Any]],
        base_solver: Mapping[str, Any],
        treatment_solver: Mapping[str, Any],
        confidence_by_uid: Mapping[str, float] | None = None,
        source: str = "consensus_trusted_machine_observation",
    ) -> dict[str, Any]:
        """Update only equal, non-NONE base/treatment assignments."""

        rows = [dict(row) for row in candidate_rows]
        _assert_no_runtime_gt(rows)
        base_map = _assignment_by_public(base_solver)
        treatment_map = _assignment_by_public(treatment_solver)
        if set(base_map) != set(treatment_map):
            raise ValueError("base/treatment public axes differ")
        public_axis = sorted(set(base_map))
        by_uid = {_candidate_uid(row): row for row in rows}
        frame = int(frame)
        confidence_by_uid = {} if confidence_by_uid is None else {
            str(key): float(value) for key, value in confidence_by_uid.items()
        }
        consensus_public_ids: list[int] = []
        disagreement_public_ids: list[int] = []
        unassigned_public_ids: list[int] = []
        updated_public_ids: list[int] = []
        skipped_uninitialized_public_ids: list[int] = []
        skipped_missing_feature_public_ids: list[int] = []
        updates: list[dict[str, Any]] = []

        for public_id in public_axis:
            base_item = base_map[public_id]
            treatment_item = treatment_map[public_id]
            base_uid = base_item.get("candidate_uid")
            treatment_uid = treatment_item.get("candidate_uid")
            base_uid = None if base_uid in (None, "", "None") else str(base_uid)
            treatment_uid = None if treatment_uid in (None, "", "None") else str(treatment_uid)
            if base_uid is None and treatment_uid is None:
                unassigned_public_ids.append(public_id)
                continue
            if base_uid is None or treatment_uid is None or base_uid != treatment_uid:
                disagreement_public_ids.append(public_id)
                continue
            consensus_public_ids.append(public_id)
            record = self.records.get(public_id)
            if record is None:
                skipped_uninitialized_public_ids.append(public_id)
                continue
            candidate = by_uid.get(treatment_uid)
            if candidate is None:
                raise ValueError(f"consensus candidate {treatment_uid} is absent")
            observation = _feature_from_candidate(candidate)
            if observation is None:
                skipped_missing_feature_public_ids.append(public_id)
                continue
            key = (public_id, frame, treatment_uid)
            if key in self._update_keys:
                continue
            same_frame_keys = [item for item in self._update_keys if item[0] == public_id and item[1] == frame]
            if same_frame_keys:
                raise RuntimeError(
                    f"public identity {public_id} already has a different learned update at frame {frame}"
                )
            previous_state = record._current_state.copy()
            previous_hash = record.state_hash()
            previous_anchor_hash = record.anchor_hash()
            with torch.no_grad():
                previous_tensor = torch.as_tensor(previous_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                new_state, reliability, _candidate_state = self.updater(previous_tensor, observation_tensor)
            updated = _unit(new_state.detach().cpu().numpy()[0], "new learned state")
            record._current_state = updated
            record.last_update_frame = frame
            record.machine_update_count += 1
            self._update_keys.add(key)
            confidence = confidence_by_uid.get(treatment_uid, candidate.get("confidence", candidate.get("presence_score", 0.0)))
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0
            audit = {
                "schema_version": UPDATE_SCHEMA_VERSION,
                "public_id": public_id,
                "association_state_id": record.association_state_id,
                "frame": frame,
                "candidate_uid": treatment_uid,
                "base_assignment": dict(base_item),
                "treatment_assignment": dict(treatment_item),
                "consensus": True,
                "candidate_feature_hash": _hash_bytes(observation),
                "previous_state_hash": previous_hash,
                "new_state_hash": record.state_hash(),
                "human_anchor_hash": previous_anchor_hash,
                "source": str(source),
                "confidence": confidence,
                "reliability": float(reliability.detach().cpu().numpy()[0]),
                "checkpoint_sha256": record.checkpoint_sha256,
                "encoder_sha256": record.encoder_sha256,
                "runtime_future_gt_used": False,
            }
            if audit["human_anchor_hash"] != record.anchor_hash():
                raise RuntimeError("human anchor changed during learned update")
            self.update_audit.append(audit)
            updates.append(audit)
            updated_public_ids.append(public_id)

        return {
            "schema_version": "N72R20_LEARNED_IDENTITY_CONSENSUS_UPDATE_V1",
            "frame": frame,
            "public_id_axis": public_axis,
            "consensus_public_ids": consensus_public_ids,
            "disagreement_public_ids": disagreement_public_ids,
            "unassigned_public_ids": unassigned_public_ids,
            "updated_public_ids": updated_public_ids,
            "skipped_uninitialized_public_ids": skipped_uninitialized_public_ids,
            "skipped_missing_feature_public_ids": skipped_missing_feature_public_ids,
            "updates": updates,
            "runtime_future_gt_used": False,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "encoder_sha256": self.encoder_sha256,
            "records": {str(public): record.to_dict() for public, record in sorted(self.records.items())},
            "update_keys": [list(key) for key in sorted(self._update_keys)],
            "update_audit": json.loads(json.dumps(self.update_audit, sort_keys=True)),
            "runtime_future_gt_used": False,
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != self.schema_version:
            raise ValueError("learned identity memory schema mismatch")
        if payload.get("checkpoint_sha256") != self.checkpoint_sha256:
            raise ValueError("learned identity checkpoint SHA mismatch on restore")
        if payload.get("encoder_sha256") != self.encoder_sha256:
            raise ValueError("learned identity encoder SHA mismatch on restore")
        restored: dict[int, LearnedIdentityRecord] = {}
        for public_s, item in dict(payload.get("records", {})).items():
            public_id = int(public_s)
            record = LearnedIdentityRecord(
                public_id=public_id,
                association_state_id=int(item["association_state_id"]),
                _human_anchor=item["human_anchor"],
                _current_state=item["current_state"],
                initialized_frame=int(item["initialized_frame"]),
                last_update_frame=int(item["last_update_frame"]),
                machine_update_count=int(item["machine_update_count"]),
                human_update_count=int(item["human_update_count"]),
                checkpoint_sha256=str(item["checkpoint_sha256"]),
                encoder_sha256=str(item["encoder_sha256"]),
                interaction_source=str(item["interaction_source"]),
            )
            if record.public_id != public_id:
                raise ValueError("snapshot public authority mismatch")
            if item.get("human_anchor_hash") != record.anchor_hash() or item.get("current_state_hash") != record.state_hash():
                raise ValueError("snapshot state hash mismatch")
            restored[public_id] = record
        update_keys = {tuple((int(item[0]), int(item[1]), str(item[2]))) for item in payload.get("update_keys", [])}
        self.records = restored
        self._update_keys = update_keys
        self.update_audit = json.loads(json.dumps(payload.get("update_audit", []), sort_keys=True))

    def digest(self) -> str:
        return _hash_json(self.snapshot())


__all__ = [
    "FEATURE_DIM",
    "LearnedIdentityMemoryBank",
    "LearnedIdentityRecord",
    "SCHEMA_VERSION",
    "UPDATE_SCHEMA_VERSION",
    "sha256_file",
]
