"""Fresh causal committed-write risk, separate from MOT association authority.

Current TARGET / verified OTHER / UNKNOWN correctness is not a future-safe
association reward. The bank features below are identical for offline sealed
teacher-state replay and actual own-memory deployment.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .acib_runtime import Evidence, unit, overlap
from .committed_identity_memory import CommittedIdentityMemory, MemoryCommitPolicy
from .intervention_features import FEATURE_NAMES, feature_vector, validate_runtime_rows

BANK_FEATURES = ("bank_size", "bank_drift", "current_bank_max_cosine", "current_bank_mean_cosine",
                 "bank_mean_confidence", "bank_latest_gap", "own_pending_confirmations",
                 "own_pending_cosine", "own_pending_iou", "own_pending_native_same")
INPUT_NAMES = FEATURE_NAMES + BANK_FEATURES


def write_input(features, anchor, bank, pending, frame, candidate):
    if set(features) != set(FEATURE_NAMES):
        raise ValueError("Only current causal joint features may enter the writer")
    validate_runtime_rows([candidate])
    vector = unit(candidate["feature"])
    if any(e.frame >= frame for e in bank) or pending is not None and pending["frame"] >= frame:
        raise ValueError("Current/future observations cannot be prior memory")
    agreement = 0. if pending is None else float(vector @ pending["feature"])
    motion = 0. if pending is None else overlap(candidate["box_xyxy"], pending["box"])
    consecutive = bool(pending is not None and pending["frame"] == frame - 1 and agreement >= .9 and motion >= .3)
    count = pending["count"] + 1 if consecutive else 1
    native = (candidate.get("native_scope"), candidate["native_tid"])
    if bank:
        mean = np.mean([e.embedding for e in bank], axis=0)
        drift = float(1. - np.dot(mean, anchor) / max(1.e-8, np.linalg.norm(mean)))
        cosines = [float(vector @ e.embedding) for e in bank]
    else:
        drift, cosines = 0., []
    values = dict(bank_size=len(bank), bank_drift=drift,
                  current_bank_max_cosine=max(cosines, default=-1.),
                  current_bank_mean_cosine=float(np.mean(cosines)) if cosines else -1.,
                  bank_mean_confidence=float(np.mean([e.confidence for e in bank])) if bank else 0.,
                  bank_latest_gap=frame - max(e.frame for e in bank) if bank else 1000,
                  own_pending_confirmations=count, own_pending_cosine=agreement, own_pending_iou=motion,
                  own_pending_native_same=float(pending is not None and pending["native"] == native))
    scales = dict(bank_size=8., bank_latest_gap=100., own_pending_confirmations=3.)
    extra = np.asarray([np.clip(values[k] / scales.get(k, 1.), -10., 10.) for k in BANK_FEATURES], np.float32)
    result = np.concatenate((feature_vector(features), extra))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite own-bank input")
    return result, values


class MemoryRiskHead(nn.Module):
    def __init__(self, family):
        super().__init__()
        if family not in ("LOGISTIC", "MLP"):
            raise ValueError("Unregistered fresh write-risk family")
        self.family = family
        self.net = (nn.Linear(len(INPUT_NAMES), 3) if family == "LOGISTIC" else
                    nn.Sequential(nn.Linear(len(INPUT_NAMES), 32), nn.Tanh(), nn.Linear(32, 3)))

    def forward(self, x):
        return self.net(x)


class FreshMemoryRiskPredictor:
    def __init__(self, checkpoint):
        saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if saved["schema"] != "N72R21R2_COMMITTED_WRITE_CURRENT_RISK_V1" or saved["input_names"] != list(INPUT_NAMES):
            raise ValueError("Fresh same-state committed-write checkpoint required")
        if saved["association_authority"] or saved["selection"]["status"] == "UNCALIBRATED_NOT_DEPLOYABLE":
            raise ValueError("No uncalibrated/association authority for M-A writer")
        self.model = MemoryRiskHead(saved["family"]).eval()
        self.model.load_state_dict(saved["model"], strict=True)
        self.model.requires_grad_(False)
        self.mean, self.scale = np.asarray(saved["FIT_mean"], np.float32), np.asarray(saved["FIT_scale"], np.float32)
        if (self.mean.shape != (len(INPUT_NAMES),) or self.scale.shape != self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0)):
            raise ValueError("Finite FIT-only bank normalizer required")
        self.selection = saved["selection"]

    def predict(self, vector):
        vector = np.asarray(vector, np.float32)
        if vector.shape != self.mean.shape or not np.isfinite(vector).all():
            raise ValueError("Exact finite own-bank feature axis required")
        with torch.inference_mode():
            p = torch.softmax(self.model(torch.from_numpy((vector - self.mean) / self.scale)[None])[0]
                              / self.selection["temperature"], dim=0).cpu().tolist()
        return dict(beneficial=p[0], harmful=p[1] + p[2], verified_OTHER=p[1], unknown=p[2],
                    semantics="CURRENT_COMMITTED_IDENTITY_CORRECTNESS_NOT_FUTURE_MOT_SAFETY")


class FreshRiskCommittedMemory(CommittedIdentityMemory):
    def __init__(self, model, anchor, token, *, predictor, selection=None):
        if not isinstance(predictor, FreshMemoryRiskPredictor):
            raise ValueError("Strictly loaded actual fresh writer required")
        super().__init__(model, anchor, token, write_policy=MemoryCommitPolicy(family="risk", capacity=8,
                         aggregation="attention", delay_frames=1), write_predictor=predictor)
        self.risk_selection = dict(predictor.selection if selection is None else selection)

    def observe_committed(self, frame, rows, committed_uid, features):
        validate_runtime_rows(rows)
        if frame != self.last_frame or self.last_commit_observation is not None and frame <= self.last_commit_observation:
            raise ValueError("One original current commit, one possible crop write")
        candidate = next((r for r in rows if str(r["candidate_uid"]) == committed_uid), None)
        if committed_uid is not None and candidate is None:
            raise ValueError("Committed write UID outside current axis")
        self.last_commit_observation = frame
        if candidate is None:
            self.write_pending = None
            return dict(accepted=False, candidate_uid=None, reasons=["COMMITTED_NONE"], rollback=False)
        vector, values = write_input(features, self.anchor, self.bank, self.write_pending, frame, candidate)
        prediction = self.write_predictor.predict(vector)
        crop = unit(candidate["feature"])
        self.write_pending = dict(frame=frame, count=values["own_pending_confirmations"], feature=crop.copy(),
                                  box=list(candidate["box_xyxy"]), native=(candidate.get("native_scope"), candidate["native_tid"]))
        point, reasons = self.risk_selection, []
        if point["status"] == "CALIBRATION_ABSTAIN":
            reasons.append("NO_NONVACUOUS_INNER_CURRENT_WRITE_POINT")
        if prediction["beneficial"] < point["probability_min"]:
            reasons.append("LOW_CURRENT_WRITE_CORRECTNESS")
        if prediction["unknown"] > point["unknown_max"]:
            reasons.append("UNKNOWN_UNVERIFIED_WRITE_ABSTAIN")
        accepted = not reasons
        if accepted:
            self.write_snapshots.append(tuple(self.bank))
            self.write_snapshots = self.write_snapshots[-4:]
            crop.setflags(write=False)
            self.bank.append(Evidence(crop, self.recording, int(frame), self.camera, frame / self.fps,
                float(features["proposed_probability"]), "ACTUAL_GLOBAL_COMMIT_R2_CURRENT_RISK",
                float(features["quality"]), float(crop @ self.anchor), str(committed_uid)))
            self.bank = self.bank[-self.capacity:]
        if hashlib.sha256(self.anchor.tobytes()).hexdigest() != self.anchor_sha:
            raise RuntimeError("Immutable sole human anchor changed")
        return dict(accepted=accepted, candidate_uid=str(committed_uid) if accepted else None,
                    reasons=reasons or ["FRESH_CURRENT_COMMITTED_WRITE_RISK_ACCEPTED"],
                    confirmations=values["own_pending_confirmations"], prediction=prediction, rollback=False,
                    trusted_size=len(self.bank), writer_input_vector=vector.tolist(), own_bank_features=values,
                    pending_used_for_identity_score=False, proposed_UID_not_write_authority=True,
                    rule_probability_anchor_motion_native_delay_gates_NOT_silently_added=True,
                    current_correctness_NOT_future_safe_memory_or_MOT_guarantee=True)

    def snapshot(self):
        return super().snapshot() | {"R2_current_risk_selection": self.risk_selection,
                                    "M_A_association_authority": False,
                                    "hard_multicue_rule_gate_NOT_active": True}
