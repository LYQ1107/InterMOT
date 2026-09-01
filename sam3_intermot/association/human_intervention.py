"""Human-conditioned identity state interventions (N10)."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from sam3_intermot.association.identity_state import IdentityState
from sam3_intermot.association.online_associator import box_iou
from sam3_intermot.association.state_manager import StateManager
from sam3_intermot.interaction.n8_temporal_observer import (
    ERROR_PRIORITY,
    N8Config,
    N8TemporalObserver,
)
from sam3_intermot.interaction.simulator import GTFrame


class HumanFeatureExtractor:
    """Lazy CPU OSNet extractor for human-provided boxes (miss interactions)."""

    feature_dim = 512

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = Path(checkpoint)
        self._ext = None

    def _ensure(self):
        if self._ext is None:
            from torchreid.reid.utils.feature_extractor import FeatureExtractor

            self._ext = FeatureExtractor(
                model_name="osnet_x1_0",
                model_path=str(self.checkpoint),
                image_size=(256, 128),
                device="cpu",
                verbose=False,
            )
        return self._ext

    def extract(self, seq_dir: Path, frame: int, box: np.ndarray) -> np.ndarray:
        from PIL import Image

        ext = self._ensure()
        img = Image.open(seq_dir / f"{frame + 1:08d}.jpg").convert("RGB")
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(self.feature_dim, dtype=np.float32)
        crop = np.asarray(img.crop((x1, y1, x2, y2)), dtype=np.uint8)
        feat = ext([crop]).cpu().numpy()[0].astype(np.float32)
        n = float(np.linalg.norm(feat))
        return feat / n if n > 1e-6 else np.zeros(self.feature_dim, dtype=np.float32)

    def extract_mask(
        self,
        seq_dir: Path,
        frame: int,
        box: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Extract a feature from a supplied human mask inside the human box.

        The mask is an input supplied by the interaction protocol.  It is not
        taken from a native track and is never inferred here.  Full-frame and
        box-crop masks are both accepted so callers can pass either protocol
        representation without changing the evidence boundary.
        """
        from PIL import Image

        ext = self._ensure()
        img = Image.open(seq_dir / f"{frame + 1:08d}.jpg").convert("RGB")
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(self.feature_dim, dtype=np.float32)
        crop = np.asarray(img.crop((x1, y1, x2, y2)), dtype=np.uint8)
        mask_array = np.asarray(mask)
        if mask_array.ndim > 2:
            mask_array = np.squeeze(mask_array)
        if mask_array.ndim != 2:
            raise ValueError("human mask must be a 2-D array")
        if mask_array.shape == (img.height, img.width):
            mask_array = mask_array[y1:y2, x1:x2]
        elif mask_array.shape != (y2 - y1, x2 - x1):
            raise ValueError(
                f"human mask shape {mask_array.shape} does not match frame/crop "
                f"{(img.height, img.width)}/{(y2 - y1, x2 - x1)}"
            )
        mask_array = np.asarray(mask_array > 0, dtype=bool)
        if not np.any(mask_array):
            return np.zeros(self.feature_dim, dtype=np.float32)
        masked_crop = crop.copy()
        masked_crop[~mask_array] = 0
        feat = ext([masked_crop]).cpu().numpy()[0].astype(np.float32)
        n = float(np.linalg.norm(feat))
        return feat / n if n > 1e-6 else np.zeros(self.feature_dim, dtype=np.float32)


class HumanEventDetector:
    """N8 verified-error detector driven by the N10 pre output rows."""

    def __init__(self, sequence: str, num_frames: int) -> None:
        self.seq = sequence
        self.num_frames = num_frames
        self.observer = N8TemporalObserver(
            {},
            {},
            num_frames,
            N8Config(budget=0, sequence=sequence),
            sequence=sequence,
        )

    def detect(self, frame: int, pre_rows: List[Tuple[int, np.ndarray]], gt: GTFrame) -> List[dict]:
        raw = [(pid, np.asarray(box, dtype=float)) for pid, box in pre_rows]
        return self.observer._detect_errors(frame, raw, gt)


def find_obs_by_box(obs_list: List[dict], box: np.ndarray, thr: float = 0.3) -> Optional[dict]:
    if box is None:
        return None
    best, best_iou = None, thr
    for o in obs_list:
        iou = box_iou(o["box"], np.asarray(box, dtype=float))
        if iou > best_iou:
            best, best_iou = o, iou
    return best


def human_evidence(
    event: dict,
    frame: int,
    extractor: Optional[HumanFeatureExtractor],
    seq_dir: Optional[Path],
    fallback_obs: Optional[dict] = None,
) -> dict:
    """Build appearance evidence from the explicitly supplied human ROI.

    ``fallback_obs`` is accepted only for spatial/native bookkeeping.  Its
    feature is deliberately never read: a missing human extractor therefore
    produces an auditable unavailable write instead of a machine-feature
    substitute.
    """
    human_box_value = event.get("gt_box", event.get("box"))
    box_value = human_box_value
    if box_value is None and fallback_obs is not None:
        box_value = fallback_obs.get("box")
    box = (
        np.asarray(box_value, dtype=float).reshape(-1)
        if box_value is not None
        else np.zeros(4, dtype=float)
    )
    feature_dim = int(getattr(extractor, "feature_dim", 512)) if extractor is not None else 512
    zero = np.zeros(feature_dim, dtype=np.float32)
    native_tid = -1 if fallback_obs is None else int(fallback_obs.get("native_tid", -1))
    quality = event.get("quality", 1.0)
    try:
        quality = float(quality)
    except (TypeError, ValueError):
        quality = 0.0
    quality = float(np.clip(quality if np.isfinite(quality) else 0.0, 0.0, 1.0))
    event_id = event.get("write_event_id") or event.get("event_id") or event.get("id")
    has_mask = event.get("mask") is not None
    result = {
        "obs_id": -1,
        "box": box,
        "conf": 1.0,
        "feat": zero,
        "has_feat": 0.0,
        "native_tid": native_tid,
        "native_age": 0.0,
        "human_obs": 1.0,
        "quality": quality,
        "source_frame": int(frame),
        "write_event_id": None if event_id is None else str(event_id),
        "source": "human_roi_mask" if has_mask else "human_roi_box",
        "status": "HUMAN_FEATURE_NOT_AVAILABLE",
    }
    # A fallback observation may provide a box for the already-authorized
    # spatial transaction, but it is not a human ROI.  This branch is what
    # keeps the second side of a swap unavailable when ``other_gt_box`` is
    # absent.
    if extractor is None or seq_dir is None or human_box_value is None or box.size != 4:
        return result
    try:
        if has_mask:
            extract_mask = getattr(extractor, "extract_mask", None)
            if not callable(extract_mask):
                raise RuntimeError("extractor_has_no_supported_mask_interface")
            feat = extract_mask(Path(seq_dir), int(frame), box, event["mask"])
        else:
            feat = extractor.extract(Path(seq_dir), int(frame), box)
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        if feat.size != feature_dim or not np.all(np.isfinite(feat)):
            raise ValueError(f"human feature has invalid shape/values: {feat.shape}")
        norm = float(np.linalg.norm(feat))
        if norm <= 1e-6:
            result["status"] = "HUMAN_FEATURE_EXTRACTION_FAILED"
            result["failure_reason"] = "zero_feature"
            return result
        result["feat"] = feat / norm
        result["has_feat"] = 1.0
        result["status"] = "PASS"
    except Exception as exc:  # preserve the spatial transaction; audit the source failure
        result["status"] = "HUMAN_FEATURE_EXTRACTION_FAILED"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return result


def apply_intervention(
    manager: StateManager,
    event: dict,
    frame: int,
    obs_list: List[dict],
    pre_rows: List[Tuple[int, np.ndarray]],
    extractor: Optional[HumanFeatureExtractor],
    seq_dir: Optional[Path],
) -> dict:
    """Apply one accepted human intervention to the identity state."""
    action = event["action_type"]
    event_id = event.get("event_id") or event.get("id")
    record = {
        "frame": frame + 1,
        "event_type": event["event_type"],
        "event_id": None if event_id is None else str(event_id),
        "applied": False,
        "rebinds": [],
        "adds": [],
        "scope_pids": [],
    }

    def write_memory(pid: int, evidence: dict, competitors: List[dict]) -> bool:
        """Write only the independent human ROI evidence."""
        status = str(evidence.get("status", "HUMAN_FEATURE_NOT_AVAILABLE"))
        ledger = {
            "status": status,
            "pid": int(pid),
            "frame": int(frame),
            "source_frame": int(evidence.get("source_frame", frame)),
            "age": max(0, int(frame) - int(evidence.get("source_frame", frame))),
            "event_id": None if event_id is None else str(event_id),
            "source": evidence.get("source", "human_roi_box"),
            "quality": float(evidence.get("quality", 0.0)),
            "human_feature_available": bool(evidence.get("has_feat", 0.0)),
        }
        if not bool(evidence.get("has_feat", 0.0)):
            if evidence.get("failure_reason") is not None:
                ledger["failure_reason"] = str(evidence["failure_reason"])
            record.setdefault("appearance_memory", []).append(ledger)
            return False
        feat = np.asarray(evidence["feat"], dtype=np.float32)
        comp = [
            o["feat"]
            for o in competitors
            if bool(o.get("has_feat", 0.0))
            and np.linalg.norm(np.asarray(o.get("feat", 0), dtype=np.float32)) > 1e-6
        ]
        ok = manager.update_human_appearance(
            int(pid), frame, feat, quality=float(evidence.get("quality", 1.0)),
            competing_embeddings=comp, write_event_id=str(event_id) if event_id is not None else None,
        )
        ledger["status"] = "PASS" if ok else ("DISABLED" if not manager.cfg.use_appearance_memory else "WRITE_REJECTED")
        record.setdefault("appearance_memory", []).append(ledger)
        return ok
    if action == "AUTHORITATIVE_REASSIGN":
        canonical = event["canonical_public_id"]
        wrong = event["current_public_id"]
        obs = find_obs_by_box(obs_list, event["gt_box"])
        if canonical is None or int(canonical) < 1 or obs is None:
            return record
        st = manager.get_or_create(int(canonical), obs, frame)
        st.add_anchor(obs["feat"], authority=1.0, frame=frame)
        st.add_positive(int(obs["native_tid"]), manager.native_expiry(frame))
        write_memory(
            int(canonical),
            human_evidence(event, frame, extractor, seq_dir, fallback_obs=obs),
            [o for o in obs_list if o is not obs],
        )
        wrong_st = manager.states.get(int(wrong))
        if wrong_st is not None:
            wrong_st.add_negative(int(obs["native_tid"]), manager.native_expiry(frame))
        record["rebinds"].append((int(wrong), obs["box"].copy()))
        record["adds"].append((int(canonical), obs["box"].copy()))
        record["scope_pids"] = [int(canonical), int(wrong)]
        record.update({"canonical_pid": int(canonical), "obs_native_tid": int(obs["native_tid"]), "applied": True})
    elif action == "ATOMIC_ID_SWAP":
        c_a = event["canonical_public_id"]
        c_b = event["other_canonical_public_id"]
        obs_a = find_obs_by_box(obs_list, event["gt_box"])
        box_b = None
        for p, b in pre_rows:
            if int(p) == int(event.get("other_auto_tid", -1)):
                box_b = b
                break
        obs_b = find_obs_by_box(obs_list, box_b) if box_b is not None else None
        if (
            obs_a is None
            or obs_b is None
            or c_a is None
            or c_b is None
            or int(c_a) < 1
            or int(c_b) < 1
        ):
            return record
        st_a = manager.get_or_create(int(c_a), obs_a, frame)
        st_b = manager.get_or_create(int(c_b), obs_b, frame)
        st_a.add_anchor(obs_a["feat"], authority=1.0, frame=frame)
        st_b.add_anchor(obs_b["feat"], authority=1.0, frame=frame)
        st_a.add_positive(int(obs_a["native_tid"]), manager.native_expiry(frame))
        st_b.add_positive(int(obs_b["native_tid"]), manager.native_expiry(frame))
        st_a.add_negative(int(obs_b["native_tid"]), manager.native_expiry(frame))
        st_b.add_negative(int(obs_a["native_tid"]), manager.native_expiry(frame))
        write_memory(
            int(c_a),
            human_evidence(event, frame, extractor, seq_dir, fallback_obs=obs_a),
            [o for o in obs_list if o is not obs_a],
        )
        # The event carries only the first GT box; do not claim a human ROI
        # for the second swapped identity unless an explicit box is supplied.
        write_memory(
            int(c_b),
            human_evidence(
                {
                    "gt_box": event.get("other_gt_box"),
                    "mask": event.get("other_mask"),
                    "quality": event.get("other_quality", event.get("quality", 1.0)),
                    "event_id": event_id,
                },
                frame,
                extractor,
                seq_dir,
                fallback_obs=obs_b,
            ),
            [o for o in obs_list if o is not obs_b],
        )
        record["rebinds"].append((int(event["current_public_id"]), obs_a["box"].copy()))
        record["adds"].append((int(c_a), obs_a["box"].copy()))
        record["rebinds"].append((int(event["other_auto_tid"]), obs_b["box"].copy()))
        record["adds"].append((int(c_b), obs_b["box"].copy()))
        record["scope_pids"] = [
            int(c_a),
            int(c_b),
            int(event["current_public_id"]),
            int(event["other_auto_tid"]),
        ]
        record.update(
            {
                "pid_a": int(c_a),
                "pid_b": int(c_b),
                "obs_a_native": int(obs_a["native_tid"]),
                "obs_b_native": int(obs_b["native_tid"]),
                "applied": True,
            }
        )
    elif action == "RECOVER_IDENTITY":
        canonical = event["canonical_public_id"]
        if canonical is None or int(canonical) < 1:
            canonical = int(manager._new_pid())
            event["public_mot_id"] = canonical
            event["canonical_public_id"] = canonical
        obs = _human_obs(event, frame, extractor, seq_dir)
        st = manager.get_or_create(int(canonical), obs, frame)
        st.add_anchor(obs["feat"], authority=1.0, frame=frame)
        st.add_positive(int(obs["native_tid"]), manager.native_expiry(frame))
        write_memory(int(canonical), obs, [o for o in obs_list if o is not obs])
        record["adds"].append((int(canonical), obs["box"].copy()))
        record["scope_pids"] = [int(canonical)]
        record.update({"canonical_pid": int(canonical), "new_pid": int(canonical), "applied": True})
    elif action == "ADD_NEW_IDENTITY":
        obs = _human_obs(event, frame, extractor, seq_dir)
        pid = manager._new_pid()
        st = IdentityState(pid, obs["feat"], obs["box"], frame, obs["native_tid"])
        st.add_anchor(obs["feat"], authority=1.0, frame=frame)
        st.add_positive(int(obs["native_tid"]), manager.native_expiry(frame))
        manager.states[pid] = st
        write_memory(int(pid), obs, [o for o in obs_list if o is not obs])
        record["adds"].append((pid, obs["box"].copy()))
        record["scope_pids"] = [pid]
        event["public_mot_id"] = pid
        event["canonical_public_id"] = pid
        record.update({"new_pid": pid, "applied": True})
    return record


def _human_obs(
    event: dict,
    frame: int,
    extractor: Optional[HumanFeatureExtractor],
    seq_dir: Optional[Path],
) -> dict:
    return human_evidence(event, frame, extractor, seq_dir)


def select_events(events: List[dict], budget_left: int) -> List[dict]:
    required = [e for e in events if e.get("interaction_required")]
    required.sort(
        key=lambda e: (
            ERROR_PRIORITY.get(e["event_type"], 99),
            e.get("user_identity_id") or 0,
        )
    )
    return required[:budget_left]


def merge_rows(
    pre_rows: List[Tuple[int, np.ndarray]],
    new_rows: List[Tuple[int, np.ndarray]],
) -> List[Tuple[int, np.ndarray]]:
    out: Dict[int, np.ndarray] = {}
    for pid, box in pre_rows:
        out[int(pid)] = np.asarray(box, dtype=float).copy()
    for pid, box in new_rows:
        out[int(pid)] = np.asarray(box, dtype=float).copy()
    return sorted(out.items(), key=lambda kv: kv[0])
