"""N10 online association rollout: AUTO inference + optional HUMAN state interventions."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.nn as nn

from sam3_intermot.association.human_intervention import (
    HumanEventDetector,
    HumanFeatureExtractor,
    apply_intervention,
    select_events,
)
from sam3_intermot.association.state_manager import StateManager, StateManagerConfig
from sam3_intermot.interaction.simulator import GTFrame


class N10Rollout:
    def __init__(
        self,
        sequence: str,
        num_frames: int,
        gt_frames: Dict[int, GTFrame],
        model: Optional[nn.Module],
        manager_cfg: StateManagerConfig,
        mode: str = "auto",  # auto | human
        budget: int = 0,
        apply_interventions: bool = True,
        seq_dir: Optional[Path] = None,
        feature_extractor: Optional[HumanFeatureExtractor] = None,
    ) -> None:
        self.sequence = sequence
        self.num_frames = num_frames
        self.gt_frames = gt_frames
        self.model = model
        self.mode = mode
        self.budget = budget
        self.apply_interventions = apply_interventions
        self.seq_dir = seq_dir
        self.extractor = feature_extractor
        self.manager = StateManager(manager_cfg)
        self.detector = HumanEventDetector(sequence, num_frames)
        self.pre_rows: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        self.post_rows: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        self.verified_errors: List[dict] = []
        self.interaction_events: List[dict] = []
        self.intervention_log: List[dict] = []
        self.state_hashes: List[dict] = []
        self.accepted_count = 0
        self.invariant_violations: List[str] = []

    def run(self, obs_by_frame: Dict[int, List[dict]]) -> None:
        for f in range(self.num_frames):
            self._frame(f, obs_by_frame.get(f, []))

    def _frame(self, f: int, obs_list: List[dict]) -> None:
        pre = self.manager.rollout_frame(f, obs_list, self.model)
        self.pre_rows[f] = [(int(pid), np.asarray(box, dtype=float).copy()) for pid, box in pre]
        gt = self.gt_frames.get(f, GTFrame())
        events = self.detector.detect(f, self.pre_rows[f], gt)
        self.verified_errors.extend(events)
        post = [(int(pid), np.asarray(box, dtype=float).copy()) for pid, box in pre]
        if self.mode == "human" and self.accepted_count < self.budget:
            chosen = select_events(events, self.budget - self.accepted_count)
            for ev in chosen:
                ev["accepted"] = True
                ev["budget_available"] = True
                self.accepted_count += 1
                self.interaction_events.append(ev)
                if self.apply_interventions:
                    rec = apply_intervention(
                        self.manager,
                        ev,
                        f,
                        obs_list,
                        post,
                        self.extractor,
                        self.seq_dir,
                    )
                    rec["accepted"] = True
                    rec["event_type"] = ev["event_type"]
                    rec["canonical_public_id"] = ev.get("canonical_public_id")
                    rec["dataset_gt_id"] = ev.get("dataset_gt_id")
                    self.intervention_log.append(rec)
                    # The candidate audit was captured before this spatial
                    # transaction.  Attach only supplied human-event
                    # provenance; never rewrite the current-frame scores or
                    # expose future GT to the replay tape.
                    self.manager.annotate_human_event(f, ev, rec)
                    if rec["applied"]:
                        self.manager.mark_scope(rec.get("scope_pids", []), f)
                        # keep the N8 observer canonical in sync with the
                        # corrected identity (mirrors N8 _apply_one semantics)
                        gid = ev.get("dataset_gt_id")
                        pid = ev.get("public_mot_id") or rec.get("canonical_pid")
                        if gid is not None and pid is not None:
                            mem = self.detector.observer.memory.records.get(gid)
                            if mem is not None:
                                mem["canonical_public_id"] = int(pid)
                                mem["last_correct_public_id"] = int(pid)
                        for remove_pid, box in rec["rebinds"]:
                            post = [
                                (p, b)
                                for p, b in post
                                if not (int(p) == int(remove_pid) and np.allclose(b, box, atol=1e-3))
                            ]
                        for add_pid, box in rec["adds"]:
                            post.append((int(add_pid), np.asarray(box, dtype=float).copy()))
                    post = sorted(dict(post).items(), key=lambda kv: kv[0])
        self.post_rows[f] = post
        self.state_hashes.append(
            {
                "frame": f,
                "sequence": self.sequence,
                "state_hash": self.state_hash(),
                "accepted_in_frame": sum(
                    1 for e in self.interaction_events if e["frame"] == f + 1
                ),
                "n_states": len(self.manager.states),
            }
        )

    def state_hash(self) -> str:
        payload = {
            "next_pid": self.manager.next_pid,
            "states": {
                str(pid): st.to_dict()
                for pid, st in sorted(self.manager.states.items())
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def gt_audit(self) -> dict:
        return self.detector.observer.gt_audit

    def summary(self) -> dict:
        by_type: Dict[str, int] = {}
        for e in self.verified_errors:
            by_type[e["event_type"]] = by_type.get(e["event_type"], 0) + 1
        return {
            "sequence": self.sequence,
            "mode": self.mode,
            "budget": self.budget,
            "num_frames": self.num_frames,
            "accepted_count": self.accepted_count,
            "events_by_type": by_type,
            "state": self.manager.state_summary(),
            "invariant_violations": self.invariant_violations,
            "gt_audit": self.gt_audit(),
            "n_interventions_applied": sum(1 for r in self.intervention_log if r["applied"]),
        }
