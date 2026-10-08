#!/usr/bin/env python3
"""M1 execution/export boundary; runtime sees no offline labels or base tape."""
from __future__ import annotations
import argparse
import time

from sam3_intermot.association.causal_identity_tracker import CausalIdentityTracker
from sam3_intermot.association.identity_authority import AdapterEnsemble, AuthorityConfig, AuthorityController
from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank
from scripts.n72r20r4_common import *


def make_bank() -> LearnedIdentityMemoryBank:
    return LearnedIdentityMemoryBank.from_checkpoint(MEMORY_CHECKPOINT, encoder_sha256=ENCODER_SHA, expected_encoder_sha256=ENCODER_SHA, expected_checkpoint_sha256=MEMORY_SHA)


def run_rollout(sequence: str, *, config: AuthorityConfig, split: str = "train", adapter: AdapterEnsemble | None = None, controller: AuthorityController | None = None, frames: list | None = None, event: dict | None = None, reference: list[dict] | None = None) -> tuple[list[dict], dict]:
    frames = load_frames(sequence, split) if frames is None else frames
    event = events(split)[sequence] if event is None else event
    tracker = CausalIdentityTracker(config=config, event=event, adapter=adapter, bank=make_bank(), controller=controller)
    started = time.perf_counter()
    encoded = None
    offsets = [0]
    for _, rows in frames:
        offsets.append(offsets[-1] + len(rows))
    if adapter is not None:
        x = np.stack([r["feature"] for _, rows in frames for r in rows])
        # Stateless candidate projection is cached in RAM. Frame decisions
        # access only the slice belonging to the current observed frame.
        encoded = [np.empty_like(x) for _ in adapter.models]
        for start in range(0, len(x), 2048):
            for i, chunk in enumerate(adapter.encode_candidates(x[start:start+2048])):
                encoded[i][start:start+len(chunk)] = chunk
    output = []
    for payload, rows in frames:
        frame = int(payload["frame"])
        enc = None if encoded is None else [x[offsets[frame]:offsets[frame+1]] for x in encoded]
        baseline_solver = None
        if reference is not None:
            baseline_solver = {"public_assignments": [{"public_id": int(p), "candidate_uid": uid} for p, uid in reference[frame]["assignments"].items()]}
        decision = tracker.step(rows, frame, encoded_candidates=enc, baseline_solver=baseline_solver)
        output.append({k: v for k, v in decision.items() if k not in {"base_matrix", "fused_matrix", "solver"}})
    elapsed = time.perf_counter() - started
    return output, {"sequence": sequence, "split": split, "frames": len(frames), "seconds": elapsed, "fps": len(frames)/max(elapsed, 1e-9), "adapter_calls": tracker.adapter_calls, "model_manifest": [] if adapter is None else adapter.manifest, "runtime_future_gt_used": False, "runtime_gt_read": False, "score_before_update": True, "candidate_source": str(DEV_ROOT if split == "train" else VAL_ROOT), "configuration": config.to_dict(), "pre_event_frames": int(event["event_frame"]), "pre_event_outputs": sum(len(r["outputs"]) for r in output if r["frame"] < int(event["event_frame"])), "code_sha256": code_manifest()}


def trajectory_text(trace: list[dict]) -> str:
    lines = []
    for r in trace:
        ids = set()
        uids = set()
        for row in sorted(r["outputs"], key=lambda row: row["public_id"]):
            if row["public_id"] in ids or row["candidate_uid"] in uids:
                raise RuntimeError("MOT export duplicate candidate/public ID")
            ids.add(row["public_id"]); uids.add(row["candidate_uid"])
            x1,y1,x2,y2 = row["box_xyxy"]
            if x2 <= x1 or y2 <= y1:
                raise RuntimeError("MOT export invalid box")
            lines.append(f'{r["frame"]+1},{row["public_id"]},{x1:.4f},{y1:.4f},{max(1.0,x2-x1):.4f},{max(1.0,y2-y1):.4f},{row["confidence"]:.6f},-1,-1,-1\n')
    return "".join(lines)


def export_run(name: str, sequence: str, trace: list[dict], profile: dict, *, group: str = "dev") -> dict:
    check_storage(reserve_gib=0.02)
    directory = ASSETS / group / "trackers" / name / "data"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sequence}.txt"
    payload = trajectory_text(trace)
    if path.exists() and path.read_text() != payload:
        raise FileExistsError(f"refusing to overwrite changed completed trajectory: {path}")
    path.write_text(payload)
    trace_path = ASSETS / group / "traces" / name / f"{sequence}.jsonl.zst"
    if not trace_path.exists():
        write_zstd(trace_path, trace)
    record = {**profile, "trajectory_path": path, "trajectory_sha256": sha256(path), "trajectory_rows": len(payload.splitlines()), "trace_path": trace_path, "trace_sha256": sha256(trace_path), "name": name}
    write_json(ASSETS / group / "manifests" / name / f"{sequence}.json", record)
    return plain(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=SEQUENCES[0])
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--lifecycle", choices=["dynamic", "legacy_fixed"], default="dynamic")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.split == "val" and not (OUT / "val/FROZEN_POLICY.json").exists():
        raise ValueError("VAL requires frozen policy before access")
    trace, profile = run_rollout(args.sequence, config=AuthorityConfig(lifecycle=args.lifecycle), split=args.split)
    print(json.dumps(export_run("BASELINE_CAUSAL", args.sequence, trace, profile, group=args.split), sort_keys=True))
