#!/usr/bin/env python
"""Run one N5 continuous-observer protocol on one sequence.

Environment:
  N5_PROTOCOL        p0|p1|p2|p3|p4
  N5_BUDGET          accepted-event budget (P4 only)
  N5_SEQ             dancetrack sequence name
  N5_OUT_DIR         per-job output directory
  N5_FRAMES          optional frame limit for gate runs
  N5_P0_SOURCE       canonical P0 mot dir used by P1
  N5_SKIP_TRACKEVAL  1 to skip per-job TrackEval
"""

import json
import gc
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from sam3_intermot.backend.sam3_backend import Sam3Backend
from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.evaluation.mot_export import export_mot_file, validate_mot_file
from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.identity.registry import ObjectIdentityRegistry
from sam3_intermot.interaction.continuous_observer import (
    CommandType,
    ContinuousObserverDriver,
    GTUsageAudit,
    N5Config,
    P1OfflineDriver,
    read_mot_rows,
)
from sam3_intermot.tracking.track_manager import TrackManager


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = os.environ.get("N5_PROTOCOL", "p3")
BUDGET = int(os.environ.get("N5_BUDGET", "0"))
SEQ = os.environ["N5_SEQ"]
OUT_DIR = Path(os.environ.get("N5_OUT_DIR", ROOT / "outputs" / "n5" / "tmp"))
FRAME_LIMIT = int(os.environ.get("N5_FRAMES", "0")) or None
P0_SOURCE = Path(
    os.environ.get(
        "N5_P0_SOURCE",
        ROOT / "outputs/n5/integrity/canonical_mot_results/b0",
    )
)
SKIP_TRACKEVAL = os.environ.get("N5_SKIP_TRACKEVAL", "0") == "1"


def _p0(config) -> None:
    (OUT_DIR / "pre_mot").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "post_mot").mkdir(parents=True, exist_ok=True)
    src = P0_SOURCE / f"{SEQ}.txt"
    shutil.copy2(src, OUT_DIR / "pre_mot" / f"{SEQ}.txt")
    shutil.copy2(src, OUT_DIR / "post_mot" / f"{SEQ}.txt")
    audit = GTUsageAudit()
    _write_summary(
        {
            "sequence": SEQ,
            "protocol": "p0",
            "budget": 0,
            "num_frames": len(read_mot_rows(src)),
            "total_commands": 0,
            "accepted_commands": 0,
            "by_type": {},
            "pre_rows": sum(len(v) for v in read_mot_rows(src).values()),
            "post_rows": sum(len(v) for v in read_mot_rows(src).values()),
            "invariant_violations": [],
        },
        audit,
    )


def _p1(config, dataset) -> None:
    (OUT_DIR / "pre_mot").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "post_mot").mkdir(parents=True, exist_ok=True)
    pre_path = P0_SOURCE / f"{SEQ}.txt"
    shutil.copy2(pre_path, OUT_DIR / "pre_mot" / f"{SEQ}.txt")
    pre_rows = read_mot_rows(pre_path)
    gt_frames = dataset.load_gt(SEQ)
    num_frames = dataset.num_frames(SEQ)
    if FRAME_LIMIT is not None:
        num_frames = min(num_frames, FRAME_LIMIT)
        pre_rows = {f: v for f, v in pre_rows.items() if f < num_frames}
        gt_frames = {f: v for f, v in gt_frames.items() if f < num_frames}
    driver = P1OfflineDriver(
        pre_rows, gt_frames, sequence=SEQ, num_frames=num_frames
    )
    post = driver.run()
    _export_rows(OUT_DIR / "post_mot" / f"{SEQ}.txt", post)
    with (OUT_DIR / "events.jsonl").open("w", encoding="utf-8") as f:
        for e in driver.events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    audit = GTUsageAudit()
    audit.gt_read_current_after_prediction = num_frames
    audit.gt_used_for_user_observation = num_frames
    audit.gt_used_for_command_generation = num_frames
    audit.gt_used_for_offline_scoring = num_frames
    by_type = {}
    for e in driver.events:
        by_type[e["action_type"]] = by_type.get(e["action_type"], 0) + 1
    _write_summary(
        {
            "sequence": SEQ,
            "protocol": "p1",
            "budget": 0,
            "num_frames": num_frames,
            "total_commands": driver.commands_total,
            "accepted_commands": driver.commands_total,
            "by_type": by_type,
            "pre_rows": sum(len(v) for v in pre_rows.values()),
            "post_rows": sum(len(v) for v in post.values()),
            "invariant_violations": [],
        },
        audit,
    )


def _stateful(config, dataset, seq: str) -> None:
    backend = Sam3Backend(
        checkpoint_path=config["backend"]["checkpoint_path"],
        max_num_objects=config["backend"]["max_num_objects"],
        multiplex_count=config["backend"]["multiplex_count"],
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
    )
    manager = TrackManager()
    lineages = IdentityLineageRegistry()
    registry = ObjectIdentityRegistry(manager, lineages)
    started = time.time()
    try:
        data_root = Path(config["dataset"]["root"])
        video = data_root / config["dataset"]["split"] / seq / "img1"
        backend.start_video(str(video))
        num_frames = dataset.num_frames(seq)
        gt_frames = dataset.load_gt(seq)
        if FRAME_LIMIT is not None:
            num_frames = min(num_frames, FRAME_LIMIT)
            gt_frames = {f: v for f, v in gt_frames.items() if f < num_frames}
        n5_cfg = N5Config(
            protocol=PROTOCOL,
            budget=BUDGET,
            correct_localization=PROTOCOL == "p2",
            correct_false_track=PROTOCOL == "p2",
            stateful=True,
        )
        driver = ContinuousObserverDriver(
            backend,
            manager,
            lineages,
            registry,
            n5_cfg,
            num_frames,
            gt_frames,
            sequence=seq,
            video_source=str(video),
        )
        summary = driver.run()
        (OUT_DIR / "pre_mot").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "post_mot").mkdir(parents=True, exist_ok=True)
        export_mot_file(OUT_DIR / "pre_mot" / f"{seq}.txt", driver.pre_rows)
        export_mot_file(OUT_DIR / "post_mot" / f"{seq}.txt", driver.post_rows)
        pre_viol = validate_mot_file(
            OUT_DIR / "pre_mot" / f"{seq}.txt", num_frames=num_frames
        )
        post_viol = validate_mot_file(
            OUT_DIR / "post_mot" / f"{seq}.txt", num_frames=num_frames
        )
        with (OUT_DIR / "events.jsonl").open("w", encoding="utf-8") as f:
            for e in driver.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        (OUT_DIR / "invariant_violations.csv").write_text(
            "source,violation\n"
            + "".join(
                f"pre,{v}\n" for v in pre_viol + summary.invariant_violations
            )
            + "".join(f"post,{v}\n" for v in post_viol),
            encoding="utf-8",
        )
        _write_summary(
            {
                "sequence": seq,
                "protocol": PROTOCOL,
                "budget": BUDGET,
                "num_frames": num_frames,
                "total_commands": summary.total_commands,
                "accepted_commands": summary.accepted_commands,
                "by_type": summary.by_type,
                "rejected": summary.rejected,
                "rolled_back": summary.rolled_back,
                "pre_rows": summary.pre_rows,
                "post_rows": summary.post_rows,
                "invariant_violations": summary.invariant_violations,
                "pre_mot_violations": pre_viol,
                "post_mot_violations": post_viol,
            },
            driver.gt_access.audit,
        )
    finally:
        backend.close()
        del backend
        gc.collect()
        torch.cuda.empty_cache()


def _stateful_backbone(config, dataset, seq: str) -> None:
    """Stateful protocols on the frozen P0 backbone (no SAM re-propagation)."""
    from sam3_intermot.backend.output_types import PromptObjectObservation

    manager = TrackManager()
    lineages = IdentityLineageRegistry()
    registry = ObjectIdentityRegistry(manager, lineages)
    num_frames = dataset.num_frames(seq)
    gt_frames = dataset.load_gt(seq)
    if FRAME_LIMIT is not None:
        num_frames = min(num_frames, FRAME_LIMIT)
        gt_frames = {f: v for f, v in gt_frames.items() if f < num_frames}
    backbone: dict = {}
    pre_rows = read_mot_rows(P0_SOURCE / f"{seq}.txt")
    for f in range(num_frames):
        for tid, box in pre_rows.get(f, []):
            backbone.setdefault(f, []).append(
                PromptObjectObservation(
                    frame_idx=f,
                    sam_object_id=tid,
                    mask=np.zeros((1, 1), dtype=bool),
                    box_xyxy=box,
                    confidence=1.0,
                    source="p0_frozen_backbone",
                )
            )
    n5_cfg = N5Config(
        protocol=PROTOCOL,
        budget=BUDGET,
        correct_localization=PROTOCOL == "p2",
        correct_false_track=PROTOCOL == "p2",
        stateful=True,
    )
    driver = ContinuousObserverDriver(
        None,
        manager,
        lineages,
        registry,
        n5_cfg,
        num_frames,
        gt_frames,
        sequence=seq,
        backbone=backbone,
    )
    summary = driver.run()
    (OUT_DIR / "pre_mot").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "post_mot").mkdir(parents=True, exist_ok=True)
    export_mot_file(OUT_DIR / "pre_mot" / f"{seq}.txt", driver.pre_rows)
    export_mot_file(OUT_DIR / "post_mot" / f"{seq}.txt", driver.post_rows)
    pre_viol = validate_mot_file(
        OUT_DIR / "pre_mot" / f"{seq}.txt", num_frames=num_frames
    )
    post_viol = validate_mot_file(
        OUT_DIR / "post_mot" / f"{seq}.txt", num_frames=num_frames
    )
    with (OUT_DIR / "events.jsonl").open("w", encoding="utf-8") as f:
        for e in driver.events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (OUT_DIR / "invariant_violations.csv").write_text(
        "source,violation\n"
        + "".join(f"pre,{v}\n" for v in pre_viol + summary.invariant_violations)
        + "".join(f"post,{v}\n" for v in post_viol),
        encoding="utf-8",
    )
    _write_summary(
        {
            "sequence": seq,
            "protocol": PROTOCOL,
            "budget": BUDGET,
            "num_frames": num_frames,
            "total_commands": summary.total_commands,
            "accepted_commands": summary.accepted_commands,
            "by_type": summary.by_type,
            "rejected": summary.rejected,
            "rolled_back": summary.rolled_back,
            "pre_rows": summary.pre_rows,
            "post_rows": summary.post_rows,
            "invariant_violations": summary.invariant_violations,
            "pre_mot_violations": pre_viol,
            "post_mot_violations": post_viol,
            "mode": "p0_backbone",
        },
        driver.gt_access.audit,
    )


def _export_rows(
    path: Path, rows: dict, frame_shift: int = 0
) -> None:
    """Export (tid, box, conf) rows to MOT format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for f in sorted(rows):
        for tid, box, conf in sorted(rows[f], key=lambda x: x[0]):
            x1, y1, x2, y2 = box
            lines.append(
                f"{f + 1},{tid},{x1:.2f},{y1:.2f},{max(0.0, x2 - x1):.2f},"
                f"{max(0.0, y2 - y1):.2f},{conf:.3f},-1,-1,-1"
            )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_summary(data: dict, audit: GTUsageAudit) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "gt_access_audit.json").write_text(
        json.dumps(audit.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "runtime.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "budget": BUDGET,
                "sequence": SEQ,
                "started": started_epoch,
                "finished": time.time(),
                "wall_seconds": time.time() - started_epoch,
                "gpu_used": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "gt_usage_ok.json").write_text(
        json.dumps({"ok": audit.ok()}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"sequence": SEQ, "protocol": PROTOCOL, "budget": BUDGET, **data}))


started_epoch = time.time()


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    dataset = DanceTrackDataset(
        str(Path(cfg["dataset"]["root"])), sequences=[SEQ], split=cfg["dataset"]["split"]
    )
    if PROTOCOL == "p0":
        _p0(cfg)
    elif PROTOCOL == "p1":
        _p1(cfg, dataset)
    elif os.environ.get("N5_P0_BACKBONE", "0") == "1":
        _stateful_backbone(cfg, dataset, SEQ)
    else:
        _stateful(cfg, dataset, SEQ)


if __name__ == "__main__":
    main()
