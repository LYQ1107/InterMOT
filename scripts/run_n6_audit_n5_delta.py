#!/usr/bin/env python
"""N6-A read-only reproduction of the N5 pre/post identity anomaly."""

import csv
import json
from pathlib import Path

import numpy as np

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.continuous_observer import match_boxes, read_mot_rows


ROOT = Path(".")
OUT = ROOT / "outputs/n6/audit"
SEQS = ["dancetrack0004", "dancetrack0005", "dancetrack0007"]
PROTOCOLS = ["p2", "p3"]
DS = DanceTrackDataset("/path/to/dancetrack", split="val")


def rows_to_obs(rows):
    return [(tid, np.asarray(box, dtype=float)) for tid, box in rows]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    summary = {}
    for seq in SEQS:
        gt = DS.load_gt(seq)
        gt_ids_all = sorted({gid for f in gt.values() for gid in f.gt_ids})
        seq_rows = []
        for proto in PROTOCOLS:
            base = ROOT / "outputs/n5" / (
                "p2_oracle_state_all" if proto == "p2" else "p3_continuous_id_miss"
            ) / seq
            pre = read_mot_rows(base / "pre_mot" / f"{seq}.txt")
            post = read_mot_rows(base / "post_mot" / f"{seq}.txt")
            stats = {
                "frames": 0,
                "id_changed_same_gt": 0,
                "post_only_duplicate_frame_id": 0,
                "post_public_id_two_boxes": 0,
                "post_gt_numeric_collision": 0,
                "post_gt_wrong_numeric_collision": 0,
                "gt_to_post_ids": {},
                "gt_to_pre_ids": {},
                "post_id_to_gts": {},
                "new_post_ids_total": 0,
            }
            for f in sorted(set(pre) | set(post) | set(gt)):
                stats["frames"] += 1
                pre_rows = pre.get(f, [])
                post_rows = post.get(f, [])
                g = gt.get(f)
                gt_boxes = [np.asarray(b, float) for b in g.boxes] if g else []
                gt_ids = g.gt_ids if g else []
                # duplicate (frame,id) within post
                ids = [tid for tid, _ in post_rows]
                if len(ids) != len(set(ids)):
                    stats["post_only_duplicate_frame_id"] += 1
                # same public id covering two boxes
                if len(ids) != len(set(ids)):
                    stats["post_public_id_two_boxes"] += 1
                pre_match = match_boxes(
                    gt_boxes, [b for _, b in pre_rows], 0.5
                )
                post_match = match_boxes(
                    gt_boxes, [b for _, b in post_rows], 0.5
                )
                pre_pi_to_tid = {pi: tid for pi, (tid, _) in enumerate(pre_rows)}
                post_pi_to_tid = {pi: tid for pi, (tid, _) in enumerate(post_rows)}
                for gi, pi, _iou in pre_match:
                    gid = gt_ids[gi]
                    stats["gt_to_pre_ids"].setdefault(gid, set()).add(pre_pi_to_tid[pi])
                post_ids_this_frame = set(ids)
                for gi, pi, _iou in post_match:
                    gid = gt_ids[gi]
                    pid = post_pi_to_tid[pi]
                    stats["gt_to_post_ids"].setdefault(gid, set()).add(pid)
                    stats["post_id_to_gts"].setdefault(pid, set()).add(gid)
                    if pid == gid:
                        stats["post_gt_numeric_collision"] += 1
                    if pid in gt_ids_all and pid != gid:
                        stats["post_gt_wrong_numeric_collision"] += 1
                # pre->post change for the same GT target
                for gi, pi, _ in post_match:
                    pre_pi = next(
                        (p for p, _p2, _i in pre_match if p == gi), None
                    )
                    if pre_pi is not None:
                        pre_id = pre_pi_to_tid.get(pre_pi)
                        post_id = post_pi_to_tid[pi]
                        if pre_id != post_id:
                            stats["id_changed_same_gt"] += 1
                            csv_rows.append(
                                {
                                    "sequence": seq,
                                    "protocol": proto,
                                    "frame": f + 1,
                                    "gt_id": gt_ids[gi],
                                    "pre_public_id": pre_id,
                                    "post_public_id": post_id,
                                    "event": "public_id_changed_same_gt",
                                }
                            )
                # new post ids not present in pre at this frame
                pre_ids = {tid for tid, _ in pre_rows}
                new_ids = post_ids_this_frame - pre_ids
                if new_ids:
                    stats["new_post_ids_total"] += len(new_ids)
                    for pid in sorted(new_ids):
                        csv_rows.append(
                            {
                                "sequence": seq,
                                "protocol": proto,
                                "frame": f + 1,
                                "gt_id": None,
                                "pre_public_id": None,
                                "post_public_id": pid,
                                "event": "new_public_id_in_post",
                            }
                        )
                # duplicate (frame,id) rows flagged per frame
                if len(ids) != len(set(ids)):
                    csv_rows.append(
                        {
                            "sequence": seq,
                            "protocol": proto,
                            "frame": f + 1,
                            "gt_id": None,
                            "pre_public_id": None,
                            "post_public_id": None,
                            "event": "duplicate_public_id_in_post",
                        }
                    )
            row = {
                "sequence": seq,
                "protocol": proto,
                "frames": stats["frames"],
                "id_changed_same_gt": stats["id_changed_same_gt"],
                "post_only_duplicate_frame_id": stats["post_only_duplicate_frame_id"],
                "post_public_id_two_boxes": stats["post_public_id_two_boxes"],
                "post_gt_numeric_collision": stats["post_gt_numeric_collision"],
                "post_gt_wrong_numeric_collision": stats["post_gt_wrong_numeric_collision"],
                "new_post_ids_total": stats["new_post_ids_total"],
                "gt_ids": gt_ids_all,
                "gt_to_pre_id_counts": {
                    str(k): sorted(v)
                    for k, v in stats["gt_to_pre_ids"].items()
                },
                "gt_to_post_id_counts": {
                    str(k): sorted(v)
                    for k, v in stats["gt_to_post_ids"].items()
                },
                "post_id_to_gt_counts": {
                    str(k): sorted(v)
                    for k, v in stats["post_id_to_gts"].items()
                },
            }
            seq_rows.append(row)
        summary[seq] = seq_rows
    with (OUT / "n5_pre_post_identity_delta.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sequence",
                "protocol",
                "frame",
                "gt_id",
                "pre_public_id",
                "post_public_id",
                "event",
            ],
        )
        w.writeheader()
        w.writerows(csv_rows)
    (OUT / "n5_pre_post_identity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for seq in SEQS:
        for proto in PROTOCOLS:
            s = next(r for r in summary[seq] if r["protocol"] == proto)
            print(
                seq,
                proto,
                "id_changed",
                s["id_changed_same_gt"],
                "dup_rows",
                s["post_only_duplicate_frame_id"],
                "new_post_ids",
                s["new_post_ids_total"],
                "gt_numeric_collision",
                s["post_gt_numeric_collision"],
                "wrong_collision",
                s["post_gt_wrong_numeric_collision"],
                "gt->post ids",
                s["gt_to_post_id_counts"],
                "post id->gts",
                s["post_id_to_gt_counts"],
            )


if __name__ == "__main__":
    main()
