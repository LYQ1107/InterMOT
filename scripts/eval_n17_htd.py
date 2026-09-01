#!/usr/bin/env python
"""Evaluate HTD-v1 on the N17 candidate-creation benchmark (calibration)."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(".")
OUT = ROOT / "outputs/n17"
DT = Path("/path/to/dancetrack")
CLIP_CKPT = ROOT / "outputs/n15/checkpoints/clip_reid_hf/person_vit_clip_reid.pth"
CACHE = OUT / "enc_cache"


def load_mem(seq, f):
    p = CACHE / f"{seq}_{f}.npy"
    if not p.exists():
        return None
    return torch.from_numpy(np.load(p)).float().reshape(72, 72, 256)


def iou_xyxy(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def load_gt_boxes(seq):
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    d = DanceTrackDataset(str(DT), sequences=[], split="train").load_gt(seq)
    return {f: list(zip(g.gt_ids, g.boxes)) for f, g in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/n17/models/htd_v1_shard0.pt")
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--pres-tau", type=float, default=0.5)
    ap.add_argument("--manifest", default="cal_episodes.csv")
    ap.add_argument("--tag", default="")
    ap.add_argument("--modulate", type=int, default=1)
    args = ap.parse_args()
    from scripts.run_n15_extract_features import build_clipreid
    from sam3_intermot.recovery.htd import HTD
    import torchvision.transforms as T

    ck = torch.load(ROOT / args.model, map_location="cpu", weights_only=False)
    model = HTD(modulate=bool(args.modulate)).cuda().eval()
    model.load_state_dict(ck["state"])
    clip = build_clipreid(str(CLIP_CKPT), "cuda")
    clip_tf = T.Compose(
        [
            T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    with (OUT / args.manifest).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.max_samples:
        rng = np.random.default_rng(23)
        rng.shuffle(rows)
        rows = rows[: args.max_samples]
    print(f"eval episodes={len(rows)}", flush=True)
    detail = []
    gt_cache = {}
    iw, ih = 1920.0, 1080.0
    for r in rows:
        seq = r["sequence"]
        t, f = int(r["t"]), int(r["f"])
        rm = load_mem(seq, t)
        sm = load_mem(seq, f)
        if rm is None or sm is None:
            continue
        hb = np.asarray(json.loads(r["human_box"]), dtype=float)
        present = int(r["target_present"]) == 1
        fb = np.asarray(json.loads(r["target_box"]), dtype=float) if present else None
        generic_miss = int(r["generic_miss"]) == 1 if r["generic_miss"] != "" else False
        box = torch.tensor(
            [(hb[0] + hb[2]) / 2 / iw, (hb[1] + hb[3]) / 2 / ih,
             (hb[2] - hb[0]) / iw, (hb[3] - hb[1]) / ih],
            dtype=torch.float32, device="cuda",
        )
        img = Image.open(DT / "train" / seq / "img1" / f"{t + 1:08d}.jpg").convert("RGB")
        x1, y1, x2, y2 = [int(round(v)) for v in hb]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        crop = img.crop((x1, y1, x2, y2))
        x = clip_tf(crop).unsqueeze(0).cuda()
        with torch.no_grad():
            _, x12, xproj = clip(x)
            h = F.normalize(torch.cat([x12[:, 0], xproj[:, 0]], dim=1), dim=-1)
            pred = model(
                rm.unsqueeze(0).cuda(), box.unsqueeze(0), h, sm.unsqueeze(0).cuda()
            )
            boxes = model.decode_boxes(pred["boxes"][0]).detach().cpu().numpy()
            tgt = torch.sigmoid(pred["targetness"][0]).detach().cpu().numpy()
            presence = float(torch.sigmoid(pred["presence"][0]).item())
        prop_xyxy = boxes * np.asarray([iw, ih, iw, ih])
        best_iou = 0.0
        if present:
            for pxy in prop_xyxy:
                best_iou = max(best_iou, iou_xyxy(pxy, fb))
            recall_03 = best_iou >= 0.3
            recall_05 = best_iou >= 0.5
            recall_07 = best_iou >= 0.7
            order = np.argsort(-tgt)
            top1 = iou_xyxy(prop_xyxy[order[0]], fb) >= 0.5
            top3 = any(iou_xyxy(pxy, fb) >= 0.5 for pxy in prop_xyxy[order[:3]])
        else:
            recall_03 = recall_05 = recall_07 = top1 = top3 = False
        ghost = (not present) and presence >= args.pres_tau
        fc = 0
        if present:
            if seq not in gt_cache:
                gt_cache[seq] = load_gt_boxes(seq)
            for pxy in prop_xyxy:
                for ogid, obox in gt_cache[seq].get(f, []):
                    if ogid == int(r["gid"]):
                        continue
                    if iou_xyxy(pxy, obox) >= 0.5:
                        fc = 1
        detail.append(
            {
                "sequence": seq, "t": t, "gid": r["gid"], "f": f, "delta": r["delta"],
                "present": int(present), "generic_miss": int(generic_miss),
                "best_iou": round(best_iou, 3), "recall_03": int(recall_03),
                "recall_05": int(recall_05), "recall_07": int(recall_07),
                "top1": int(top1), "top3": int(top3),
                "presence": round(presence, 3), "ghost": int(ghost),
                "false_capture": fc, "crowd": r["crowd"],
            }
        )
    tag = f"_{args.tag}" if args.tag else ""
    with (OUT / f"candidate_creation{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        w.writeheader()
        w.writerows(detail)
    per_seq = defaultdict(lambda: [0, 0, 0, 0, 0])
    for d in detail:
        s = per_seq[d["sequence"]]
        s[0] += 1
        s[1] += int(d["present"])
        s[2] += int(d["generic_miss"])
        s[3] += int(d["recall_05"]) if d["present"] else 0
        s[4] += int(d["ghost"])
    with (OUT / f"candidate_creation_per_sequence{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "n", "present", "generic_miss", "recall05_present", "ghost"])
        for seq, s in sorted(per_seq.items()):
            w.writerow([seq, *s])
    pres = [d for d in detail if d["present"]]
    miss = [d for d in detail if d["generic_miss"]]
    abs_ = [d for d in detail if not d["present"]]
    with (OUT / f"absence_results{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_absent", "ghost_rate", "mean_presence"])
        w.writerow([len(abs_), round(np.mean([d["ghost"] for d in abs_]), 4) if abs_ else "",
                    round(np.mean([d["presence"] for d in abs_]), 4) if abs_ else ""])
    with (OUT / f"natural_miss_results{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_miss", "CCR_05", "recall_03", "recall_07", "top1", "top3"])
        w.writerow([
            len(miss),
            round(np.mean([d["recall_05"] for d in miss]), 4) if miss else "",
            round(np.mean([d["recall_03"] for d in miss]), 4) if miss else "",
            round(np.mean([d["recall_07"] for d in miss]), 4) if miss else "",
            round(np.mean([d["top1"] for d in miss]), 4) if miss else "",
            round(np.mean([d["top3"] for d in miss]), 4) if miss else "",
        ])
    with (OUT / f"hard_negative_results{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_present", "fc_rate", "recall05_present", "best_iou_mean"])
        w.writerow([
            len(pres),
            round(np.mean([d["false_capture"] for d in pres]), 4) if pres else "",
            round(np.mean([d["recall_05"] for d in pres]), 4) if pres else "",
            round(np.mean([d["best_iou"] for d in pres]), 4) if pres else "",
        ])
    with (OUT / f"novel_candidate_results{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_miss", "novel_created", "novel_rate"])
        n_created = sum(d["recall_05"] for d in miss)
        w.writerow([len(miss), n_created,
                    round(n_created / max(1, len(miss)), 4)])
    print(
        f"n={len(detail)} present={len(pres)} miss={len(miss)} absent={len(abs_)}\n"
        f"recall05_present={np.mean([d['recall_05'] for d in pres]) if pres else 0:.4f} "
        f"CCR_miss={np.mean([d['recall_05'] for d in miss]) if miss else 0:.4f} "
        f"top1={np.mean([d['top1'] for d in pres]) if pres else 0:.4f} "
        f"ghost_rate={np.mean([d['ghost'] for d in abs_]) if abs_ else 0:.4f} "
        f"fc_rate={np.mean([d['false_capture'] for d in pres]) if pres else 0:.4f}"
    )


if __name__ == "__main__":
    main()
