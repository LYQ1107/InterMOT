#!/usr/bin/env python
"""Train N9 association models on DanceTrack train decision episodes."""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.continuous_observer import match_boxes, read_mot_rows
from sam3_intermot.n9.models import MLP, PairwiseMLP, SetAssociator
from sam3_intermot.n9.relink_benchmark import load_decision_csv


ROOT = Path(".")
FEAT = ROOT / "outputs/n9/features"
BENCH = ROOT / "outputs/n9/benchmark/train"
P0_TRAIN = ROOT / "outputs/n9/p0_train"
OUT = ROOT / "outputs/n9/models"
MOTION_DIM = 10


def load_feats(seq):
    p = FEAT / f"{seq}.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    key = {(int(f), int(t)): i for i, (f, t) in enumerate(zip(d["frame"], d["tid"]))}
    return key, d["feat"].astype(np.float32)


def feat_at(key, feats, frame, tid):
    i = key.get((int(frame), int(tid)))
    if i is None:
        return None
    v = feats[i]
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def match_tid(gt, rows, frame, gid):
    gtf = gt.get(frame)
    fr = rows.get(frame, [])
    if gtf is None or not fr:
        return None
    ms = match_boxes(
        [np.asarray(b, float) for b in gtf.boxes],
        [np.asarray(b, float) for _, b in fr],
        0.5,
    )
    for gi, pi, _ in ms:
        if gtf.gt_ids[gi] == gid:
            return int(fr[pi][0])
    return None


def mem_feat(key, feats, gid_to_tid, gid, mem_frames):
    vals = []
    for f in mem_frames:
        tid = gid_to_tid.get(f, {}).get(gid)
        if tid is None:
            continue
        v = feat_at(key, feats, f, tid)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    v = np.mean(vals, axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def motion_vec(mem_frames, gap, crowd, mem_last_tid, row_box, p0_rows, tid_changed):
    # rough last box from p0 rows of the last memory frame's tid
    last_box = None
    if mem_frames and mem_last_tid is not None:
        for f in reversed(mem_frames):
            for t, b in p0_rows.get(f, []):
                if int(t) == int(mem_last_tid):
                    last_box = np.asarray(b, float)
                    break
            if last_box is not None:
                break
    iou = 0.0
    dist = 1000.0
    rb = np.asarray(row_box, float)
    if last_box is not None:
        ix1, iy1 = max(last_box[0], rb[0]), max(last_box[1], rb[1])
        ix2, iy2 = min(last_box[2], rb[2]), min(last_box[3], rb[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (last_box[2] - last_box[0]) * (last_box[3] - last_box[1]) + (
            rb[2] - rb[0]
        ) * (rb[3] - rb[1]) - inter
        iou = inter / union if union > 0 else 0.0
        c1 = np.asarray([(last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2])
        c2 = np.asarray([(rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2])
        dist = float(np.linalg.norm(c1 - c2))
    return np.asarray(
        [
            min(1.0, (gap or 0) / 200.0),
            min(1.0, len(mem_frames) / 10.0),
            min(1.0, crowd / 20.0),
            iou,
            min(1.0, dist / 1000.0),
            min(1.0, (rb[2] - rb[0]) / 2000.0),
            min(1.0, (rb[3] - rb[1]) / 1000.0),
            min(1.0, (mem_frames[-1] if mem_frames else 0) / 2000.0),
            float(tid_changed),
            0.0,
        ],
        dtype=np.float32,
    )


class Dataset:
    def __init__(self, seqs, split_tag, mode):
        self.seqs = seqs
        self.split = split_tag
        self.mode = mode
        self.ds = DanceTrackDataset("/path/to/dancetrack", split="train")
        self.cache = {}

    def _load_seq(self, seq):
        if seq in self.cache:
            return self.cache[seq]
        key, feats = load_feats(seq)
        p0 = read_mot_rows(P0_TRAIN / f"{seq}.txt")
        gt = self.ds.load_gt(seq)
        gid_to_tid = {}
        for f, fr in p0.items():
            gtf = gt.get(f)
            if gtf is None or not gtf.boxes or not fr:
                continue
            ms = match_boxes(
                [np.asarray(b, float) for b in gtf.boxes],
                [np.asarray(b, float) for _, b in fr],
                0.5,
            )
            gid_to_tid[f] = {gtf.gt_ids[gi]: int(fr[pi][0]) for gi, pi, _ in ms}
        eps = load_decision_csv(BENCH / f"{seq}.csv")
        pairs = []
        for e in eps:
            if e["miss"] or e["pos_tid"] is None or not e["mem_frames"]:
                continue
            mf = mem_feat(key, feats, gid_to_tid, e["gid"], e["mem_frames"])
            pf = feat_at(key, feats, e["frame"], e["pos_tid"])
            if mf is None or pf is None:
                continue
            prev_tid = gid_to_tid.get(e["mem_frames"][-1], {}).get(e["gid"])
            pos_motion = motion_vec(
                e["mem_frames"], e["gap"] or 0, e["crowd"], prev_tid, _box(p0, e["frame"], e["pos_tid"]), p0, e["tid_changed"]
            )
            pos_box = _box(p0, e["frame"], e["pos_tid"])
            pairs.append(
                {
                    "gid": e["gid"],
                    "frame": e["frame"],
                    "pos_tid": e["pos_tid"],
                    "mem_frames": e["mem_frames"],
                    "mem": mf,
                    "pos": pf,
                    "pos_box": pos_box,
                    "pos_motion": pos_motion,
                    "neg": [
                        feat_at(key, feats, e["frame"], nt)
                        for nt in e["neg_tids"]
                    ],
                    "neg_tids": e["neg_tids"],
                    "crowd": e["crowd"],
                    "tid_changed": e["tid_changed"],
                    "gap": e["gap"] or 0,
                }
            )
        self.cache[seq] = {
            "pairs": [p for p in pairs if all(n is not None for n in p["neg"]) or not p["neg"]],
            "p0": p0,
            "gt": gt,
            "key": key,
            "feats": feats,
            "gid_to_tid": gid_to_tid,
        }
        return self.cache[seq]

    def pairwise_batches(self, batch_size=256):
        pos, neg = [], []
        for seq in self.seqs:
            d = self._load_seq(seq)
            for p in d["pairs"]:
                pos.append(p)
                for nf, nt in zip(p["neg"], p["neg_tids"]):
                    nb = _box(d["p0"], p["frame"], nt)
                    neg.append((p, nf, motion_vec(p["mem_frames"], p["gap"], p["crowd"], None, nb, d["p0"], p["tid_changed"])))
        rng = random.Random(0)
        rng.shuffle(pos)
        rng.shuffle(neg)
        n = min(len(pos), len(neg))
        pos, neg = pos[:n], neg[:n]
        batches = []
        for i in range(0, n, batch_size):
            pb = pos[i : i + batch_size]
            nb = neg[i : i + batch_size]
            if len(pb) < 4:
                break
            batches.append(
                (
                    torch.as_tensor(np.stack([p["mem"] for p in pb])),
                    torch.as_tensor(np.stack([p["pos"] for p in pb])),
                    torch.as_tensor(np.stack([p["pos_motion"] for p in pb])),
                    torch.as_tensor(np.stack([p["mem"] for p, _, _ in nb])),
                    torch.as_tensor(np.stack([nf for _, nf, _ in nb])),
                    torch.as_tensor(np.stack([mv for _, _, mv in nb])),
                )
            )
        return batches

    def frame_sets(self, seq):
        d = self._load_seq(seq)
        sets = defaultdict(list)
        for p in d["pairs"]:
            sets[p["frame"]].append(p)
        out = []
        for f, ps in sorted(sets.items()):
            mems, rows, pos = [], {}, []
            row_feats = {}
            row_boxes = {}
            changed_gids = {p["gid"] for p in ps if p["tid_changed"]}
            anchor_feats = {p["gid"]: p["pos"] for p in ps if p["tid_changed"] and p["pos"] is not None}
            for t, b in d["p0"].get(f, []):
                fv = feat_at(d["key"], d["feats"], f, int(t))
                if fv is not None:
                    row_feats[int(t)] = fv
                    row_boxes[int(t)] = b
            for p in ps:
                mems.append((p["gid"], p["mem"], p["pos_motion"]))
            tids_sorted = sorted(row_feats)
            for mi, p in enumerate(ps):
                if p["pos_tid"] in row_feats and p["pos"] is not None:
                    pos.append((mi, tids_sorted.index(p["pos_tid"])))
            rows = [(t, row_feats[t], _row_motion(row_boxes[t], f)) for t in tids_sorted]
            if len(mems) < 2 or len(rows) < 2:
                continue
            out.append(
                {
                    "frame": f,
                    "mems": mems,
                    "rows": rows,
                    "pos": pos,
                    "gid_set": {p["gid"] for p in ps},
                    "sequence": seq,
                    "changed_gids": changed_gids,
                    "anchor_feats": anchor_feats,
                }
            )
        return out


def _box(p0, frame, tid):
    for t, b in p0.get(frame, []):
        if int(t) == int(tid):
            return np.asarray(b, float)
    return np.zeros(4, dtype=float)


def _row_motion(box, frame):
    rb = np.asarray(box, float)
    return np.asarray(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            min(1.0, (rb[2] - rb[0]) / 2000.0),
            min(1.0, (rb[3] - rb[1]) / 1000.0),
            min(1.0, frame / 2000.0),
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def collate_sets(frame_sets, max_objects=24):
    batches = []
    idx = list(range(len(frame_sets)))
    random.Random(1).shuffle(idx)
    B = 8
    for i in range(0, len(idx), B):
        chunk = [frame_sets[j] for j in idx[i : i + B]]
        R = min(max_objects, max(len(fs["rows"]) for fs in chunk))
        M = min(max_objects, max(len(fs["mems"]) for fs in chunk))
        mem_feats = np.zeros((len(chunk), M, 512), np.float32)
        mem_mot = np.zeros((len(chunk), M, MOTION_DIM), np.float32)
        row_feats = np.zeros((len(chunk), R, 512), np.float32)
        row_mot = np.zeros((len(chunk), R, MOTION_DIM), np.float32)
        mem_mask = np.zeros((len(chunk), M), bool)
        row_mask = np.zeros((len(chunk), R), bool)
        target = np.full((len(chunk), M), -1, np.int64)
        target_row = np.full((len(chunk), R), -1, np.int64)
        for bi, fs in enumerate(chunk):
            for mi, (gid, mf, mm) in enumerate(fs["mems"][:M]):
                mem_feats[bi, mi] = mf
                mem_mot[bi, mi] = mm
                mem_mask[bi, mi] = True
            for ri, (tid, rf, rm) in enumerate(fs["rows"][:R]):
                row_feats[bi, ri] = rf
                row_mot[bi, ri] = rm
                row_mask[bi, ri] = True
            tid_to_idx = {t: i for i, (t, _, _) in enumerate(fs["rows"][:R])}
            pos_map = {mi: ri for mi, ri in fs["pos"] if mi < M and ri < R}
            for mi, ri in pos_map.items():
                target[bi, mi] = ri
                target_row[bi, ri] = mi
        batches.append(
            (
                torch.as_tensor(mem_feats),
                torch.as_tensor(row_feats),
                torch.as_tensor(mem_mot),
                torch.as_tensor(row_mot),
                torch.as_tensor(mem_mask),
                torch.as_tensor(row_mask),
                torch.as_tensor(target),
                torch.as_tensor(target_row),
            )
        )
    return batches


def build_anchor_episodes(ds, all_sets):
    by_seq = defaultdict(list)
    for fs in all_sets:
        by_seq[fs["sequence"]].append(fs)
    eps = []
    for seq, sets in by_seq.items():
        sets.sort(key=lambda fs: fs["frame"])
        for i, fs in enumerate(sets):
            for gid in fs["changed_gids"]:
                anchor = fs["anchor_feats"].get(gid)
                if anchor is None:
                    continue
                for fs2 in sets[i + 1 : i + 12]:
                    if 1 <= fs2["frame"] - fs["frame"] <= 10 and gid in fs2["gid_set"]:
                        eps.append((fs, fs2, gid, anchor))
                        break
    return eps


def train_set(train_seqs, cal_seqs, args, hcpim=False):
    ds = Dataset(train_seqs, "train", "set")
    model = SetAssociator(feat_dim=512, motion_dim=MOTION_DIM, d=args.hidden, layers=2, heads=2)
    if hcpim and (OUT / "set_associator.pt").exists():
        model.load_state_dict(torch.load(OUT / "set_associator.pt", map_location="cpu"))
        print(json.dumps({"init": "set_associator.pt"}), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    all_sets = []
    for seq in train_seqs:
        all_sets.extend(ds.frame_sets(seq))
    print(json.dumps({"frame_sets": len(all_sets)}), flush=True)
    anchor_eps = build_anchor_episodes(ds, all_sets) if hcpim else []
    print(json.dumps({"anchor_episodes": len(anchor_eps)}), flush=True)
    best = -1
    for epoch in range(args.epochs):
        model.train()
        tot, cnt = 0.0, 0
        for batch in collate_sets(all_sets):
            mem_feats, row_feats, mem_mot, row_mot, mem_mask, row_mask, target, target_row = batch
            logits = model(mem_feats, row_feats, mem_mot, row_mot, mem_mask, row_mask)
            loss = 0.0
            n = 0
            for bi in range(logits.shape[0]):
                mm = mem_mask[bi]
                valid = mm & (target[bi] >= 0)
                if valid.any():
                    loss = loss + F.cross_entropy(
                        logits[bi].transpose(0, 1)[valid], target[bi][valid]
                    )
                    n += 1
                rr = row_mask[bi] & (target_row[bi] >= 0)
                if rr.any():
                    loss = loss + F.cross_entropy(logits[bi][rr], target_row[bi][rr])
                    n += 1
            if n == 0:
                continue
            loss = loss / n
            if hcpim and anchor_eps and cnt % 5 == 0:
                anchor_loss = anchor_objective(model, anchor_eps)
                if anchor_loss is not None:
                    loss = loss + 0.5 * anchor_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            cnt += 1
        acc = evaluate_set(model, ds, cal_seqs)
        print(json.dumps({"epoch": epoch, "loss": round(tot / max(1, cnt), 4), "cal_acc": round(acc, 4)}), flush=True)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), OUT / ("set_associator.pt" if not hcpim else "hcpim.pt"))
    return model


def anchor_objective(model, eps):
    """Correction-conditioned future objective on tid-change episodes."""
    import random as _r

    _r.Random(3).shuffle(eps)
    eps = eps[:8]
    total = 0.0
    cnt = 0
    for fs, fs2, gid, anchor in eps:
        mem_feats = np.stack([m[1] for m in fs2["mems"]])
        mem_mot = np.stack([m[2] for m in fs2["mems"]])
        row_feats = np.stack([r[1] for r in fs2["rows"]])
        row_mot = np.stack([r[2] for r in fs2["rows"]])
        gid_to_mi = {m[0]: i for i, m in enumerate(fs2["mems"])}
        mi = gid_to_mi.get(gid)
        if mi is None:
            continue
        logits_a = model(
            torch.as_tensor(mem_feats[None]),
            torch.as_tensor(row_feats[None]),
            torch.as_tensor(mem_mot[None]),
            torch.as_tensor(row_mot[None]),
        )[0]
        mem_b = mem_feats.copy()
        a = anchor / (np.linalg.norm(anchor) + 1e-9)
        blended = 0.7 * a + 0.3 * mem_b[mi]
        blended /= np.linalg.norm(blended) + 1e-9
        mem_b[mi] = blended
        logits_b = model(
            torch.as_tensor(mem_b[None]),
            torch.as_tensor(row_feats[None]),
            torch.as_tensor(mem_mot[None]),
            torch.as_tensor(row_mot[None]),
        )[0]
        target_row = next((ri for mi2, ri in fs2["pos"] if mi2 == mi), None)
        if target_row is None:
            continue
        ce_a = F.cross_entropy(logits_a[target_row][None], torch.as_tensor([mi]))
        ce_b = F.cross_entropy(logits_b[target_row][None], torch.as_tensor([mi]))
        gain = torch.clamp_min(ce_b - ce_a + 0.05, 0.0)
        preserve = torch.abs(logits_b - logits_a).mean()
        total = total + ce_b + gain + 0.3 * preserve
        cnt += 1
    return (total / max(1, cnt)).detach() if False else total / max(1, cnt)


def evaluate_set(model, ds, seqs):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for seq in seqs:
            for fs in ds.frame_sets(seq):
                if not fs["pos"]:
                    continue
                mem_feats = np.stack([m[1] for m in fs["mems"]])
                mem_mot = np.stack([m[2] for m in fs["mems"]])
                row_feats = np.stack([r[1] for r in fs["rows"]])
                row_mot = np.stack([r[2] for r in fs["rows"]])
                logits = model(
                    torch.as_tensor(mem_feats[None]),
                    torch.as_tensor(row_feats[None]),
                    torch.as_tensor(mem_mot[None]),
                    torch.as_tensor(row_mot[None]),
                )[0]
                for mi, ri in fs["pos"]:
                    scores = logits[ri].numpy()
                    if int(np.argmax(scores)) == int(mi):
                        correct += 1
                    total += 1
    return correct / max(1, total)


def train_pairwise(train_seqs, cal_seqs, args):
    ds = Dataset(train_seqs, "train", "pairwise")
    model = PairwiseMLP(feat_dim=512, motion_dim=MOTION_DIM, hidden=args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = -1
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        cnt = 0
        for mb in ds.pairwise_batches(args.batch_size):
            pos_mem, pos_row, pos_mot, neg_mem, neg_row, neg_mot = mb
            p = model(pos_mem, pos_row, pos_mot)
            n = model(neg_mem, neg_row, neg_mot)
            loss = F.binary_cross_entropy_with_logits(
                torch.cat([p, n]), torch.cat([torch.ones_like(p), torch.zeros_like(n)])
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            cnt += 1
        acc = evaluate_pairwise(model, cal_seqs)
        print(json.dumps({"epoch": epoch, "loss": round(tot / max(1, cnt), 4), "cal_acc": round(acc, 4)}), flush=True)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), OUT / "pairwise_mlp.pt")
    return model


def evaluate_pairwise(model, seqs):
    ds = Dataset(seqs, "cal", "pairwise")
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for seq in seqs:
            d = ds._load_seq(seq)
            for p in d["pairs"]:
                pos = torch.as_tensor(np.dot(p["mem"], p["pos"]))
                pm = torch.as_tensor(p["pos_motion"])
                mem = torch.as_tensor(p["mem"])
                pr = model(mem.unsqueeze(0), torch.as_tensor(p["pos"]).unsqueeze(0), pm.unsqueeze(0))
                scores = [pr.item()]
                for nf, nt in zip(p["neg"], p["neg_tids"]):
                    nb = _box(d["p0"], p["frame"], nt)
                    mv = motion_vec(p["mem_frames"], p["gap"], p["crowd"], None, nb, d["p0"], p["tid_changed"])
                    nr = model(mem.unsqueeze(0), torch.as_tensor(nf).unsqueeze(0), torch.as_tensor(mv).unsqueeze(0))
                    scores.append(nr.item())
                if scores and scores.index(max(scores)) == 0:
                    correct += 1
                total += 1
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pairwise", "set", "hcpim"], default="pairwise")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    all_seqs = sorted(p.stem for p in BENCH.glob("*.csv") if p.stem != "summary")
    train_seqs = os.environ.get("N9_TRAIN_SEQS", "").split() or all_seqs[:30]
    cal_seqs = os.environ.get("N9_CAL_SEQS", "").split() or all_seqs[30:]
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    if args.mode == "pairwise":
        train_pairwise(train_seqs, cal_seqs, args)
    elif args.mode == "set":
        train_set(train_seqs, cal_seqs, args)
    elif args.mode == "hcpim":
        train_set(train_seqs, cal_seqs, args, hcpim=True)


if __name__ == "__main__":
    main()
