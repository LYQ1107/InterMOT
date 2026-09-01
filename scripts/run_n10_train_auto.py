#!/usr/bin/env python
"""Train N10 AUTO association models on chunk-based online rollouts.

Teacher forcing: identity memories are built chronologically from GT-matched
observations inside each sequence (GT never enters model features).  Inference
uses the same scorer inside a free online state machine.
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sam3_intermot.association.identity_state import IdentityState
from sam3_intermot.association.observation_tape import load_tape, tape_rows_by_frame
from sam3_intermot.association.online_associator import (
    MOTION_DIM,
    SetAssociator,
    PairwiseMLP,
    motion_vec,
)
from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.continuous_observer import match_boxes
from sam3_intermot.interaction.simulator import GTFrame


ROOT = Path(".")
TAPE = ROOT / "outputs/n10/tapes"
DT = Path("/path/to/dancetrack")
OUT = ROOT / "outputs/n10/models"
FEAT_DIM = 512


def load_seq_data(seq: str, split: str):
    tape = load_tape(TAPE / f"{seq}.npz")
    obs_by_frame = tape_rows_by_frame(tape)
    ds = DanceTrackDataset(str(DT), sequences=[seq], split=split)
    gt = ds.load_gt(seq)
    num_frames = ds.num_frames(seq)
    gid_to_obs: Dict[int, Dict[int, int]] = {}
    for f, obs in obs_by_frame.items():
        gtf = gt.get(f)
        if gtf is None or not gtf.boxes or not obs:
            continue
        ms = match_boxes(
            [np.asarray(b, float) for b in gtf.boxes],
            [np.asarray(o["box"], dtype=float) for o in obs],
            0.5,
        )
        gid_to_obs[f] = {
            gtf.gt_ids[gi]: obs[pi]["obs_id"] for gi, pi, _ in ms
        }
    return {
        "obs_by_frame": obs_by_frame,
        "gid_to_obs": gid_to_obs,
        "gt": gt,
        "num_frames": num_frames,
        "seq": seq,
        "split": split,
    }


def _state_for_gid(states: Dict[int, IdentityState], gid: int, obs: dict, frame: int) -> IdentityState:
    st = states.get(gid)
    if st is None:
        st = IdentityState(-gid, obs["feat"], obs["box"], frame, obs["native_tid"])
        states[gid] = st
    return st


def walk_sequence(data: dict, max_lost: int = 90) -> Iterator[dict]:
    """Teacher-forced online walk; yields per-frame sample records."""
    states: Dict[int, IdentityState] = {}
    obs_by_frame = data["obs_by_frame"]
    gid_to_obs = data["gid_to_obs"]
    for f in range(data["num_frames"]):
        obs_list = obs_by_frame.get(f, [])
        obs_index = {o["obs_id"]: i for i, o in enumerate(obs_list)}
        gmap = gid_to_obs.get(f, {})
        matched_gids = set()
        for gid, oid in gmap.items():
            i = obs_index.get(oid)
            if i is None:
                continue
            obs = obs_list[i]
            st = states.get(gid)
            if st is None:
                _state_for_gid(states, gid, obs, f)
                continue
            matched_gids.add(gid)
            tid_changed = 1.0 if obs["native_tid"] != st.last_native_tid else 0.0
            gap = max(0, f - st.last_seen_frame)
            crowd = len(obs_list)
            hard = 1.0 + 1.5 * tid_changed + 2.0 * (gap > 10) + 1.0 * (crowd > 8)
            mem = st.effective_feat()
            pos_mot = motion_vec(st, obs, f, crowd)
            yield {
                "gid": gid,
                "frame": f,
                "mem": mem,
                "pos": obs["feat"],
                "pos_mot": pos_mot,
                "neg": [
                    (o["feat"], motion_vec(st, o, f, crowd))
                    for j, o in enumerate(obs_list)
                    if j != i
                ][:8],
                "weight": hard,
            }
            st.update_machine(obs["feat"], obs["box"], f, obs["native_tid"], 0.9)
        # lost/terminate
        for gid, st in list(states.items()):
            if gid in matched_gids:
                continue
            if gid in gmap:
                continue
            if f - st.last_seen_frame > max_lost:
                del states[gid]


def pairwise_batches(
    seqs: List[str],
    split: str,
    batch_size: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    buf_mem, buf_pos, buf_pos_mot, buf_neg_mem, buf_neg_row, buf_neg_mot, buf_wp, buf_wn = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for seq in seqs:
        data = load_seq_data(seq, split)
        for rec in walk_sequence(data):
            buf_mem.append(rec["mem"])
            buf_pos.append(rec["pos"])
            buf_pos_mot.append(rec["pos_mot"])
            buf_wp.append(rec["weight"])
            for nf, nm in rec["neg"]:
                buf_neg_mem.append(rec["mem"])
                buf_neg_row.append(nf)
                buf_neg_mot.append(nm)
                buf_wn.append(rec["weight"])
            if len(buf_pos) >= batch_size:
                yield _pair_batch(buf_mem, buf_pos, buf_pos_mot, buf_neg_mem, buf_neg_row, buf_neg_mot, buf_wp, buf_wn, batch_size)
                buf_mem, buf_pos, buf_pos_mot, buf_neg_mem, buf_neg_row, buf_neg_mot, buf_wp, buf_wn = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
    if len(buf_pos) >= 8:
        yield _pair_batch(buf_mem, buf_pos, buf_pos_mot, buf_neg_mem, buf_neg_row, buf_neg_mot, buf_wp, buf_wn, len(buf_pos))


def _pair_batch(mem, pos, pm, nmem, nrow, nmot, wp, wn, size):
    pos = pos[:size]
    neg = nrow[:size * 8]
    mems = mem[:size] + nmem[:size * 8]
    rows = pos + neg
    mott = pm[:size] + nmot[:size * 8]
    weights = wp[:size] + wn[:size * 8]
    labels = [1.0] * len(pos) + [0.0] * len(neg)
    return (
        torch.as_tensor(np.stack(mems)),
        torch.as_tensor(np.stack(rows)),
        torch.as_tensor(np.stack(mott)),
        torch.as_tensor(np.asarray(weights, dtype=np.float32)),
        torch.as_tensor(np.asarray(labels, dtype=np.float32)),
    )


def evaluate_pairwise(model, seqs: List[str], split: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for seq in seqs:
            data = load_seq_data(seq, split)
            states: Dict[int, IdentityState] = {}
            obs_by_frame = data["obs_by_frame"]
            gid_to_obs = data["gid_to_obs"]
            for f in range(data["num_frames"]):
                obs_list = obs_by_frame.get(f, [])
                obs_index = {o["obs_id"]: i for i, o in enumerate(obs_list)}
                gmap = gid_to_obs.get(f, {})
                valid = [
                    (gid, oid, obs_index[oid])
                    for gid, oid in gmap.items()
                    if oid in obs_index
                ]
                rows = [o["feat"] for o in obs_list]
                for gid, oid, i in valid:
                    st = states.get(gid)
                    if st is None:
                        continue
                    mems = np.repeat(st.effective_feat()[None], len(obs_list), axis=0)
                    mott = np.stack(
                        [motion_vec(st, o, f, len(obs_list)) for o in obs_list]
                    )
                    s = model(
                        torch.as_tensor(mems),
                        torch.as_tensor(np.stack(rows)),
                        torch.as_tensor(mott),
                    ).numpy()
                    if int(np.argmax(s)) == i:
                        correct += 1
                    total += 1
                for gid, oid, i in valid:
                    obs = obs_list[i]
                    st = states.get(gid)
                    if st is None:
                        _state_for_gid(states, gid, obs, f)
                    else:
                        st.update_machine(obs["feat"], obs["box"], f, obs["native_tid"], 0.9)
    return correct / max(1, total)


def set_batches(seqs: List[str], split: str, batch_size: int = 8):
    frames = []
    for seq in seqs:
        data = load_seq_data(seq, split)
        states: Dict[int, IdentityState] = {}
        obs_by_frame = data["obs_by_frame"]
        gid_to_obs = data["gid_to_obs"]
        for f in range(data["num_frames"]):
            obs_list = obs_by_frame.get(f, [])
            obs_index = {o["obs_id"]: i for i, o in enumerate(obs_list)}
            gmap = gid_to_obs.get(f, {})
            valid = [
                (gid, oid, obs_index[oid])
                for gid, oid in gmap.items()
                if oid in obs_index
            ]
            mems = []
            pos = []
            for gid, oid, i in valid:
                st = states.get(gid)
                if st is None:
                    _state_for_gid(states, gid, obs_list[i], f)
                    continue
                mems.append((gid, st, i))
            if len(mems) >= 2 and len(obs_list) >= 2:
                for mi, (gid, st, i) in enumerate(mems):
                    pos.append((mi, i))
                frames.append(
                    {
                        "frame": f,
                        "mems": mems,
                        "rows": obs_list,
                        "pos": pos,
                    }
                )
            for gid, oid, i in valid:
                st = states.get(gid)
                if st is not None:
                    obs = obs_list[i]
                    st.update_machine(obs["feat"], obs["box"], f, obs["native_tid"], 0.9)
    rng = random.Random(1)
    rng.shuffle(frames)
    for i in range(0, len(frames), batch_size):
        chunk = frames[i : i + batch_size]
        if len(chunk) < 1:
            break
        M = max(len(x["mems"]) for x in chunk)
        R = max(len(x["rows"]) for x in chunk)
        mem_feats = np.zeros((len(chunk), M, FEAT_DIM), np.float32)
        row_feats = np.zeros((len(chunk), R, FEAT_DIM), np.float32)
        mem_mot = np.zeros((len(chunk), M, MOTION_DIM), np.float32)
        row_mot = np.zeros((len(chunk), R, MOTION_DIM), np.float32)
        mem_mask = np.zeros((len(chunk), M), bool)
        row_mask = np.zeros((len(chunk), R), bool)
        target = np.full((len(chunk), M), -1, np.int64)
        for bi, fr in enumerate(chunk):
            for mi, (gid, st, i) in enumerate(fr["mems"]):
                mem_feats[bi, mi] = st.effective_feat()
                mem_mot[bi, mi] = np.asarray(
                    [
                        min(1.0, max(0, fr["frame"] - st.last_seen_frame) / 200.0),
                        min(1.0, (fr["frame"] - st.birth_frame) / 2000.0),
                        0.0, 0.0, 0.0, 0.0, 0.0,
                        min(1.0, st.last_seen_frame / 2000.0),
                        0.0, 0.0, 0.0, 0.0,
                    ],
                    np.float32,
                )
                mem_mask[bi, mi] = True
            for ri, o in enumerate(fr["rows"]):
                row_feats[bi, ri] = o["feat"]
                row_mot[bi, ri] = np.asarray(
                    [
                        0.0, 0.0, min(1.0, len(fr["rows"]) / 20.0), 0.0, 0.0,
                        min(1.0, (o["box"][2] - o["box"][0]) / 2000.0),
                        min(1.0, (o["box"][3] - o["box"][1]) / 1000.0),
                        min(1.0, fr["frame"] / 2000.0),
                        0.0, 0.0, float(o["has_feat"]), 0.0,
                    ],
                    np.float32,
                )
                row_mask[bi, ri] = True
            for mi, ri in fr["pos"]:
                target[bi, mi] = ri
        yield (
            torch.as_tensor(mem_feats),
            torch.as_tensor(row_feats),
            torch.as_tensor(mem_mot),
            torch.as_tensor(row_mot),
            torch.as_tensor(mem_mask),
            torch.as_tensor(row_mask),
            torch.as_tensor(target),
        )


def evaluate_set(model, seqs: List[str], split: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in set_batches(seqs, split, 1):
            mem_feats, row_feats, mem_mot, row_mot, mem_mask, row_mask, target = batch
            logits = model(mem_feats, row_feats, mem_mot, row_mot, mem_mask, row_mask)[0]
            for mi in range(logits.shape[1]):
                if target[0, mi] < 0:
                    continue
                scores = logits[:, mi].numpy()
                if int(np.argmax(scores)) == int(target[0, mi]):
                    correct += 1
                total += 1
    return correct / max(1, total)


def train_pairwise(train_seqs, cal_seqs, args, split):
    model = PairwiseMLP(feat_dim=FEAT_DIM, motion_dim=MOTION_DIM, hidden=args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        cnt = 0
        t0 = time.time()
        for batch in pairwise_batches(train_seqs, split, args.batch_size):
            mem, row, mot, w, lab = batch
            logits = model(mem, row, mot)
            loss = F.binary_cross_entropy_with_logits(logits, lab, weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            cnt += 1
        acc = evaluate_pairwise(model, cal_seqs, split)
        rec = {
            "epoch": epoch,
            "loss": round(tot / max(1, cnt), 4),
            "cal_acc": round(acc, 4),
            "wall_seconds": round(time.time() - t0, 2),
        }
        print(json.dumps(rec), flush=True)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), OUT / "n10_pairwise_mlp.pt")
    return model


def train_set(train_seqs, cal_seqs, args, split):
    model = SetAssociator(feat_dim=FEAT_DIM, motion_dim=MOTION_DIM, d=args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        cnt = 0
        t0 = time.time()
        for batch in set_batches(train_seqs, split, 8):
            mem_feats, row_feats, mem_mot, row_mot, mem_mask, row_mask, target = batch
            logits = model(mem_feats, row_feats, mem_mot, row_mot, mem_mask, row_mask)
            loss = torch.tensor(0.0)
            n = 0
            for bi in range(logits.shape[0]):
                valid = mem_mask[bi] & (target[bi] >= 0)
                if valid.any():
                    loss = loss + F.cross_entropy(
                        logits[bi].transpose(0, 1)[valid], target[bi][valid]
                    )
                    n += 1
            if n == 0:
                continue
            loss = loss / n
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            cnt += 1
        acc = evaluate_set(model, cal_seqs, split)
        rec = {
            "epoch": epoch,
            "loss": round(tot / max(1, cnt), 4),
            "cal_acc": round(acc, 4),
            "wall_seconds": round(time.time() - t0, 2),
        }
        print(json.dumps(rec), flush=True)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), OUT / "n10_set_associator.pt")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pairwise", "set"], default="pairwise")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    all_seqs = sorted(
        p.stem for p in TAPE.glob("*.npz") if p.stem != "observation_manifest"
    )
    train_seqs = os.environ.get("N10_TRAIN_SEQS", "").split() or all_seqs[:30]
    cal_seqs = os.environ.get("N10_CAL_SEQS", "").split() or all_seqs[30:40]
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    print(json.dumps({"train": len(train_seqs), "cal": len(cal_seqs)}), flush=True)
    if args.mode == "pairwise":
        train_pairwise(train_seqs, cal_seqs, args, args.split)
    else:
        train_set(train_seqs, cal_seqs, args, args.split)


if __name__ == "__main__":
    main()
