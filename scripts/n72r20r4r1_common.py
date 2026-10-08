"""R4R1-only writes, frozen source lineage and small persistent artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import torch

from scripts.n72r20r4_common import plain, read_json, write_json, sha256, SEQUENCES, SEEDS, DATASET, DEV_ROOT, VAL_ROOT, MEMORY_CHECKPOINT, MEMORY_SHA, ENCODER_SHA
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames, read_zstd_jsonl
from scripts.n72r20r4_adapter_integration import fold_split, strict_ensemble
from scripts.n72r20r4_run_causal_tracker import make_bank, trajectory_text

ROOT=Path(__file__).resolve().parents[1]
STAGE="N72R20R4R1"
OUT=ROOT/"outputs"/STAGE
ASSETS=Path(os.environ.get("N72R20R4R1_ASSET_ROOT",str(ROOT.parent/"InterMOT_N72R20R4R1_assets")))
R4OUT=ROOT/"outputs/N72R20R4"
R4ASSETS=ROOT.parent/"InterMOT_N72R20R4_assets"
SOURCE_SHA="36b0e198fa084ce1288fd134736bea159a2ea731"
BRANCH="codex/n72r20r4r1-global-opportunity-safe-authority"


def check_storage(reserve_mib: float=0) -> dict[str,Any]:
    free=shutil.disk_usage(ROOT).free
    owned=sum(p.stat().st_size for p in ASSETS.rglob("*") if p.is_file()) if ASSETS.exists() else 0
    amended=(OUT/"RESOURCE_POLICY_AMENDMENT_01.json").exists()
    bounded_amendment=OUT/'RESOURCE_POLICY_AMENDMENT_02.json'
    hard_floor=read_json(bounded_amendment)['lightweight_hard_floor_gib'] if bounded_amendment.exists() else 99 if amended else 100
    if free-reserve_mib*(1<<20)<hard_floor*(1<<30) or (free<100*(1<<30) and reserve_mib>=1):
        raise RuntimeError(f"FAIL_ASSET_OR_STORAGE: {free/(1<<30):.4f} GiB free; no new heavy cache")
    if owned+reserve_mib*(1<<20)>48*(1<<20):
        raise RuntimeError("R4R1 persistent 48 MiB target exceeded; use streaming/own regenerable temporary files")
    return {"free_bytes":free,"free_gib":free/(1<<30),"own_stage_bytes":owned,"reserved_mib":reserve_mib,
        "status":"LIGHTWEIGHT_ONLY" if free<100*(1<<30) else "WARNING" if free<105*(1<<30) else "PASS",
        "heavy_cache_forbidden":free<100*(1<<30),"resource_amendment":amended}


def events(split: str="train") -> dict[str,dict]:
    if split=="val":
        policy=OUT/"val/FROZEN_POLICY.json"
        if not policy.exists() or not read_json(policy).get("development_gate_pass"):
            raise ValueError("R4R1 VAL requires a frozen passing development policy")
        source=VAL_ROOT.parent/"val_events.json"
    elif split=="train":source=DEV_ROOT/"interaction_events.json"
    else:raise ValueError("TEST and other splits are not authorized")
    return {str(e["sequence"]):e for e in read_json(source)["events"]}


def load_frames(sequence: str, split: str="train") -> list:
    if split=="val":events("val")
    elif split!="train" or sequence not in SEQUENCES:raise ValueError("unregistered training sequence/split")
    result=[];rejected=[]
    for payload,rows in load_candidate_frames(DEV_ROOT if split=="train" else VAL_ROOT,sequence):
        valid=[]
        for row in rows:
            box=np.asarray(row["box_xyxy"],dtype=float)
            if not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:
                rejected.append({"frame":payload["frame"],"candidate_uid":row["candidate_uid"],"box_xyxy":row["box_xyxy"]});continue
            valid.append(row)
        result.append((payload,valid))
    if rejected:write_json(OUT/"audit/input_geometry"/f"{split}__{sequence}.json",{"rejected":rejected,"same_as_R4_pre_solver_policy":True,"historical_assets_modified":False})
    return result


def assert_fit_sequence(sequence: str, heldout: str) -> None:
    if sequence not in fold_split(heldout)[0]:raise ValueError("training sequence overlaps inner/outer or is not registered")


def source_hashes() -> dict[str,str]:
    paths=[]
    for stage in ("N72R20R3R2","N72R20R3R2R1","N72R20R3R2R2","N72R20R3R2R3","N72R20R4"):
        paths.extend(p for p in (ROOT/"outputs"/stage).rglob("*") if p.is_file() and p.suffix in {".json",".md",".zst"})
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}


def code_manifest() -> dict[str,str]:
    paths=sorted(ROOT.glob("scripts/n72r20r4r1*.py"))+sorted((ROOT/"sam3_intermot/association").glob("opportunity*.py"))
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def stream_zstd(path: Path, rows) -> dict:
    """Single-thread compression; never accumulate the entire JSONL in RAM."""
    if path.exists():raise FileExistsError(f"immutable completed stream exists: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    count=0;digest=hashlib.sha256()
    with temporary.open("wb") as output:
        process=subprocess.Popen(["zstd","-q","-c","-T1","-3"],stdin=subprocess.PIPE,stdout=output)
        try:
            for row in rows:
                encoded=(json.dumps(plain(row),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
                digest.update(encoded);process.stdin.write(encoded);count+=1
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError("zstd failed")
        finally:
            if process.poll() is None:process.terminate();process.wait()
    temporary.replace(path)
    return {"path":str(path),"rows":count,"sha256":sha256(path),"uncompressed_sha256":digest.hexdigest()}
