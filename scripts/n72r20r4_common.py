"""R4 paths, immutable provenance and bounded external result storage."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from scripts.n72r20r1_build_base_score_tape import load_candidate_frames, read_zstd_jsonl

ROOT = Path(__file__).resolve().parents[1]
STAGE = "N72R20R4"
OUT = ROOT / "outputs" / STAGE
ASSETS = Path(os.environ.get("N72R20R4_ASSET_ROOT", str(ROOT.parent / "InterMOT_N72R20R4_assets")))
DEV_ROOT = ROOT.parent / "InterMOT_N72R20R2_assets"
VAL_ROOT = ROOT.parent / "InterMOT_N72R20R3R3_assets" / "val_candidates"
DATASET = Path(os.environ.get("DANCETRACK_ROOT", str(ROOT.parent / "InterMOT_N72R16_assets/dataset")))
SEQUENCES = ("dancetrack0001", "dancetrack0002", "dancetrack0023", "dancetrack0024", "dancetrack0039", "dancetrack0057", "dancetrack0062", "dancetrack0072")
SEEDS = (720321, 720322, 720323)
MEMORY_CHECKPOINT = ROOT / "outputs/N72R18/checkpoints/identity_memory_gru.pt"
MEMORY_SHA = "94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2"
ENCODER_SHA = "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154"


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_zstd(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        process = subprocess.Popen(["zstd", "-q", "-c", "-T1", "-3"], stdin=subprocess.PIPE, stdout=output)
        try:
            for row in rows:
                process.stdin.write((json.dumps(plain(row), sort_keys=True, allow_nan=False) + "\n").encode())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("zstd output failure")
        finally:
            if process.poll() is None:
                process.terminate()
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def code_manifest() -> dict[str, str]:
    files = sorted(ROOT.glob("scripts/n72r20r4*.py")) + sorted(ROOT.glob("sam3_intermot/association/causal*.py")) + [ROOT / "sam3_intermot/association/identity_authority.py", ROOT / "sam3_intermot/association/global_assignment_adapter.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in files if p.exists()}


def check_storage(*, reserve_gib: float = 0) -> dict[str, Any]:
    free = shutil.disk_usage(ROOT).free / (1 << 30)
    status = "STOP_HEAVY" if free - reserve_gib < 100 else "WARNING" if free < 105 else "PASS"
    if status == "STOP_HEAVY":
        raise RuntimeError(f"FAIL_ASSET_OR_STORAGE: free={free:.3f} GiB, reserved={reserve_gib}; minimum 100 GiB")
    return {"free_gib": free, "reserve_gib": reserve_gib, "status": status}


def events(split: str = "train") -> dict[str, dict[str, Any]]:
    if split == "train":
        path = DEV_ROOT / "interaction_events.json"
    elif split == "val":
        path = VAL_ROOT.parent / "val_events.json"
    else:
        raise ValueError("only train and frozen val are authorized")
    return {str(e["sequence"]): e for e in read_json(path)["events"]}


def load_frames(sequence: str, split: str = "train") -> list:
    if split not in {"train", "val"}:
        raise ValueError("test is not authorized")
    frames = load_candidate_frames(DEV_ROOT if split == "train" else VAL_ROOT, sequence)
    rejected = []
    result = []
    for payload, rows in frames:
        valid = []
        for row in rows:
            box = np.asarray(row["box_xyxy"], dtype=float)
            if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                rejected.append({"frame": payload["frame"], "candidate_uid": row["candidate_uid"], "box_xyxy": row["box_xyxy"], "reason": "zero_or_invalid_area_before_solver", "GT_used": False})
                continue
            valid.append(row)
        result.append((payload, valid))
    if rejected:
        write_json(OUT / "audit/input_geometry" / f"{split}__{sequence}.json", {"sequence": sequence, "split": split, "rejected": rejected, "same_policy_all_variants": True, "raw_assets_modified": False, "posthoc_box_clipping": False})
    return result


def historical_hashes() -> dict[str, str]:
    result = {}
    for stage in ("N72R20R3R2", "N72R20R3R2R1", "N72R20R3R2R2", "N72R20R3R2R3"):
        for p in sorted((ROOT / "outputs" / stage).rglob("*")):
            if p.is_file() and p.suffix in {".json", ".md", ".zst"}:
                result[str(p.relative_to(ROOT))] = sha256(p)
    return result
