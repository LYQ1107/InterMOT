#!/usr/bin/env python3
"""Share byte-identical sealed R4 outputs without removing any result path.

Only files named in completed R4 export manifests are eligible. Historical
assets, checkpoints, source data and in-progress outputs are never targets.
The content SHA and all original manifests remain unchanged. This is a
storage operation, not a policy, model, trace or trajectory modification.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from scripts.n72r20r4_common import ASSETS, OUT, plain, read_json, sha256, write_json


def share_identical_exports(asset_root: Path, manifest_paths: list[Path]) -> dict:
    root = asset_root.resolve(strict=True)
    canonical = {}
    files = {}
    for manifest_path in sorted(manifest_paths):
        manifest = read_json(manifest_path)
        for field in ("trajectory", "trace"):
            if f"{field}_path" not in manifest:
                continue
            path = Path(manifest[f"{field}_path"])
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("dedup target outside the current stage or symlink")
            expected = manifest[f"{field}_sha256"]
            if path in files and files[path] != expected:
                raise ValueError("conflicting export manifests")
            files[path] = expected
    records = []
    freed_blocks = 0
    for path, expected in sorted(files.items()):
        if sha256(path) != expected:
            raise ValueError(f"sealed export SHA mismatch: {path}")
        source = canonical.setdefault(expected, path)
        if source == path or os.path.samefile(source, path):
            continue
        if source.read_bytes() != path.read_bytes():
            raise ValueError("equal SHA with unequal bytes")
        before = path.stat()
        temporary = path.parent / (".r4-identical-" + uuid.uuid4().hex)
        os.link(source, temporary)
        try:
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        if sha256(path) != expected or not os.path.samefile(source, path):
            raise RuntimeError("hardlink verification failed")
        freed_blocks += before.st_blocks if before.st_nlink == 1 else 0
        records.append({"path": str(path), "identical_source": str(source), "sha256": expected})
    return {"sealed_files_checked": len(files), "new_shared_paths": len(records),
            "reclaimed_bytes": freed_blocks * 512, "records": records,
            "all_original_result_paths_preserved": True,
            "all_content_SHA_unchanged": True, "historical_assets_touched": False}


def run() -> dict:
    # An export manifest is written last, after trace and trajectory hashing.
    manifests = [p for p in ASSETS.rglob("*.json")
                 if "manifests" in p.parts and "failed_runs" not in p.parts]
    result = share_identical_exports(ASSETS, manifests)
    ledger_path = OUT / "audit/STORAGE_SHARING.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {"operations": []}
    ledger["operations"].append(result)
    write_json(ledger_path, ledger)
    return result


if __name__ == "__main__":
    import json
    result = run()
    print(json.dumps(plain({k: v for k, v in result.items() if k != "records"}), sort_keys=True))
