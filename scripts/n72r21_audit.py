"""Read-only local discovery, history seals and resource checks for N72R21."""
from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import pwd
import subprocess

from scripts.n72r21_common import ROOT, OUT, ASSETS, DATA, read_json, write_json, sha256, storage, utcnow

STAGES = ("N72R18", "N72R20", "N72R20R3", "N72R20R3R1", "N72R20R3R2", "N72R20R3R2R1",
          "N72R20R3R2R2", "N72R20R3R2R3", "N72R20R4", "N72R20R4R1")


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=180)
    return {"command": args, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def run():
    before = storage()
    before.update({"owner": getpass.getuser(), "inodes": command(["df", "-i", str(ROOT)]),
                   "mounts": command(["findmnt", "-T", str(ROOT)]),
                   "gpus": command(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]),
                   "own_processes": command(["ps", "-u", getpass.getuser(), "-o", "pid,etimes,comm"]),
                   "allocated_gpu_not_inferred_from_visibility": True,
                   "proxy_variable_names_present": [k for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy") if os.environ.get(k)]})
    write_json("storage/STORAGE_BEFORE.json", before)
    snapshots = {}
    history = {}
    manifests = []
    for stage in STAGES:
        source = ROOT / "outputs" / stage
        report = source / "FINAL_REPORT.md"
        if not report.exists():
            report = ROOT / "docs" / f"{stage}_FINAL_REPORT.md"
        result = source / "FINAL_RESULT.json"
        status = source / "stage_status.json"
        doc = read_json(result if result.exists() else status)
        history[stage] = {"report": str(report.relative_to(ROOT)), "result": str(result.relative_to(ROOT)) if result.exists() else None,
                          "status_fallback": str(status.relative_to(ROOT)) if not result.exists() else None,
                          "decision": doc.get("final_decision", doc.get("decision", doc.get("status"))),
                          "result_keys": list(doc),
                          "scalar_fields": {k:v for k,v in doc.items() if v is None or isinstance(v, (str, bool, int, float))}}
        for path in source.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".md"}:
                snapshots[str(path.relative_to(ROOT))] = sha256(path)
                if "checkpoint" in path.name.lower() or "sha256" in path.name.lower() or path.name == "STRICT_FOLD_CHECKPOINTS.json":
                    value = read_json(path) if path.suffix == ".json" else None
                    manifests.append({"path": str(path.relative_to(ROOT)), "sha256": snapshots[str(path.relative_to(ROOT))],
                                      "keys": list(value) if isinstance(value, dict) else None})
        for path in (report, ROOT / "docs" / f"{stage}_FINAL_REPORT.md"):
            if path.exists(): snapshots[str(path.relative_to(ROOT))] = sha256(path)
    # Core historical code is sealed before any new implementation.
    for pattern in ("scripts/n72r18*.py", "scripts/n72r20*.py", "sam3_intermot/association/*.py", "sam3_intermot/identity_verification/*.py"):
        for path in ROOT.glob(pattern): snapshots[str(path.relative_to(ROOT))] = sha256(path)
    write_json("historical_audit/HISTORY_BEFORE.json", {"utc": utcnow(), "sha256": snapshots, "count": len(snapshots)})
    write_json("historical_audit/HISTORICAL_RESULTS.json", {"stages": history, "checkpoint_manifests": manifests})
    # Do not print another user's workload arguments or read credential files.
    candidates = command(["find", "/data3", "/data2", "/data1", "/home", "-maxdepth", "6", "-type", "d", "(", "-iname", "*chirla*", "-o", "-iname", "*lasot*", "-o", "-iname", "*personpath*", "-o", "-iname", "*person-path*", "-o", "-iname", "*dancetrack*", ")", "-print"])
    paths = [s for s in candidates["stdout"].splitlines() if s]
    raw_roots = [s for s in paths if "/TAO-Amodal/frames/" in s or "chirla" in s.lower() or "personpath" in s.lower()]
    archives = command(["find", str(ROOT.parent), "-maxdepth", "5", "-type", "f", "(", "-iname", "*chirla*", "-o", "-iname", "*lasot*.zip", "-o", "-iname", "*lasot*.tar*", "-o", "-iname", "*personpath*.zip", ")", "-print"])
    dance = ROOT.parent / "InterMOT_N72R16_assets/dataset"
    dance_check = {}
    for split in ("train", "val"):
        sequences = sorted((dance / split).glob("dancetrack*"))
        dance_check[split] = [{"sequence": p.name, "img1_readable": os.access(p / "img1", os.R_OK),
                               "frames": sum(1 for f in (p / "img1").iterdir() if f.is_file()),
                               "gt_readable": os.access(p / "gt/gt.txt", os.R_OK)} for p in sequences]
    lasot = []
    tao = ROOT.parent / "TAO-Amodal/frames"
    for split in ("train", "val", "test"):
        for p in sorted((tao / split / "LaSOT").glob("person-*")):
            files = [f for f in p.iterdir() if f.is_file()]
            lasot.append({"path": str(p), "realpath": str(p.resolve()), "split": split, "sequence": p.name,
                          "owner": pwd.getpwuid(p.stat().st_uid).pw_name, "frames": len(files),
                          "bytes": sum(f.stat().st_size for f in files), "original_lasot_gt_present": (p / "groundtruth.txt").exists(),
                          "status": "TAO_DERIVED_SUBSET_NOT_COMPLETE_LASOT", "used_for_training": False})
    write_json("datasets/DATASET_DISCOVERY.json", {"utc": utcnow(), "searched_roots": ["/data3", "/data2", "/data1", "/home"],
               "maxdepth": 6, "find_exit_code": candidates["exit_code"], "find_errors": candidates["stderr"],
               "candidate_paths": paths, "raw_candidate_roots": raw_roots, "archive_candidates": archives["stdout"].splitlines(),
               "dancetrack_root": str(dance), "dancetrack_integrity": dance_check,
               "lasot_person_existing": lasot, "new_data_copies": 0, "heldout_contents_used_for_development": False})
    write_json("storage/CLEANUP_DRY_RUN.json", {"status": "NO_CLEANUP_REQUIRED", "targets": [], "reason": "Bounded pilot/metadata budget fits current mount reserve; no deletion of other projects or historical evidence."})
    write_json("storage/CLEANUP_EXECUTED.json", {"status": "NO_CLEANUP_REQUIRED", "deleted": [], "deleted_bytes": 0, "before": before["free_bytes"], "after": storage()["free_bytes"]})
    print(json.dumps({"history_files_sealed": len(snapshots), "dataset_candidate_paths": len(paths), "lasot_person_subsets": len(lasot), "dancetrack_train_val": [len(dance_check[s]) for s in ("train", "val")], "free_gib": before["free_gib"]}))


if __name__ == "__main__": run()
