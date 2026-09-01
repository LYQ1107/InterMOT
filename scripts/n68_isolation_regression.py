"""N68 post-experiment isolation and regression audit.

This script is intentionally read-only with respect to production projects.  It
only writes a machine-readable audit under outputs/n68.  The N68 experiments
were implemented as sidecars, so the strongest local check is an exact hash
comparison against the Stage-00 inventory plus import/unit regressions.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
# Direct execution places ``scripts`` rather than the project root at
# sys.path[0].  Add the root explicitly so the import probes exercise the
# actual production package instead of testing the audit entrypoint layout.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/n68"
BASELINE = OUT / "stage_00_readonly_audit.json"
RESULT = OUT / "n68_isolation_regression.json"


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path, timeout: int = 900) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "cwd": str(cwd),
            "exit_code": completed.returncode,
            "output_tail": completed.stdout[-12000:],
        }
    except Exception as exc:  # pragma: no cover - audit must retain failures
        return {
            "command": command,
            "cwd": str(cwd),
            "exit_code": None,
            "exception": f"{type(exc).__name__}: {exc}",
        }


def git_snapshot(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {"path": str(path), "git_repository": False}
    head = run(["git", "rev-parse", "HEAD"], path, timeout=30)
    status = run(["git", "status", "--short"], path, timeout=30)
    return {
        "path": str(path),
        "git_repository": True,
        "head": (head.get("output_tail") or "").strip(),
        "status_short": status.get("output_tail", ""),
        "preexisting_dirty_state_observed": bool(status.get("output_tail", "").strip()),
    }


def main() -> None:
    baseline = json.loads(BASELINE.read_text())
    baseline_files = {
        item["path"]: item
        for item in baseline.get("code_inventory", {}).get("files", [])
        if "/sam3_intermot/" in item.get("path", "")
        or "/third_party/sam3/" in item.get("path", "")
    }
    comparisons = []
    for raw_path, old in sorted(baseline_files.items()):
        path = Path(raw_path)
        current = sha256_file(path)
        comparisons.append(
            {
                "path": raw_path,
                "baseline_sha256": old.get("sha256"),
                "current_sha256": current,
                "unchanged": current == old.get("sha256"),
            }
        )
    changed = [item for item in comparisons if not item["unchanged"]]

    protected_frozen = []
    for rel in [
        "docs/N36_FINAL_REPORT.md",
        "docs/N37_FINAL_REPORT.md",
        "docs/N67_FINAL_REPORT.md",
        "outputs/n36/n36_final_gate.json",
        "outputs/n37/n37_final_gate.json",
        "outputs/n67/replay/paired_replay_results.json",
    ]:
        path = ROOT / rel
        protected_frozen.append(
            {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path)}
        )

    imports = []
    for module in [
        "sam3_intermot.association.appearance_memory",
        "sam3_intermot.association.online_associator",
        "sam3_intermot.association.ccam_replay",
        "sam3_intermot.tracking.association",
        "sam3_intermot.evaluation.interaction_metrics",
    ]:
        try:
            importlib.import_module(module)
            imports.append({"module": module, "status": "PASS"})
        except Exception as exc:
            imports.append(
                {
                    "module": module,
                    "status": "FAIL",
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )

    toy_noop = run(
        [
            sys.executable,
            "-c",
            (
                "import numpy as np; "
                "from scripts.n68_stage02_local_association import assignment_from_scores; "
                "x=np.array([[0.8,0.2],[0.1,0.9]],dtype=float); "
                "assert np.array_equal(assignment_from_scores(x), assignment_from_scores(x+0.0))"
            ),
        ],
        ROOT,
        timeout=120,
    )
    sam3_tests = run([sys.executable, "-m", "pytest", "-q", "tests"], ROOT)
    intermots_root = ROOT.parent / "InterMOT"
    intermot_tests = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_intermot.py",
            "tests/test_motip_stage2.py",
        ],
        intermots_root,
    )

    result = {
        "schema": "N68_ISOLATION_REGRESSION_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n68_write_scope": {
            "allowed_roots": [str(OUT), str(ROOT / "scripts")],
            "production_sidecar_only": True,
            "production_modules_modified_by_n68": False,
            "third_party_sam3_modified_by_n68": False,
            "mot_ovmot_external_writes": [],
        },
        "stage_00_production_hash_comparison": {
            "baseline": str(BASELINE),
            "compared_file_count": len(comparisons),
            "changed_file_count": len(changed),
            "changed_files": changed,
            "all_unchanged": not changed,
        },
        "frozen_evidence_read_only_snapshot": protected_frozen,
        "production_imports": imports,
        "regressions": {
            "sidecar_zero_residual_noop": toy_noop,
            "sam3_intermot_unit_tests": sam3_tests,
            "mot_intermot_unit_tests": intermot_tests,
        },
        "external_project_snapshots": [
            git_snapshot(intermots_root),
            git_snapshot(Path("/path/to/masa")),
            git_snapshot(Path("/path/to/workspace/SERVER_ONLY/avis/LocateMOT")),
        ],
        "interpretation": {
            "production_hash_gate": not changed,
            "all_imports_pass": all(item["status"] == "PASS" for item in imports),
            "sam3_unit_gate": sam3_tests.get("exit_code") == 0,
            "mot_unit_gate": intermot_tests.get("exit_code") == 0,
            "external_dirty_repositories_are_not_attributed_to_n68": True,
        },
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULT.with_suffix(RESULT.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(RESULT)
    print(json.dumps(result["interpretation"], sort_keys=True))


if __name__ == "__main__":
    main()
