"""N69 post-training isolation and targeted regression audit.

This script only writes ``outputs/n69/n69_isolation_regression.json``.  It
compares the Stage00 hashes for production/third-party/configuration trees and
frozen N36--N68 evidence, then runs import and regression checks in the
existing project environments.  N69 sidecar scripts and outputs are the only
new write scope.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/n69"
BASELINE = OUT / "stage_00_readonly_audit.json"
RESULT = OUT / "n69_isolation_regression.json"
TEST_PYTHON = Path("python") if Path("python").is_file() else Path(sys.executable)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path, timeout: int = 1800) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=timeout)
        return {"command": command, "cwd": str(cwd), "exit_code": proc.returncode, "output_tail": proc.stdout[-16000:]}
    except Exception as exc:
        return {"command": command, "cwd": str(cwd), "exit_code": None, "exception": f"{type(exc).__name__}: {exc}"}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    comparisons: list[dict[str, Any]] = []
    for inventory_name in ("configs", "environment", "sam3_intermot", "third_party_sam3"):
        inventory = baseline.get("code_inventory", {}).get(inventory_name, {})
        for item in inventory.get("files", []):
            raw = item.get("path")
            path = Path(raw) if os.path.isabs(raw) else ROOT / str(raw)
            current = sha256(path)
            comparisons.append({"path": str(path), "baseline_sha256": item.get("sha256"), "current_sha256": current, "unchanged": current == item.get("sha256")})
    frozen: list[dict[str, Any]] = []
    for item in baseline.get("frozen_inputs", []):
        path = Path(item["path"])
        current = sha256(path)
        frozen.append({"path": str(path), "baseline_sha256": item.get("sha256"), "current_sha256": current, "unchanged": current == item.get("sha256")})
    protected: list[dict[str, Any]] = []
    for item in baseline.get("protected_files", []):
        path = Path(item["path"])
        current = sha256(path)
        protected.append({"path": str(path), "baseline_sha256": item.get("sha256"), "current_sha256": current, "unchanged": current == item.get("sha256")})

    imports: list[dict[str, Any]] = []
    for module in (
        "sam3_intermot.association.appearance_memory",
        "sam3_intermot.association.online_associator",
        "sam3_intermot.association.ccam_replay",
        "sam3_intermot.backend.base",
        "scripts.n69_mapping_contract",
        "scripts.n69_stage03_target_conditioned",
        "scripts.n69_stage06_protected_guard",
    ):
        try:
            importlib.import_module(module)
            imports.append({"module": module, "status": "PASS"})
        except Exception as exc:
            imports.append({"module": module, "status": "FAIL", "exception": f"{type(exc).__name__}: {exc}"})

    current_tests = run([str(TEST_PYTHON), "-m", "pytest", "-q", "tests"], ROOT)
    adjacent_root = ROOT.parent / "InterMOT"
    adjacent_tests = run([str(TEST_PYTHON), "-m", "pytest", "-q", "tests/test_intermot.py", "tests/test_motip_stage2.py"], adjacent_root) if adjacent_root.is_dir() else {"status": "SKIPPED_NO_ADJACENT_PROJECT"}
    compile_check = run([sys.executable, "-m", "py_compile", "scripts/n69_stage03_target_conditioned.py", "scripts/n69_stage06_protected_guard.py", "scripts/n69_isolation_regression.py"], ROOT, timeout=120)

    changed_production = [item for item in comparisons if not item["unchanged"]]
    changed_frozen = [item for item in frozen if not item["unchanged"]]
    changed_protected = [item for item in protected if not item["unchanged"]]
    interpretation = {
        "production_tree_unchanged": not changed_production,
        "frozen_n36_to_n68_unchanged": not changed_frozen,
        "protected_checkpoints_unchanged": not changed_protected,
        "imports_pass": all(item.get("status") == "PASS" for item in imports),
        "sam3_intermot_tests_pass": current_tests.get("exit_code") == 0,
        "adjacent_mot_tests_pass": adjacent_tests.get("exit_code") == 0,
        "new_scripts_compile": compile_check.get("exit_code") == 0,
        "production_authorized": False,
    }
    result = {
        "schema": "N69_ISOLATION_REGRESSION_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "write_scope": {"allowed_roots": [str(OUT), str(ROOT / "scripts")], "production_paths_modified": False, "third_party_sam3_modified": False, "mot_ovmot_locatemot_masa_writes": []},
        "stage00_hash_comparison": {"file_count": len(comparisons), "changed_count": len(changed_production), "changed_files": changed_production},
        "frozen_evidence_hash_comparison": {"file_count": len(frozen), "changed_count": len(changed_frozen), "changed_files": changed_frozen},
        "protected_checkpoint_hash_comparison": {"file_count": len(protected), "changed_count": len(changed_protected), "changed_files": changed_protected},
        "imports": imports,
        "regression_interpreter": str(TEST_PYTHON),
        "regressions": {"sam3_intermot": current_tests, "adjacent_mot_ovmot": adjacent_tests, "new_scripts_compile": compile_check},
        "interpretation": interpretation,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "runtime_future_gt_used": False,
        "production_authorized": False,
    }
    atomic_json(RESULT, result)
    print(json.dumps(interpretation, sort_keys=True))


if __name__ == "__main__":
    main()
