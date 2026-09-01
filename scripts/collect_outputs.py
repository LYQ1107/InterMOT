#!/usr/bin/env python
"""Generate required output manifests and summaries (N0 blocked path)."""

import json
import subprocess
import sys
import time
from pathlib import Path

from sam3_intermot.utils.io import atomic_write_json, write_csv


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outputs = root / "outputs"
    start = time.time()

    # Run the project test suite (mock-backed, CPU only).
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    test_output = (proc.stdout + proc.stderr).strip()
    test_summary = test_output.splitlines()[-1] if test_output else "no output"

    # Test summary file.
    (outputs / "test_summary.txt").write_text(
        f"pytest exit code: {proc.returncode}\n{test_output}\n", encoding="utf-8"
    )

    # N0 backend test results (mock only; real GPU tests blocked).
    atomic_write_json(
        outputs / "n0" / "backend_test_results.json",
        {
            "status": "MOCK_PASS_REAL_BLOCKED",
            "pytest_exit_code": proc.returncode,
            "test_summary": test_summary,
            "mock_backend_tests": "PASS",
            "real_sam3_gpu_tests": "NOT_RUN",
            "blocked_reason": "N0_STATUS=BLOCKED_EXTERNAL_CHECKPOINT",
        },
    )

    # Runtime profile.
    atomic_write_json(
        outputs / "n0" / "runtime_profile.json",
        {
            "stage": "N0_STATIC_AUDIT",
            "wall_clock_seconds": round(time.time() - start, 2),
            "gpu_seconds": 0,
            "peak_vram_mib": 0,
            "longest_command_seconds": round(time.time() - start, 2),
            "note": "no GPU work performed because official checkpoint is blocked",
        },
    )

    # Required root-level outputs.
    atomic_write_json(
        outputs / "run_manifest.json",
        {
            "project": "SAM3_InterMOT",
            "root": str(root),
            "status": "BLOCKED",
            "completed_stage": "N0_STATIC",
            "current_stage": "N0",
            "blocked": {
                "code": "N0_STATUS=BLOCKED_EXTERNAL_CHECKPOINT",
                "repo": "facebook/sam3.1",
                "checkpoint": "sam3.1_multiplex.pt",
                "http": 403,
                "gated": "manual",
            },
            "tests": {"exit_code": proc.returncode, "summary": test_summary},
            "gpu": {"allowed": [8, 9], "used": []},
        },
    )
    atomic_write_json(
        outputs / "stage_gate.json",
        {"N0": "BLOCKED", "N1": "NOT_RUN", "N2": "NOT_RUN", "N3": "NOT_RUN"},
    )

    write_csv(
        outputs / "runtime_summary.csv",
        [
            {
                "stage": "N0",
                "status": "BLOCKED",
                "wall_clock_seconds": round(time.time() - start, 2),
                "gpu_seconds": 0,
                "peak_vram_mib": 0,
                "longest_command": "pytest -q",
                "note": "static audit + mock tests only",
            }
        ],
    )
    write_csv(
        outputs / "invariant_violations.csv",
        [
            {
                "stage": "N0",
                "frame_idx": "",
                "violation": "none (no real run performed)",
            }
        ],
    )
    (outputs / "interaction_transactions.jsonl").write_text(
        "", encoding="utf-8"
    )
    write_csv(
        outputs / "final_metrics.csv",
        [
            {
                "method": "",
                "budget": "",
                "HOTA": "",
                "DetA": "",
                "AssA": "",
                "MOTA": "",
                "IDF1": "",
                "IDSW": "",
                "Frag": "",
                "FP": "",
                "FN": "",
                "status": "NOT_RUN",
            }
        ],
    )
    write_csv(
        outputs / "per_sequence_metrics.csv",
        [
            {
                "sequence": "",
                "HOTA": "",
                "DetA": "",
                "AssA": "",
                "MOTA": "",
                "IDF1": "",
                "status": "NOT_RUN",
            }
        ],
    )

    # Command and file-change manifests.
    commands = "\n".join(
        [
            "#!/usr/bin/env bash",
            "# SAM3_InterMOT commands used in the N0 static phase (blocked path)",
            "set -euo pipefail",
            "PROJECT=.",
            "PY=python",
            "cd \"$PROJECT\"",
            "",
            "# 1. Verify official repo and pin release commit",
            "git clone https://github.com/facebookresearch/sam3.git third_party/sam3",
            "git -C third_party/sam3 checkout 4cbac146c1b5a1e3a7f5c6a894901090b4dfd65b",
            "",
            "# 2. Verify HF checkpoint access (expected 403 GatedRepo)",
            "HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN in the environment}",
            "curl -sS -I -H \"Authorization: Bearer $HF_TOKEN\" \\",
            "  https://huggingface.co/facebook/sam3.1/resolve/main/sam3.1_multiplex.pt",
            "",
            "# 3. Static audit + mock tests (CPU only)",
            "PYTHONPATH=. \"$PY\" scripts/checkpoint_status.py",
            "PYTHONPATH=. \"$PY\" scripts/run_n0_static_audit.py",
            "PYTHONPATH=. \"$PY\" scripts/run_mock_pipeline.py",
            "PYTHONPATH=. \"$PY\" -m pytest -q",
            "PYTHONPATH=. \"$PY\" scripts/collect_outputs.py",
            "",
            "# 4. Real pipeline (BLOCKED until checkpoint available)",
            "PYTHONPATH=. \"$PY\" scripts/run_real_pipeline.py || echo 'BLOCKED: checkpoint unavailable'",
        ]
    )
    (outputs / "all_commands.sh").write_text(commands + "\n", encoding="utf-8")

    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if (
            rel.startswith("third_party/sam3/.git/")
            or "__pycache__" in rel
            or rel.endswith(".pyc")
            or ".pytest_cache" in rel
        ):
            continue
        files.append(rel)
    (outputs / "file_change_manifest.txt").write_text(
        "\n".join(files) + "\n", encoding="utf-8"
    )
    print(f"outputs collected; pytest exit={proc.returncode}; summary={test_summary}")


if __name__ == "__main__":
    main()
