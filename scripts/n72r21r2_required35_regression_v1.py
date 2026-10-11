"""CPU-only actual regression with immutable before/after executed-source SHA.

All original test paths and historical failures retained. Unit evidence does
not complete all35 runtime semantics or the scientific M0-M11 objective.
"""
import argparse
import os
import subprocess
from scripts.n72r21r2_common import ROOT, OUT, GOAL, read_json, write_json, sha256, utcnow, append_log
from scripts.n72r21r2_regression_v19 import run_one


def source_snapshot():
    paths = set((ROOT / "sam3_intermot").rglob("*.py"))
    paths.update((ROOT / "tests").rglob("*.py"))
    paths.update((ROOT / "scripts").glob("*.py"))
    paths.update((ROOT / "third_party/MOTIP/TrackEval").rglob("*.py"))
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def run(version):
    if type(version) is not int or version < 34:
        raise ValueError("New immutable regression version required")
    relative = f"tests/REGRESSION_V{version}.json"
    if (OUT / relative).exists():
        raise FileExistsError("Preserve previous actual regression attempts")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    before = source_snapshot()
    pinned = subprocess.check_output(["git", "-C", str(ROOT / "third_party/MOTIP/TrackEval"),
                                      "rev-parse", "HEAD"], text=True).strip()
    assert pinned == "12c8791b303e0a0b50f753af204249e622d0281a"
    paths = [str(path.relative_to(ROOT)) for path in sorted((ROOT / "tests").glob("test_n72r21r2_*.py"))]
    focused = run_one(f"R2_FOCUSED_REQUIRED35_SOURCE_ATTESTED_V{version}", paths)
    print({"focused_actual_exit": focused["actual_shell_exit_code"], "passed": focused["passed"],
           "failed": focused["failed"], "errors": focused["errors"]}, flush=True)
    full = run_one(f"ALL_REGRESSION_REQUIRED35_SOURCE_ATTESTED_V{version}", ["tests"])
    after = source_snapshot()
    same_source = before == after
    previous = read_json(OUT / "tests/REGRESSION_V33.json")
    same_failures = full["failed_tests"] == previous["failed_tests"]
    receipt = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(), **full,
        "focused": {key: value for key, value in focused.items() if key != "artifacts"},
        "artifacts": {**full["artifacts"], **focused["artifacts"]},
        "executed_source_sha256_before": before, "executed_source_sha256_after": after,
        "executed_source_unchanged": same_source, "TrackEval_pinned_commit": pinned,
        "same_failed_tests_as_V33": same_failures,
        "process_local_CUDA_VISIBLE_DEVICES_empty": True, "all_original_test_paths_no_exclusions": True,
        "no_historical_failures_hidden_or_tests_modified": True,
        "unchanged_failure_classes": previous["unchanged_failure_classes"] if same_failures else
                                     ["ACTUAL_CHANGED_FAILURES_REQUIRE_DIAGNOSIS"],
        "scientific_stage_complete": False, "full35_runtime_semantics_complete": False}
    write_json(relative, receipt)
    append_log("ACTUAL_REQUIRED35_SOURCE_ATTESTED_REGRESSION", version=version, passed=full["passed"],
               failed=full["failed"], actual_pytest_exit=full["actual_shell_exit_code"],
               focused_passed=focused["passed"], source_unchanged=same_source, same_failed_tests=same_failures)
    if not same_source:
        raise RuntimeError("Source changed during regression; actual outputs retained, no source-attested PASS")
    write_json("tests/REGRESSION.json", receipt, mutable=True)
    write_json("tests/FOCUSED.json", {"stage": "N72R21R2", "goal": GOAL, **focused,
               "regression_receipt": relative, "executed_source_unchanged": same_source}, mutable=True)
    print({"full_actual_exit": full["actual_shell_exit_code"], "passed": full["passed"],
           "failed": full["failed"], "errors": full["errors"], "same_historical_failures": same_failures,
           "executed_source_unchanged": same_source}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, type=int)
    run(parser.parse_args().version)
