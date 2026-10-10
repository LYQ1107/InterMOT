"""Actual terminal-status regression receipts; retain historical failures."""
import os
import subprocess
import xml.etree.ElementTree as ET
from scripts.n72r21r2_common import ROOT, OUT, PYTHON, GOAL, read_json, write_json, sha256, append_log


def run_one(name, paths):
    log = OUT / "tests" / (name + ".log")
    xml = log.with_suffix(".xml")
    if log.exists() or xml.exists():
        raise FileExistsError("Preserve prior regression attempts")
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PATH": str(PYTHON.parent) + os.pathsep + os.environ.get("PATH", ""),
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with log.open("x") as handle:
        completed = subprocess.run([str(PYTHON), "-m", "pytest", "-q", *paths, "--junitxml=" + str(xml)],
                                   cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    tree = ET.parse(xml)
    suites = list(tree.getroot().iter("testsuite"))
    tests = sum(int(s.get("tests", 0)) for s in suites)
    failed = sum(int(s.get("failures", 0)) for s in suites)
    errors = sum(int(s.get("errors", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    failed_tests = sorted(c.get("classname") + "::" + c.get("name") for c in tree.iter("testcase") if c.find("failure") is not None)
    return {"passed": tests - failed - errors - skipped, "failed": failed, "errors": errors, "skipped": skipped,
            "seconds": sum(float(s.get("time", 0)) for s in suites), "failed_tests": failed_tests,
            "actual_shell_exit_code": completed.returncode, "full_pytest_process_exit_status_observed": True,
            "all_passed": completed.returncode == 0 and failed == errors == 0,
            "artifacts": {str(p): sha256(p) for p in (log, xml)}}


def run():
    if (OUT / "tests/REGRESSION_V19.json").exists():
        raise FileExistsError("Never replace a regression receipt")
    focused_paths = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "tests").glob("test_n72r21r2_*.py"))]
    focused = run_one("R2_FOCUSED_OPTIMIZER_STATE_V19", focused_paths)
    print({"actual_focused_terminal": focused}, flush=True)
    full = run_one("ALL_REGRESSION_OPTIMIZER_STATE_V19", ["tests"])
    previous = read_json(OUT / "tests/REGRESSION_V18.json")
    unchanged = sorted(full["failed_tests"]) == sorted(previous["failed_tests"])
    receipt = {"stage": "N72R21R2", "goal": GOAL, **full,
        "focused": {k: v for k, v in focused.items() if k != "artifacts"},
        "artifacts": {**full["artifacts"], **focused["artifacts"]},
        "same_failed_tests_as_V18": unchanged, "no_historical_failures_hidden_or_tests_modified": True,
        "unchanged_failure_classes": previous["unchanged_failure_classes"] if unchanged else ["ACTUAL_FAILURE_SET_CHANGED_REQUIRES_DIAGNOSIS"],
        "scientific_stage_complete": False}
    write_json("tests/REGRESSION_V19.json", receipt)
    append_log("ACTUAL_REGRESSION_V19_TERMINAL_STATUS", passed=full["passed"], failed=full["failed"],
               actual_exit_code=full["actual_shell_exit_code"], focused_passed=focused["passed"], same_failed_tests=unchanged)
    print({"actual_full_terminal": receipt}, flush=True)


if __name__ == "__main__":
    run()
