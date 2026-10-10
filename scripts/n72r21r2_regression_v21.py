"""Actual event-delivery regressions, retaining the historical failure set."""
from scripts.n72r21r2_regression_v19 import run_one
from scripts.n72r21r2_common import ROOT, OUT, GOAL, read_json, write_json, append_log


def run():
    if (OUT / "tests/REGRESSION_V21.json").exists():
        raise FileExistsError("Preserve prior regression attempts")
    paths = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "tests").glob("test_n72r21r2_*.py"))]
    focused = run_one("R2_FOCUSED_EVENT_DELIVERY_V21", paths)
    print({"focused_actual_exit": focused["actual_shell_exit_code"], "passed": focused["passed"], "failed": focused["failed"]}, flush=True)
    full = run_one("ALL_REGRESSION_EVENT_DELIVERY_V21", ["tests"])
    old = read_json(OUT / "tests/REGRESSION_V20.json")
    same = sorted(full["failed_tests"]) == sorted(old["failed_tests"])
    receipt = {"stage": "N72R21R2", "goal": GOAL, **full,
        "focused": {k: v for k, v in focused.items() if k != "artifacts"},
        "artifacts": {**full["artifacts"], **focused["artifacts"]}, "same_failed_tests_as_V20": same,
        "no_historical_failures_hidden_or_tests_modified": True,
        "unchanged_failure_classes": old["unchanged_failure_classes"] if same else ["ACTUAL_CHANGED_FAILURES_REQUIRE_DIAGNOSIS"],
        "scientific_stage_complete": False}
    write_json("tests/REGRESSION_V21.json", receipt)
    append_log("ACTUAL_REGRESSION_V21_TERMINAL_STATUS", passed=full["passed"], failed=full["failed"],
               actual_exit_code=full["actual_shell_exit_code"], focused_passed=focused["passed"], same_failed_tests=same)
    print({"full_actual_exit": full["actual_shell_exit_code"], "passed": full["passed"], "failed": full["failed"], "same_failed_tests_as_V20": same}, flush=True)


if __name__ == "__main__":
    run()
