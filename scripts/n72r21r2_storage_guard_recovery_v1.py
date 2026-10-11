"""Evidence-only lifecycle guard; does not change the frozen 60GiB limit."""
from pathlib import Path

STORAGE_ERROR = "RuntimeError: R2 would violate the60GiB actual filesystem reserve"


def dead_idle_storage_owner(marker, log):
    pid = marker.get("pid")
    if type(pid) is not int or pid <= 0 or Path("/proc", str(pid)).exists():
        raise RuntimeError("Live, reused or invalid PID: never replace this owner")
    if marker.get("active") is not None:
        raise RuntimeError("Active/unsealed child requires a different explicit recovery")
    text = Path(log).read_text()
    if not text.rstrip().endswith(STORAGE_ERROR) or "Traceback (most recent call last):" not in text:
        raise RuntimeError("Missing actual storage-guard traceback; do not infer a cause")
    return {"old_pid_absent": True, "active_child_marker": None,
            "actual_storage_guard_traceback_observed": True,
            "actual_old_parent_exit_status_observed": False}
