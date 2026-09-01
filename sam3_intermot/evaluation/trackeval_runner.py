"""Official TrackEval integration (read-only reuse of existing InterMOT)."""

import subprocess
from pathlib import Path
from typing import Optional


def find_trackeval_root(existing_intermot_root: str) -> Optional[str]:
    root = Path(existing_intermot_root)
    candidates = [
        root / "TrackEval",
        root / "third_party" / "TrackEval",
        root / "external" / "TrackEval",
    ]
    for cand in candidates:
        if (cand / "trackeval").is_dir():
            return str(cand)
    return None


def run_trackeval(
    trackeval_root: str,
    results_dir: str,
    gt_dir: str,
    seqmap_path: str,
    output_dir: str,
    *,
    python: str = "python",
    split_to_eval: str = "val",
) -> dict:
    """Run official TrackEval via its command-line entry.

    Returns a summary dict; called only after real N1 results exist.
    """
    root = Path(trackeval_root)
    script = root / "scripts" / "run_mot_challenge.py"
    if not script.exists():
        return {"status": "ERROR", "reason": "run_mot_challenge.py not found"}
    cmd = [
        python,
        str(script),
        "--GT_FOLDER",
        gt_dir,
        "--TRACKERS_FOLDER",
        results_dir,
        "--OUTPUT_SUB_FOLDER",
        output_dir,
        "--SEQMAP_FILE",
        seqmap_path,
        "--SPLIT_TO_EVAL",
        split_to_eval,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "status": "OK" if proc.returncode == 0 else "ERROR",
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-2000:],
    }
