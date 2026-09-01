#!/usr/bin/env python
"""Write N0 static audit outputs (repo/API facts, no GPU execution)."""

import json
import platform
import subprocess
from pathlib import Path

from sam3_intermot.utils.io import atomic_write_json


REPO_COMMIT = "4cbac146c1b5a1e3a7f5c6a894901090b4dfd65b"


def git_commit(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sam3_root = project_root / "third_party" / "sam3"
    audit = {
        "repo_url": "https://github.com/facebookresearch/sam3",
        "pinned_commit": REPO_COMMIT,
        "actual_commit": git_commit(sam3_root),
        "hf_repo": "https://huggingface.co/facebook/sam3.1",
        "checkpoint_name": "sam3.1_multiplex.pt",
        "checkpoint_sha256": "NOT_AVAILABLE_DUE_TO_GATED_ACCESS",
        "license": "SAM License (Meta)",
        "python_required": ">=3.12 (official README)",
        "pytorch_required": ">=2.7; README example pins torch==2.10.0 cu128",
        "cuda_required": "CUDA 12.6+",
        "torchvision_required": "installed with torch cu128 wheel",
        "compiled_extensions": "optional: flash-attn-3, cc_torch; not required for first version",
        "gpus": ["GPU 8", "GPU 9"],
        "official_example": (
            "build_sam3_multiplex_video_predictor(checkpoint_path=...) then "
            "handle_request(type='start_session'|'add_prompt'|'remove_object') "
            "and handle_stream_request(type='propagate_in_video')"
        ),
        "multiplex": {
            "supported": True,
            "entry_point": "sam3.model_builder.build_sam3_multiplex_video_predictor",
            "multiplex_count": 16,
            "notes": "Object Multiplex verified in official source and release notes; "
                     "runtime verification blocked until checkpoint access.",
        },
        "host_python": platform.python_version(),
        "host_cuda_available": False,
    }
    out_dir = project_root / "outputs" / "n0"
    atomic_write_json(out_dir / "static_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
