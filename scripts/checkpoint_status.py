#!/usr/bin/env python
"""Write N0 checkpoint access status."""

import json
import os
from pathlib import Path

from sam3_intermot.utils.io import atomic_write_json


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "outputs" / "n0"
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    status = {
        "N0_STATUS": "BLOCKED_EXTERNAL_CHECKPOINT",
        "repo_id": "facebook/sam3.1",
        "checkpoint_file": "sam3.1_multiplex.pt",
        "gated": "manual",
        "http_status": 403,
        "error_code": "GatedRepo",
        "error_message": (
            "Access to model facebook/sam3.1 is restricted and the configured "
            "account is not in the authorized list."
        ),
        "token_present": token_path.exists(),
        "local_checkpoint_found": False,
        "local_search_paths": [
            str(Path.home() / ".cache" / "huggingface" / "hub"),
            "/path/to/workspace/SERVER_ONLY/avis",
        ],
        "action_required": "Request HF access to facebook/sam3.1 or provide a legal "
                           "checkpoint path in configs/default.yaml backend.checkpoint_path.",
    }
    atomic_write_json(out_dir / "checkpoint_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
