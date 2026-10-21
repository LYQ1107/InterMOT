"""Verify the user-authorized NAS backend before resuming R2 workers.

This is an engineering/storage receipt only. It does not alter the frozen
scientific protocol, models, candidate matcher, association logic, or gates.
"""
import json
import shutil
from pathlib import Path

import torch

from scripts.n72r21r2_common import (
    ASSETS,
    OUT,
    ROOT,
    STORAGE_MOUNT,
    STORAGE_ROOT,
    sha256,
    storage,
    update_status,
    utcnow,
    write_json,
)


EXPECTED_MOUNT = Path("/data3/liuyeqiang/nas/A100")
NAS_ASSET_ROOT = EXPECTED_MOUNT / "exp/InterMOT_N72R21R2_assets"
LINKS = {
    ASSETS / "main_policy_v1": NAS_ASSET_ROOT / "main_policy_v1",
    ASSETS / "main_policy_v1_logs": NAS_ASSET_ROOT / "main_policy_v1_logs",
    ASSETS / "training/event_authority_v1": NAS_ASSET_ROOT / "training/event_authority_v1",
}


def main():
    assert STORAGE_MOUNT == EXPECTED_MOUNT, (STORAGE_MOUNT, EXPECTED_MOUNT)
    assert STORAGE_MOUNT.is_mount(), STORAGE_MOUNT
    assert STORAGE_ROOT == NAS_ASSET_ROOT, (STORAGE_ROOT, NAS_ASSET_ROOT)
    assert STORAGE_ROOT.is_relative_to(STORAGE_MOUNT)
    for local, target in LINKS.items():
        assert local.is_symlink(), f"compatibility path is not a symlink: {local}"
        assert local.resolve() == target.resolve(), (local, target)
        assert target.exists(), target

    records = sorted((OUT / "training/event_authority").glob("MAIN__*.json"))
    loaded = 0
    for record_path in records:
        record = json.loads(record_path.read_text())
        checkpoint = Path(record["checkpoint_path"])
        assert checkpoint.exists(), checkpoint
        assert sha256(checkpoint) == record["checkpoint_sha256"], checkpoint
        torch.load(checkpoint, map_location="cpu", weights_only=True)
        loaded += 1

    receipt = {
        "schema": "N72R21R2_NAS_STORAGE_READBACK_V1",
        "stage": "N72R21R2",
        "goal": "Event-Level Causal Identity Association: Safe One-Click Intervention with Fresh-Sequence Generalization for Online MOT",
        "utc": utcnow(),
        "storage_root": str(STORAGE_ROOT),
        "storage_mount": str(STORAGE_MOUNT),
        "nas_mount_verified": True,
        "storage_guard": storage(6 << 30),
        "local_asset_root": str(ASSETS),
        "compatibility_symlinks": {str(local): str(target) for local, target in LINKS.items()},
        "main_fit_records": len(records),
        "checkpoint_sha_verified_and_loader_smoked": loaded,
        "local_worktree_outputs_unchanged": True,
        "scientific_protocol_changed": False,
        "candidate_matcher_changed": False,
        "association_logic_changed": False,
        "val_test_sot_opened": False,
    }
    path = write_json("audit/NAS_STORAGE_READBACK_20261021.json", receipt)
    digest = sha256(path)
    update_status(
        storage_backend="AUTHORIZED_NAS_MOUNT",
        storage_root=str(STORAGE_ROOT),
        storage_mount=str(STORAGE_MOUNT),
        storage_guard_status="PASS_NAS_FREE_SPACE_ABOVE_60_GIB_RESERVE",
        nas_storage_receipt=str(path),
        nas_storage_receipt_sha256=digest,
        checkpoint_storage="NAS_WITH_LOCAL_COMPATIBILITY_SYMLINK",
        scientific_protocol_changed=False,
        candidate_matcher_changed=False,
        association_logic_changed=False,
    )
    print(json.dumps({"receipt": str(path), "receipt_sha256": digest, **receipt}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
