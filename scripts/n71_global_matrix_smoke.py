#!/usr/bin/env python3
"""CUDA smoke for the isolated N71 global-matrix scorer and solver."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n71_global_matrix_common as common  # noqa: E402


def atomic_torch_save(path: Path, payload: object) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        with open(tmp, "rb") as handle:
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


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def run_smoke(manifest: Path, output: Path, device_name: str, checkpoint_path: Path) -> dict:
    import torch

    common.set_seed(7171)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"N71 global matrix smoke requires CUDA, got {device}")
    arrays, dataset_manifest = common.load_arrays(manifest)
    meta_path = Path(dataset_manifest["output_root"]) / "group_metadata.jsonl"
    metadata = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines()]
    chosen: list[int] = []
    seen_actions: set[str] = set()
    seen_sequences: set[str] = set()
    for row in metadata:
        action, sequence = str(row["action_type"]), str(row["sequence"])
        if action in seen_actions or sequence in seen_sequences:
            continue
        chosen.append(int(row["group"]))
        seen_actions.add(action)
        seen_sequences.add(sequence)
        if len(chosen) >= 3:
            break
    if len(chosen) < 3:
        raise RuntimeError(f"smoke selected only {len(chosen)} independent action/sequence groups")
    mean, std = common.context_normalization(arrays)
    model = common.build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=1.0e-4)
    records = []
    first_input = None
    first_output = None
    for group in chosen:
        start, end = int(arrays["group_offsets"][group]), int(arrays["group_offsets"][group + 1])
        index = np.arange(start, end, dtype=np.int64)
        tensors = common.tensors_for_indices(arrays, index, mean, std, device)
        pair, none = model(*tensors[:5])
        if not torch.isfinite(pair).all() or not torch.isfinite(none).all():
            raise RuntimeError(f"nonfinite smoke logits group={group}")
        loss, detail = common.group_loss(model, arrays, [group], index, mean, std, device)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite smoke loss group={group}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).detach().cpu())
        if not np.isfinite(grad_norm):
            raise RuntimeError(f"nonfinite smoke gradient group={group}")
        optimizer.step()
        if first_input is None:
            first_input = tensors[:5]
        row = metadata[group]
        if int(row["frame"]) <= int(row["event_frame"]):
            raise RuntimeError(f"causal boundary failed group={group}: {row['frame']} <= {row['event_frame']}")
        records.append({
            "group": group,
            "sequence": row["sequence"],
            "action_type": row["action_type"],
            "event_id": row["event_id"],
            "event_frame": int(row["event_frame"]),
            "frame": int(row["frame"]),
            "candidate_count": int(row["candidate_count"]),
            "identity_count": int(row["identity_count"]),
            "input_shapes": [list(value.shape) for value in tensors[:5]],
            "loss": float(loss.detach().cpu()),
            "loss_detail": detail,
            "gradient_norm": grad_norm,
            "runtime_future_gt_used": False,
            "candidate_order_preserved": True,
            "public_id_numeric_feature": False,
        })
    if first_input is None:
        raise RuntimeError("smoke did not retain checkpoint input")
    with torch.no_grad():
        first_output = tuple(value.detach().clone() for value in model(*first_input))
    checkpoint = checkpoint_path
    atomic_torch_save(checkpoint, {"schema": "N71_GLOBAL_MATRIX_SMOKE_CHECKPOINT_V1", "state_dict": model.state_dict(), "mean": mean, "std": std, "manifest": str(manifest), "manifest_sha256": common.sha256(manifest), "runtime_future_gt_used": False})
    reloaded = common.build_model().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    reloaded.load_state_dict(payload["state_dict"])
    with torch.no_grad():
        reload_output = tuple(value.detach() for value in reloaded(*first_input))
    reload_error = max(float(torch.max(torch.abs(a - b)).cpu()) for a, b in zip(first_output, reload_output))
    if reload_error > 1.0e-6:
        raise RuntimeError(f"checkpoint reload mismatch {reload_error}")

    # Verify the learned output can be passed to the same explicit-NONE
    # Hungarian interface used by the later replay, without using GT.
    group = chosen[0]
    start, end = int(arrays["group_offsets"][group]), int(arrays["group_offsets"][group + 1])
    index = np.arange(start, end, dtype=np.int64)
    tensors = common.tensors_for_indices(arrays, index, mean, std, device)
    with torch.no_grad():
        pair, none = reloaded(*tensors[:5])
    n, p = int(metadata[group]["candidate_count"]), int(metadata[group]["identity_count"])
    public_ids = [int(value) for value in metadata[group]["public_id_order_offline_only"]]
    assignment = common.explicit_none_hungarian(pair.detach().cpu().numpy().reshape(n, p), none.detach().cpu().numpy().reshape(n, p)[:, 0], public_ids, [{"native_tid": -1} for _ in range(n)])
    if len(assignment["assigned_public_ids"]) != n:
        raise RuntimeError("explicit NONE smoke assignment length mismatch")
    regression_modules = []
    for module in ("sam3_intermot", "sam3_intermot.association.appearance_memory", "sam3_intermot.association.online_associator", "sam3_intermot.association.ccam_replay"):
        __import__(module)
        regression_modules.append(module)
    result = {
        "schema": "N71_GLOBAL_MATRIX_SMOKE_V1",
        "status": "PASS",
        "manifest": str(manifest),
        "manifest_sha256": common.sha256(manifest),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "model": common.model_metadata(model),
        "selected_groups": records,
        "distinct_action_count": len(seen_actions),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": common.sha256(checkpoint),
        "reload_max_abs_error": reload_error,
        "explicit_none_assignment_smoke": {"status": "PASS", "candidate_count": n, "identity_count": p, "assigned_public_id_count": sum(value is not None for value in assignment["assigned_public_ids"])},
        "association_import_regression": {"status": "PASS", "modules": regression_modules},
        "causal_boundary": {"event_frame_memory_read": False, "first_frame_used": "event_frame+1", "runtime_future_gt_used": False},
        "numeric_public_id_feature": False,
        "production_authorized": False,
    }
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, default=Path("/path/to/cache/SAM3_InterMOT_N71/training/N71_GLOBAL_MATRIX_SMOKE_ATTEMPT4.pt"))
    args = parser.parse_args()
    try:
        result = run_smoke(args.manifest.resolve(), args.output.resolve(), args.device, args.checkpoint.resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        attempts = ROOT / "outputs/N71/attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        existing = sorted(attempts.glob("n71_global_matrix_smoke_failure_attempt*.json"))
        path = attempts / f"n71_global_matrix_smoke_failure_attempt{len(existing) + 1}.json"
        failure = {"schema": "N71_GLOBAL_MATRIX_SMOKE_FAILURE_V1", "status": "FAIL_PRESERVED", "failure_type": type(exc).__name__, "failure_message": str(exc), "traceback": traceback.format_exc(), "manifest": str(args.manifest.resolve()), "manifest_sha256": common.sha256(args.manifest.resolve()), "device": args.device, "runtime_future_gt_used": False, "production_authorized": False}
        atomic_json(path, failure)
        print(json.dumps({"status": "FAIL_PRESERVED", "failure_artifact": str(path), "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
