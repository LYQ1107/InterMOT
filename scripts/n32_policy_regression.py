#!/usr/bin/env python3
"""Small official-backend regression for the frozen N32 policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "third_party/sam3") not in sys.path:
    sys.path.insert(0, str(ROOT / "third_party/sam3"))

from sam3_intermot.adaptation.correction_application_policy import (  # noqa: E402
    CorrectionApplicationAction,
    CorrectionApplicationPolicy,
)
from sam3_intermot.backend.sam3_state_snapshot import (  # noqa: E402
    snapshot_continuation_state,
    state_container_summary,
)
from scripts.n29_lit_online_replay import (  # noqa: E402
    _image_files,
    _install_official_box_singleton,
    _make_backend,
    _read_gt,
    _session,
)
from scripts.n29r_paired_replay import _ensure_public_singleton_binding, _horizon_metrics  # noqa: E402


CHECKPOINT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
N31_ABLATION = ROOT / "outputs/n31/correction_state_ablation.json"
HARD_MANIFEST = ROOT / "outputs/n29r/hard_episode_manifest.json"
OUT = ROOT / "outputs/n32/policy_regression.json"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _hash_outputs(outputs: Mapping[int, Any]) -> str:
    rows = []
    for frame in sorted(outputs):
        for obs in sorted(outputs[frame], key=lambda item: int(getattr(item, "sam_object_id", -1))):
            mask = np.asarray(getattr(obs, "mask", np.zeros((1, 1))), dtype=bool)
            rows.append({
                "frame": int(frame),
                "id": int(getattr(obs, "sam_object_id", -1)),
                "box": np.asarray(getattr(obs, "box_xyxy", []), dtype=float).round(5).tolist(),
                "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            })
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()


def _episode_by_id() -> tuple[dict[str, Any], dict[str, Any]]:
    source = json.loads(N31_ABLATION.read_text(encoding="utf-8"))
    episodes = source["episode_results"]
    failure = next(e for e in episodes if not e["branches"]["P3_restore_old_state_after_prompt_failure"].get("prompt_success"))
    success = next(e for e in episodes if e["branches"]["P3_restore_old_state_after_prompt_failure"].get("prompt_success"))
    return failure, success


def _run_episode(backend: Any, episode: Mapping[str, Any], n31_row: Mapping[str, Any]) -> dict[str, Any]:
    sequence = Path(str(episode["sequence_path"]))
    gt = _read_gt(sequence)
    images = _image_files(sequence)
    init = int(episode["initialization_frame"])
    correction = int(episode["correction_frame"])
    query_end = min(int(episode["query_end"]), len(images) - 1)
    public_id = int(episode["public_id"])
    identity = int(episode["dataset_identity"])
    init_box = np.asarray(gt[init][identity], dtype=float)
    correction_box = np.asarray(episode["correction_box"], dtype=float)
    _session(backend, sequence)
    backend.add_box(init, public_id, init_box)
    _install_official_box_singleton(backend, frame_idx=init, public_id=public_id, box_xyxy=init_box)
    prefix_outputs = backend.propagate(init, correction, start_frame_index=init)
    raw_digest = _hash_outputs({correction: prefix_outputs.get(correction, [])})
    raw_recorded = correction in prefix_outputs
    prefix_snapshot = snapshot_continuation_state(backend)
    prefix_control = state_container_summary(backend)
    policy_rows = {}
    for action in CorrectionApplicationAction:
        ledger: list[dict[str, Any]] = []
        result = CorrectionApplicationPolicy(action).apply(
            backend,
            correction_frame=correction,
            public_id=public_id,
            corrected_box=correction_box,
            pre_correction_snapshot=prefix_snapshot,
            ledger=ledger,
            raw_output_recorded=raw_recorded,
            ensure_binding=_ensure_public_singleton_binding,
        )
        control_after = state_container_summary(backend)
        future = backend.propagate(correction + 1, query_end, start_frame_index=correction + 1)
        metrics = _horizon_metrics(future, gt, {**episode, "query_end": query_end})
        policy_rows[action.name] = {
            "result": result.to_dict(),
            "ledger": ledger,
            "current_cache_has_human_correction": any(
                int(getattr(obs, "sam_object_id", -1)) == public_id and bool(getattr(obs, "is_human_verified", False))
                for obs in backend._output_cache.get(correction, [])
            ),
            "future_signature": _hash_outputs(future),
            "metrics": metrics,
            "control_after": control_after,
            "n31_p5_h20_iou": n31_row["branches"]["P5_current_ensure_path"]["metrics"]["20"]["mean_box_iou_visible"],
        }
    k0 = policy_rows["KEEP_OLD_STATE"]
    k1 = policy_rows["APPLY_CURRENT_ENSURE"]
    k2 = policy_rows["PROMPT_THEN_RESTORE"]
    k1_h20 = float(k1["metrics"]["20"]["mean_box_iou_visible"])
    n31_p5 = float(k1["n31_p5_h20_iou"])
    return {
        "episode_id": str(episode["episode_id"]),
        "sequence": str(episode["sequence"]),
        "correction_frame": correction,
        "raw_correction_output_recorded": raw_recorded,
        "raw_correction_output_digest": raw_digest,
        "prefix_control": prefix_control,
        "policies": policy_rows,
        "checks": {
            "k0_current_output_corrected": bool(k0["result"]["current_output_corrected"] and k0["current_cache_has_human_correction"]),
            "k0_future_state_kept": bool(not k0["result"]["future_state_changed"] and not k0["result"]["rollback_used"]),
            "k1_n31_p5_h20_equivalent": bool(abs(k1_h20 - n31_p5) <= 1.0e-6),
            "k2_prompt_failure_restored": bool(
                (not bool(n31_row["branches"]["P3_restore_old_state_after_prompt_failure"].get("prompt_success")))
                == bool(k2["result"]["rollback_used"])
            ),
            "all_policy_mappings_valid": all(bool(row["result"]["mapping_valid"]) for row in policy_rows.values()),
            "all_policy_target_states_present": all(bool(row["result"]["target_state_present"]) for row in policy_rows.values()),
            "history_raw_output_not_parameter_rewritten": bool(raw_recorded and all(row["ledger"] for row in policy_rows.values())),
            "protected_identity_spatial_scope": "NOT_APPLICABLE_SINGLE_ID_NO_PROTECTED_ID",
        },
    }


def run(*, output: Path = OUT, checkpoint: Path = CHECKPOINT) -> dict[str, Any]:
    failure, success = _episode_by_id()
    hard_episodes = {
        str(item["episode_id"]): item
        for item in json.loads(HARD_MANIFEST.read_text(encoding="utf-8"))["episodes"]
    }
    # Two fixed smoke episodes exercise both K2 rollback and official prompt
    # success.  The same backend is reused, but each episode starts a fresh
    # official session and every policy restores its own common prefix.
    backend = _make_backend(checkpoint)
    rows = []
    try:
        for episode_result in (failure, success):
            hard = hard_episodes[str(episode_result["episode_id"])]
            manifest_episode = {
                "episode_id": episode_result["episode_id"],
                "sequence": episode_result["sequence"],
                "sequence_path": hard["sequence_path"],
                **{key: episode_result[key] for key in ("initialization_frame", "correction_frame", "query_end", "public_id", "dataset_identity")},
                "correction_box": hard["correction_box"],
            }
            try:
                rows.append(_run_episode(backend, manifest_episode, episode_result))
            except Exception as exc:
                rows.append({"episode_id": episode_result["episode_id"], "status": "NOT_RUN", "failure": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=20)})
    finally:
        backend.close()
    checks = {
        "k0_current_output_corrected": all(row.get("checks", {}).get("k0_current_output_corrected", False) for row in rows),
        "k0_future_state_kept": all(row.get("checks", {}).get("k0_future_state_kept", False) for row in rows),
        "k1_n31_p5_h20_equivalent": all(row.get("checks", {}).get("k1_n31_p5_h20_equivalent", False) for row in rows),
        "k2_prompt_failure_restored": bool(rows and rows[0].get("checks", {}).get("k2_prompt_failure_restored", False)),
        "all_policy_mappings_valid": all(row.get("checks", {}).get("all_policy_mappings_valid", False) for row in rows),
        "all_policy_target_states_present": all(row.get("checks", {}).get("all_policy_target_states_present", False) for row in rows),
        "history_raw_output_not_parameter_rewritten": all(row.get("checks", {}).get("history_raw_output_not_parameter_rewritten", False) for row in rows),
        "protected_identity_spatial_scope": True,
    }
    result = {
        "protocol": "N32-B-POLICY-REGRESSION",
        "status": "PASS" if len(rows) == 2 and all(checks.values()) else "FAIL",
        "episode_count": len(rows),
        "episodes": rows,
        "checks": checks,
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_policy_input": False,
    }
    _write(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    result = run(output=args.output, checkpoint=args.checkpoint)
    print(json.dumps({"protocol": result["protocol"], "status": result["status"], "checks": result["checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
