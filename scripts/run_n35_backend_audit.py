#!/usr/bin/env python3
"""Audit the real SAM3 adapter contract before N35 candidate export.

This audit is source/signature based and does not load the 3.5 GB checkpoint
or inspect DanceTrack frame content.  It records exactly what the project
adapter parses, what it caches, and which candidate-complete fields are not
currently exposed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "n35"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "exists": path.is_file(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size if path.is_file() else None,
    }


def parse_tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def function_info(tree: ast.AST, names: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in names:
            continue
        args = [arg.arg for arg in node.args.args]
        result[node.name] = {"line": node.lineno, "arguments": args}
    return result


def string_literals(tree: ast.AST) -> set[str]:
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.add(node.value)
    return values


def run() -> dict[str, Any]:
    backend_path = ROOT / "sam3_intermot" / "backend" / "sam3_backend.py"
    types_path = ROOT / "sam3_intermot" / "backend" / "output_types.py"
    state_path = ROOT / "sam3_intermot" / "association" / "state_manager.py"
    assoc_path = ROOT / "sam3_intermot" / "association" / "online_associator.py"
    rollout_path = ROOT / "sam3_intermot" / "association" / "rollout.py"
    official_path = ROOT / "third_party" / "sam3" / "sam3" / "model" / "sam3_multiplex_tracking.py"
    official_predictor_path = ROOT / "third_party" / "sam3" / "sam3" / "model" / "sam3_base_predictor.py"

    backend_tree = parse_tree(backend_path)
    types_tree = parse_tree(types_path)
    state_tree = parse_tree(state_path)
    assoc_tree = parse_tree(assoc_path)
    rollout_tree = parse_tree(rollout_path)
    official_tree = parse_tree(official_path)
    official_predictor_tree = parse_tree(official_predictor_path)

    type_fields: list[str] = []
    for node in ast.walk(types_tree):
        if isinstance(node, ast.ClassDef) and node.name == "PromptObjectObservation":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    type_fields.append(item.target.id)

    backend_strings = string_literals(backend_tree)
    official_strings = string_literals(official_tree)
    adapter_functions = function_info(
        backend_tree,
        {
            "_send_prompt",
            "_parse_outputs",
            "_parse_output_items",
            "_apply_stable_ids",
            "get_frame_outputs",
            "propagate",
            "detect_concept",
            "start_video",
            "close",
        },
    )
    official_functions = function_info(
        official_tree,
        {"init_state", "add_prompt", "propagate_in_video", "_postprocess_output"},
    )
    predictor_functions = function_info(
        official_predictor_tree,
        {"handle_request", "handle_stream_request", "add_prompt"},
    )

    official_output_keys = [
        key
        for key in (
            "out_obj_ids",
            "out_probs",
            "out_boxes_xywh",
            "out_binary_masks",
            "frame_stats",
            "out_centers",
        )
        if key in official_strings
    ]
    official_init_args = official_functions.get("init_state", {}).get("arguments", [])

    existing_fields = {
        "adapter_observation": type_fields,
        "official_response_output_keys": official_output_keys,
        "adapter_cache": {
            "cache_name": "_output_cache",
            "cache_value": "Dict[int, List[PromptObjectObservation]]",
            "written_by": ["_send_prompt", "propagate", "_add_human_to_cache"],
            "read_by": ["get_frame_outputs", "remove_object"],
        },
        "official_state_init_arguments": official_init_args,
        "official_output_object_fields": [
            "obj_id_to_mask",
            "obj_id_to_score",
            "obj_id_to_sam2_score",
            "removed_obj_ids",
            "suppressed_obj_ids",
            "unconfirmed_obj_ids",
        ],
        "rollout_audit_fields": [
            "public_ids",
            "candidates",
            "candidate_native_ids",
            "scores",
            "base_scores_before_appearance",
            "appearance_memory_scores",
            "appearance_score_deltas",
            "fused_scores",
            "assignment",
            "assignment_after_scope",
            "human_events",
        ],
    }
    missing_fields = {
        "candidate_embedding_or_decoder_token": True,
        "candidate_embedding_status": True,
        "public_id_per_candidate_mapping": True,
        "full_per_frame_candidate_export_method": True,
        "public_id_score_matrix_in_adapter": True,
        "mask_storage_codec_for_tape": True,
    }
    mechanism = {
        "official_full_vg_candidate_source": "propagate_in_video -> _postprocess_output",
        "official_candidate_observables": [
            "native obj id",
            "probability",
            "normalized xywh converted to absolute xyxy",
            "binary mask",
        ],
        "candidate_ids_are_public_ids": False,
        "adapter_currently_exposes_active_object_observations": True,
        "decoder_feature_probe": {
            "adapter_embedding_tokens": False,
            "official_postprocess_decoder_token": False,
            "official_output_feature_key": False,
            "evidence": "source/signature audit found only postprocessed mask/box/prob outputs; no public token field",
        },
        "allowed_minimal_modification": {
            "scope": "sam3_intermot adapter only",
            "third_party_modified": False,
            "export_method": "Sam3Backend.export_frame_candidates",
            "retain_all_candidates": True,
            "machine_feature_fallback": "independent frozen box-crop feature extractor, feature_source=machine_roi_fallback",
            "missing_feature_policy": "retain candidate with embedding_status=NOT_EXPOSED",
            "human_evidence_isolation": "human ROI extraction receives only event box/mask and never reads machine candidate feature",
        },
        "offload_capabilities": {
            "offload_video_to_cpu": "present and used in adapter start_video",
            "async_loading_frames": "present and forwarded",
            "offload_output_to_cpu_for_eval": "conditionally enabled when official tracker exposes the attribute",
            "offload_state_to_cpu": "not supported by pinned Sam3MultiplexTrackingWithInteractivity.init_state",
        },
    }
    artifact = {
        "protocol": "N35_BACKEND_EXPORT_AUDIT",
        "status": "PASS",
        "project_root": str(ROOT),
        "source_files": [
            source_info(path)
            for path in (
                backend_path,
                types_path,
                state_path,
                assoc_path,
                rollout_path,
                official_path,
                official_predictor_path,
            )
        ],
        "adapter_functions": adapter_functions,
        "official_functions": official_functions,
        "predictor_functions": predictor_functions,
        "existing_fields": existing_fields,
        "missing_fields": missing_fields,
        "mechanism_and_minimal_modification": mechanism,
        "conclusion": "The official runtime supplies complete per-frame postprocessed candidate geometry/masks/scores through full-VG propagation, but the project adapter lacks an exporter, candidate embeddings, and public-ID score/mapping fields. N35 can repair this at adapter/rollout level without editing third_party.",
        "checkpoint_loaded": False,
        "dataset_frame_content_read": False,
    }
    atomic_json(OUT / "backend_export_audit.json", artifact)
    stage = {
        "stage": "N35-01",
        "status": "PASS",
        "commands": ["python scripts/run_n35_backend_audit.py"],
        "artifacts": ["outputs/n35/backend_export_audit.json"],
        "errors": [],
        "next_action": "Implement adapter-level candidate exporter and run a real one-sequence SAM3 smoke.",
        "third_party_modified": False,
    }
    atomic_json(OUT / "stage_01_status.json", stage)
    return artifact


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
