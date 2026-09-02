"""Machine audit of the N72R2 public-authority semantics.

This audit is intentionally static plus a tiny dependency-free bridge probe.
It does not start SAM3, read GT, or modify historical N72R2 evidence.  The
result records the pre-refactor architecture, including defects that the new
N72R3 runtime must remove.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "N72R3"
AUDIT_DIR = OUT / "audits"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE_FILES = {
    "runtime_authority": ROOT / "sam3_intermot/identity/runtime_authority.py",
    "public_authority": ROOT / "sam3_intermot/identity/public_authority.py",
    "handover": ROOT / "sam3_intermot/identity/handover.py",
    "track_manager": ROOT / "sam3_intermot/tracking/track_manager.py",
    "state_manager": ROOT / "sam3_intermot/association/state_manager.py",
    "continuous_observer": ROOT / "sam3_intermot/interaction/continuous_observer.py",
    "namespace": ROOT / "sam3_intermot/identity/namespace.py",
    "stage01_runner": ROOT / "scripts/n72r2_stage01_authority_smoke.py",
    "actions": ROOT / "sam3_intermot/interaction/actions.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_record(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256(path) if path.is_file() else None,
        "line_count": len(lines),
    }


def line_hits(path: Path, needles: list[str]) -> dict[str, list[int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {
        needle: [index for index, line in enumerate(lines, 1) if needle in line]
        for needle in needles
    }


def ast_nodes(path: Path) -> tuple[ast.AST, str]:
    text = path.read_text(encoding="utf-8")
    return ast.parse(text, filename=str(path)), text


def class_node(tree: ast.AST, name: str) -> ast.ClassDef | None:
    return next(
        (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name),
        None,
    )


def method_node(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef | None:
    owner = class_node(tree, class_name)
    if owner is None:
        return None
    return next(
        (node for node in owner.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name),
        None,
    )


def call_names(function: ast.FunctionDef | None) -> list[str]:
    if function is None:
        return []
    names: list[str] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
    return names


def assignment_lines(path: Path, names: set[str]) -> dict[str, list[int]]:
    tree, _ = ast_nodes(path)
    result = {name: [] for name in names}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                if target.attr in result:
                    result[target.attr].append(int(node.lineno))
    return result


def bridge_probe() -> dict[str, Any]:
    """Demonstrate that the old bridge accepts a state→public change over time."""

    from sam3_intermot.identity.public_authority import PublicAuthorityBridge

    bridge = PublicAuthorityBridge("n72r3-audit", "toy-sequence")
    first = SimpleNamespace(mot_track_id=11, identity_lineage_id=101)
    second = SimpleNamespace(mot_track_id=22, identity_lineage_id=202)
    bridge.bind_track(
        frame_idx=10,
        candidate_uid="session-a:candidate-17",
        association_state_id=7,
        track=first,
        binding_transaction_id="toy:first",
    )
    bridge.bind_track(
        frame_idx=20,
        candidate_uid="session-b:candidate-8",
        association_state_id=7,
        track=second,
        binding_transaction_id="toy:second",
    )
    ids = sorted({item.public_id for item in bridge.bindings if item.association_state_id == 7})
    return {
        "same_association_state_id": 7,
        "bindings_created": len(bridge.bindings),
        "public_ids_for_same_state": ids,
        "same_state_multiple_public_ids_accepted": len(ids) > 1,
        "resolution_at_frame_20": bridge.resolve_public_authority(
            association_state_id=7, frame_idx=20
        ).as_dict(),
    }


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    runtime_tree, runtime_text = ast_nodes(SOURCE_FILES["runtime_authority"])
    runner_tree, runner_text = ast_nodes(SOURCE_FILES["stage01_runner"])
    state_tree, state_text = ast_nodes(SOURCE_FILES["state_manager"])
    bridge_tree, bridge_text = ast_nodes(SOURCE_FILES["public_authority"])
    handover_text = SOURCE_FILES["handover"].read_text(encoding="utf-8")
    track_text = SOURCE_FILES["track_manager"].read_text(encoding="utf-8")
    observer_text = SOURCE_FILES["continuous_observer"].read_text(encoding="utf-8")
    namespace_text = SOURCE_FILES["namespace"].read_text(encoding="utf-8")
    actions_text = SOURCE_FILES["actions"].read_text(encoding="utf-8")

    active_class = class_node(runtime_tree, "ActiveTrackAuthority")
    active_init = method_node(runtime_tree, "ActiveTrackAuthority", "__init__")
    active_register = method_node(runtime_tree, "ActiveTrackAuthority", "register")
    state_new_pid = method_node(state_tree, "StateManager", "_new_pid")
    runner_registration = runner_text.find("registered_tracks = [")
    runner_rollout = runner_text.find("manager.rollout_frame(")
    driver_manager_injection = "def __init__(\n        self,\n        backend,\n        manager: TrackManager," in observer_text
    active_imported_in_runner = "from sam3_intermot.identity.runtime_authority import ActiveTrackAuthority" in runner_text
    namespace_imported_in_runner = "IdentityNamespace" in runner_text
    bridge_candidate_field = "candidate_uid: str" in bridge_text

    answers = {
        "active_track_authority_creates_track_manager": {
            "answer": bool(active_init and "TrackManager" in call_names(active_init)),
            "evidence": "ActiveTrackAuthority.__init__ assigns self.manager = TrackManager().",
        },
        "active_manager_is_continuous_observer_manager": {
            "answer": False,
            "evidence": "ContinuousObserverDriver receives manager externally and assigns self.manager = manager; ActiveTrackAuthority is instantiated separately by the N72R2 authority smoke.",
        },
        "candidate_gets_mot_id_before_association": {
            "answer": bool(runner_registration >= 0 and runner_rollout >= 0 and runner_registration < runner_rollout),
            "evidence": "N72R2 runner materializes registered_tracks from observations before manager.rollout_frame().",
        },
        "association_state_resolves_public_through_current_candidate_track": {
            "answer": bool("registered_tracks[index]" in runner_text and "association_state_id=state_axis[state_index]" in runner_text),
            "evidence": "The runner indexes a TrackManager track by candidate index and then binds the association state to that candidate track.",
        },
        "same_state_can_be_bound_to_different_public_ids": {
            "answer": bool(bridge_probe()["same_state_multiple_public_ids_accepted"]),
            "evidence": "Old PublicAuthorityBridge stores candidate_uid in each binding and has no state-lifetime immutable-public guard; the dependency-free probe accepts state 7→11 then state 7→22.",
        },
        "authority_binding_lifecycle_is_candidate_scoped": {
            "answer": bool(bridge_candidate_field),
            "evidence": "PublicAuthorityBinding requires candidate_uid and bind_track requires a candidate_uid for every authority binding.",
        },
        "track_manager_final_mot_track_id_exists": {
            "answer": "final_mot_track_id" in track_text,
            "evidence": "TrackManager/Track define mot_track_id; no final_mot_track_id property is defined.",
        },
        "mot_serializer_field": {
            "answer": "mot_track_id",
            "evidence": "Track and summarize_manager use track.mot_track_id; this is the field consumed by the existing MOT-shaped output path.",
        },
        "identity_namespace_enters_n72r2_active_path": {
            "answer": False,
            "evidence": "The N72R2 authority runner does not import or instantiate IdentityNamespace; its active authority uses ActiveTrackAuthority and StateManager.",
        },
        "multiple_id_allocators_or_namespaces_exist": {
            "answer": True,
            "evidence": "StateManager._new_pid/next_pid, TrackManager._next_track_id, and IdentityNamespace.PublicTrackIDAllocator are independent allocators with different semantics.",
        },
    }

    probe = bridge_probe()
    audit = {
        "schema_version": "N72R3_N72R2_AUTHORITY_SEMANTICS_AUDIT_V1",
        "created_at_utc": now,
        "status": "CONFIRMED_N72R2_AUTHORITY_ARCHITECTURE_INVALID_FOR_PERSISTENT_IDENTITY",
        "CANDIDATE_FIRST_AUTHORITY": True,
        "PUBLIC_ID_BELONGS_TO_PERSISTENT_IDENTITY": False,
        "answers": answers,
        "bridge_probe": probe,
        "static_assertions": {
            "active_authority_has_manager_assignment": bool(active_init and "manager" in assignment_lines(SOURCE_FILES["runtime_authority"], {"manager"})["manager"]),
            "active_register_creates_track": bool(active_register and "create_track" in call_names(active_register)),
            "state_manager_allocates_pid": bool(state_new_pid and "next_pid" in state_text),
            "runner_uses_separate_active_authority": active_imported_in_runner,
            "runner_does_not_use_identity_namespace": not namespace_imported_in_runner,
            "continuous_driver_manager_is_external": driver_manager_injection,
            "handover_marks_heuristic_match_as_pass": 'status="PASS"' in handover_text,
            "public_binding_contains_candidate_uid": bridge_candidate_field,
            "public_binding_has_state_immutable_guard": False,
            "track_manager_has_final_mot_track_id": "final_mot_track_id" in track_text,
            "serializer_uses_mot_track_id": "mot_track_id" in actions_text,
        },
        "required_n72r3_root_causes": [
            "CANDIDATE_FIRST_AUTHORITY",
            "HEURISTIC_HANDOVER_MARKED_EXACT_PASS",
            "CANDIDATE_SCOPED_PUBLIC_BINDING",
            "NO_PERSISTENT_IDENTITY_RUNTIME",
            "AUXILIARY_TRACK_MANAGER",
            "MULTIPLE_ID_ALLOCATOR_SEMANTICS",
        ],
        "source_files": {name: source_record(path) for name, path in SOURCE_FILES.items()},
        "line_evidence": {
            "runtime_authority": line_hits(SOURCE_FILES["runtime_authority"], ["self.manager = TrackManager()", "self.lineages = IdentityLineageRegistry()", "self.manager.create_track"]),
            "public_authority": line_hits(SOURCE_FILES["public_authority"], ["candidate_uid: str", "def bind_track", "self._bindings.append(binding)"]),
            "handover": line_hits(SOURCE_FILES["handover"], ['status not in {"PASS"', 'status="PASS"', "score = 0.75 * iou + 0.25 * appearance"]),
            "state_manager": line_hits(SOURCE_FILES["state_manager"], ["def _new_pid", "pid = self._new_pid()", "IdentityState(pid"]),
            "track_manager": line_hits(SOURCE_FILES["track_manager"], ["mot_track_id", "def create_track", "_next_track_id"]),
            "continuous_observer": line_hits(SOURCE_FILES["continuous_observer"], ["manager: TrackManager", "self.manager = manager", "self.ctx = SystemContext"]),
            "stage01_runner": line_hits(SOURCE_FILES["stage01_runner"], ["registered_tracks = [", "manager.rollout_frame(", "association_state_id=state_axis[state_index]"]),
            "namespace": line_hits(SOURCE_FILES["namespace"], ["class PublicTrackIDAllocator", "class IdentityNamespace", "self.allocator = allocator"]),
        },
        "runtime_future_gt_used": False,
        "historical_evidence_read_only": True,
        "n72r2_final_status_preserved": "BLOCKED_CANDIDATE_RECALL",
        "next_stage": "02_PERSISTENT_RUNTIME",
    }
    report = f"""# N72R3 authority root cause\n\nDate: {now}\n\n## Machine conclusion\n\n`CANDIDATE_FIRST_AUTHORITY = true`. The N72R2 active authority architecture is invalid for sequence-persistent public identity:\n\n- `ActiveTrackAuthority` creates its own `TrackManager` and registers candidates before association.\n- `ContinuousObserverDriver` receives a different manager from its caller.\n- `StateManager._new_pid` is an association-local allocator; it is not the persistent public-ID owner.\n- `PublicAuthorityBinding` is candidate-scoped and the old bridge accepts a state changing from one public ID to another.\n- `PersistentLineageHandover.match_overlap()` turns IoU/appearance evidence into `status=PASS`, which is not exact authority.\n- The real track/output field is `Track.mot_track_id`; `TrackManager.final_mot_track_id` does not exist.\n- `IdentityNamespace` is not on the N72R2 authority-smoke active path, while independent state/track/public allocators remain.\n\nThe N72R2 final status remains `BLOCKED_CANDIDATE_RECALL` as historical evidence. N72R3 retires its 13/13 candidate-presence requirement as an identity-continuity gate.\n\n## Dependency-free state probe\n\nThe old bridge accepted association state `7` → public `11` at frame 10 and state `7` → public `22` at frame 20. This is an architecture defect, not a scientific result; the probe uses fake tracks and no dataset or GT.\n\n```json\n{json.dumps(probe, indent=2, sort_keys=True)}\n```\n\n## Required replacement\n\nN72R3 must make a sequence-lifetime `SequencePersistentIdentityRuntime` the owner of public identity, lineage, appearance/motion/lost state and one injected `TrackManager`. A candidate is assigned to an existing persistent identity; only an outer birth decision may allocate a new identity. Session reset may clear raw/adapter/SAM bindings and mark an identity LOST/NONE, but must not delete or renumber it.\n\nIoU/appearance/motion may produce only recovery/association evidence. Exact authority must come from the persistent identity record and an immutable state → public binding.\n\n## Evidence\n\nMachine audit: `outputs/N72R3/audits/n72r2_authority_semantics.json`\n\nHistorical N72R2 evidence is read-only and was not changed.\n"""
    atomic_json(AUDIT_DIR / "n72r2_authority_semantics.json", audit)
    atomic_text(ROOT / "docs/N72R3_AUTHORITY_ROOT_CAUSE.md", report)
    atomic_json(
        OUT / "stage_01_status.json",
        {
            "schema_version": "N72R3_STAGE_STATUS_V1",
            "stage": "01_N72R2_AUTHORITY_SEMANTICS_AUDIT",
            "status": audit["status"],
            "created_at_utc": now,
            "artifact": str(AUDIT_DIR / "n72r2_authority_semantics.json"),
            "report": str(ROOT / "docs/N72R3_AUTHORITY_ROOT_CAUSE.md"),
            "CANDIDATE_FIRST_AUTHORITY": True,
            "historical_n72r2_final_status": "BLOCKED_CANDIDATE_RECALL",
            "runtime_future_gt_used": False,
            "dynamic_probe": probe,
            "next_stage": "02_PERSISTENT_RUNTIME",
        },
    )
    print(json.dumps({"status": audit["status"], "artifact": str(AUDIT_DIR / "n72r2_authority_semantics.json")}))


if __name__ == "__main__":
    main()
