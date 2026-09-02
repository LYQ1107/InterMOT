"""Root-confined path validation for N72R1 external inputs."""

from __future__ import annotations

from pathlib import Path


class PathSafetyError(ValueError):
    pass


def resolve_within_root(reference: str | Path, root: str | Path, *, require_file: bool = False) -> Path:
    """Resolve a reference and reject traversal, symlink escape, and root mixups."""

    root_path = Path(root).resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise PathSafetyError(f"allowed root is not a directory: {root_path}")
    reference_path = Path(reference)
    candidate = reference_path if reference_path.is_absolute() else root_path / reference_path
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise PathSafetyError(f"path escapes allowed root: {reference}") from exc
    if require_file and (not resolved.exists() or not resolved.is_file()):
        raise PathSafetyError(f"path is not an existing file: {reference}")
    return resolved


def validate_namespaced_reference(
    reference: str | Path,
    *,
    root: str | Path,
    namespace: str,
    expected_namespace: str,
    require_file: bool = False,
) -> Path:
    if namespace != expected_namespace:
        raise PathSafetyError(f"{namespace} reference cannot be used under {expected_namespace} root")
    return resolve_within_root(reference, root, require_file=require_file)


def check_distinct_roots(candidate_root: str | Path, raw_root: str | Path) -> dict[str, str]:
    candidate = Path(candidate_root).resolve()
    raw = Path(raw_root).resolve()
    if candidate == raw:
        raise PathSafetyError("candidate-root and raw-root must be distinct namespaces")
    return {"candidate_root": str(candidate), "raw_root": str(raw)}


__all__ = ["PathSafetyError", "check_distinct_roots", "resolve_within_root", "validate_namespaced_reference"]
