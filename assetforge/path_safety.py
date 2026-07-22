from __future__ import annotations

import json
import shutil
from pathlib import Path


_OUTPUT_MARKER = ".assetforge-output.json"
_OUTPUT_MARKER_CONTENT = {"owner": "assetforge", "schemaVersion": 1}


def safe_output_child(
    root: str | Path,
    *parts: str,
    label: str = "output path",
) -> Path:
    """Resolve an output child and reject traversal through names or symlinks."""

    resolved_root = Path(root).expanduser().resolve()
    candidate = resolved_root.joinpath(*parts)
    try:
        relative = candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its output root: {candidate}") from exc
    current = resolved_root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{label} crosses a symbolic link: {current}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its output root: {candidate}") from exc
    return resolved


def reset_output_directory(path: str | Path, *, label: str = "generated output") -> Path:
    """Create an empty engine-owned directory without deleting unknown files.

    A non-empty directory is reset only when a valid AssetForge ownership marker
    is already present. This makes direct API and CLI output paths fail closed
    when they point at an arbitrary user directory.
    """

    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{label} is a symbolic link: {candidate}")
    resolved = candidate.resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    if not resolved.exists():
        resolved.mkdir(parents=True)
    marker = resolved / _OUTPUT_MARKER
    entries = list(resolved.iterdir())
    if entries:
        for descendant in resolved.rglob("*"):
            if descendant.is_symlink():
                raise ValueError(f"{label} contains a symbolic link: {descendant}")
    if marker.exists() or marker.is_symlink():
        if marker.is_symlink() or not marker.is_file():
            raise ValueError(f"{label} has an invalid AssetForge ownership marker: {marker}")
        try:
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{label} has an unreadable AssetForge ownership marker: {marker}"
            ) from exc
        if marker_data != _OUTPUT_MARKER_CONTENT:
            raise ValueError(f"{label} has an invalid AssetForge ownership marker: {marker}")
    elif entries:
        raise ValueError(
            f"{label} is non-empty and is not marked as AssetForge-owned: {resolved}"
        )

    if entries:
        for child in resolved.iterdir():
            if child == marker:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    marker.write_text(
        json.dumps(_OUTPUT_MARKER_CONTENT, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved


__all__ = ["reset_output_directory", "safe_output_child"]
