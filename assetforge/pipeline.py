"""Provider-independent one-shot asset build pipeline.

The generator is deliberately outside this module. Any image model, desktop
tool, or hand-authored workflow can hand AssetForge either a complete sheet or
an already split frame directory; the deterministic stages after that input
are identical.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import __version__
from .exporters import export_assets
from .frames import ingest_frames, infer_source_sheet_anchors, split_source_sheet
from .json_utils import strict_json_loads
from .profile import Profile
from .validation import validate_frames


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_generation_manifest(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"generation manifest not found: {manifest_path}")
    data = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("generation manifest must be a JSON object")
    if not data.get("backend"):
        raise ValueError("generation manifest requires a non-empty backend")
    return {"path": manifest_path.name, "data": data}


def build_pipeline(
    profile: Profile,
    input_path: str | Path,
    work: str | Path,
    output: str | Path,
    *,
    character: str,
    tier: str,
    animation: str,
    direction: str,
    input_kind: str = "auto",
    columns: int | None = None,
    rows: int = 1,
    frame_count: int | None = None,
    auto_anchor: bool = True,
    source_anchor: tuple[int, int] | None = None,
    source_anchors: list[tuple[int, int]] | None = None,
    source_bounds: tuple[int, int, int, int] | None = None,
    backend: str = "external",
    generation_manifest: str | Path | None = None,
    resource_prefix: str | None = None,
    deploy_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete model-neutral sheet/frame -> validated export flow."""

    source = Path(input_path).expanduser().resolve()
    if source.is_symlink() or not source.exists():
        raise FileNotFoundError(f"pipeline input not found: {source}")
    if input_kind == "auto":
        input_kind = "sheet" if source.is_file() else "frames"
    if input_kind not in {"sheet", "frames"}:
        raise ValueError("input kind must be 'sheet' or 'frames'")
    if input_kind == "sheet" and columns is None:
        raise ValueError("--columns is required when --input-kind is sheet")
    if input_kind == "sheet" and (columns or 0) < 1:
        raise ValueError("columns must be positive")
    if rows < 1:
        raise ValueError("rows must be positive")

    work_root = Path(work).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    frame_input = source
    anchors = source_anchors
    bounds = source_bounds
    placement_mode = "per-frame-anchor"
    if input_kind == "sheet":
        sheet_frames = work_root / "source-frames"
        split = split_source_sheet(
            source,
            sheet_frames,
            columns=columns or 1,
            rows=rows,
            frame_count=frame_count,
            prefix=animation,
        )
        frame_input = sheet_frames
        if auto_anchor:
            anchors, bounds = infer_source_sheet_anchors(split)
        else:
            if source_anchor is None or bounds is None:
                raise ValueError("manual sheet placement requires source anchor and source bounds")
            placement_mode = "shared-motion"

    normalized = work_root / "frames"
    manifest = ingest_frames(
        profile,
        frame_input,
        normalized,
        tier,
        animation,
        direction,
        placement_mode=placement_mode,
        source_anchor=source_anchor,
        source_anchors=anchors,
        source_bounds=bounds,
        allow_source_resize=input_kind == "sheet" and auto_anchor,
    )
    report_path = work_root / "validation-report.json"
    validation = validate_frames(profile, normalized, tier, animation, report_path, placement_mode=placement_mode)
    generation = _load_generation_manifest(generation_manifest)
    pipeline_manifest = {
        "schemaVersion": 1,
        "assetforgeVersion": __version__,
        "backend": backend,
        "inputKind": input_kind,
        "source": {"name": source.name, "sha256": _sha256(source)} if source.is_file() else {"name": source.name},
        "character": character,
        "tier": tier,
        "animation": animation,
        "direction": direction,
        "frameCount": len(manifest.get("frames", [])),
        "columns": columns,
        "rows": rows,
        "placementMode": placement_mode,
        "validation": {"ok": validation.get("ok"), "report": report_path.name},
        "generationManifest": generation,
    }
    pipeline_manifest_path = work_root / "pipeline-manifest.json"
    pipeline_manifest_path.write_text(json.dumps(pipeline_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "ok": False,
        "backend": backend,
        "inputKind": input_kind,
        "pipelineManifest": str(pipeline_manifest_path),
        "manifest": manifest,
        "validation": validation,
    }
    if not validation["ok"]:
        result["stage"] = "validate"
        return result
    result["export"] = export_assets(
        profile, normalized, output, character, tier, animation, direction, resource_prefix, deploy_dir
    )
    result["ok"] = True
    return result
