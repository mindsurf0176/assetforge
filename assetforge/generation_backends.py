"""Model-neutral generation backend contracts.

Backends produce a complete animation sheet. AssetForge owns all deterministic
processing after that point, so a provider cannot silently change frame layout or
promotion rules.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class SheetRequest:
    character: str
    animation: str
    direction: str
    frame_count: int
    columns: int
    rows: int = 1
    reference: Path | None = None
    pose_guide: Path | None = None
    seed: int | None = None
    prompt: str = ""
    negative_prompt: str = ""


@dataclass(frozen=True)
class SheetResult:
    sheet: Path
    backend: str
    metadata: dict[str, Any]


class SheetGenerator(Protocol):
    name: str

    def generate(self, request: SheetRequest, output: Path) -> SheetResult:
        ...


def write_generation_manifest(
    result: SheetResult,
    request: SheetRequest,
    output: Path,
) -> Path:
    """Persist reproducibility metadata next to a generated sheet."""

    output.mkdir(parents=True, exist_ok=True)
    output_root = output.resolve()

    def portable(path: Path | None) -> str | None:
        if path is None:
            return None
        return os.path.relpath(path.resolve(), output_root)

    manifest = {
        "schemaVersion": 1,
        "backend": result.backend,
        "character": request.character,
        "animation": request.animation,
        "direction": request.direction,
        "frameCount": request.frame_count,
        "columns": request.columns,
        "rows": request.rows,
        "reference": portable(request.reference),
        "poseGuide": portable(request.pose_guide),
        "seed": request.seed,
        "prompt": request.prompt,
        "negativePrompt": request.negative_prompt,
        "sheet": portable(result.sheet),
        "metadata": result.metadata,
    }
    path = output / "generation-manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


class DiffusersSheetGenerator:
    """Explicit extension point for Diffusers-based sheet workflows.

    The actual model pipeline is configured by the caller because ControlNet,
    IP-Adapter, and model-specific scheduler settings differ across checkpoints.
    """

    name = "diffusers"

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def generate(self, request: SheetRequest, output: Path) -> SheetResult:
        raise NotImplementedError(
            "configure the model-specific Diffusers pipeline and implement its "
            "single-sheet output mapping"
        )


def codex_imagegen_sheet_prompt(request: SheetRequest) -> str:
    """Build the canonical prompt for the built-in Codex Imagegen backend."""

    reference = "Use the supplied identity reference image as Image 1." if request.reference else "No reference image is supplied; establish one clear identity and keep it fixed."
    guide = "Use the supplied pose guide as Image 2." if request.pose_guide else "Design a readable eight-phase motion arc across the cells."
    return f"""Use case: stylized-concept
Asset type: production candidate animation sheet for a 2D game
Primary request: generate exactly one complete {request.animation} animation sheet for {request.character}, facing {request.direction}. The sheet contains exactly {request.frame_count} unique poses arranged in a {request.rows}-row by {request.columns}-column grid, read left-to-right across each row, then continue on the next row.
Input images: {reference} {guide}
Scene/backdrop: genuinely transparent background; if transparency is not possible, use one perfectly flat saturated green background #00ff00 with no texture or gradient
Subject: one consistent full-body character only; preserve the same face, hair, costume, weapons, silhouette, proportions, palette, camera distance, and baseline in every cell
Style/medium: dense authored native pixel art with crisp hard pixel clusters, no anti-aliasing, no painterly smoothing
Composition/framing: every cell has identical dimensions and camera framing; one pose per cell; generous empty margin; feet rest on one shared horizontal ground line; no crop or overlap into neighboring cells
Color palette: consistent restrained palette across all cells; no palette drift
Constraints: output only the animation sheet, no labels, no numbers, no borders, no UI, no watermark; keep all {request.frame_count} cells populated with distinct poses; preserve equipment attachment points
Avoid: checkerboard, black/gray/white backdrop, gradients, glow haze, light matte halo, extra characters, duplicated limbs, missing hands or feet, changing weapon length, front view, three-quarter view, camera zoom changes, random costume changes, interpolated duplicate holds
Motion: {request.animation} should read as a continuous loop with distinct contact, passing, weight-transfer, and recovery poses while maintaining one identity."""
