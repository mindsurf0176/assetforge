from __future__ import annotations

import json
import statistics
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .frames import (
    _metric,
    frame_paths,
    neutral_foreground_fringe_pixels,
    select_requested_animation_paths,
)
from .profile import Profile


def _drift(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    median = statistics.median(values)
    return 0.0 if median <= 0 else (max(values) - min(values)) / median


def enclosed_transparent_hole_areas(image: Image.Image, min_alpha: int) -> list[int]:
    """Return pixel areas for transparent regions that cannot reach the canvas edge.

    Eight-neighbour connectivity deliberately matches the background flood-fill used
    during normalization: a diagonal path to the exterior is enough to make a pixel
    background rather than a hole in the foreground silhouette.
    """

    alpha = np.asarray(image.convert("RGBA"), dtype=np.uint8)[:, :, 3]
    transparent = alpha <= min_alpha
    height, width = transparent.shape
    exterior = np.zeros((height, width), dtype=bool)
    pending: deque[tuple[int, int]] = deque()

    def seed(y: int, x: int) -> None:
        if transparent[y, x] and not exterior[y, x]:
            exterior[y, x] = True
            pending.append((y, x))

    for x in range(width):
        seed(0, x)
        if height > 1:
            seed(height - 1, x)
    for y in range(height):
        seed(y, 0)
        if width > 1:
            seed(y, width - 1)

    neighbours = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    while pending:
        y, x = pending.popleft()
        for dy, dx in neighbours:
            next_y, next_x = y + dy, x + dx
            if (
                0 <= next_y < height
                and 0 <= next_x < width
                and transparent[next_y, next_x]
                and not exterior[next_y, next_x]
            ):
                exterior[next_y, next_x] = True
                pending.append((next_y, next_x))

    enclosed = transparent & ~exterior
    visited = np.zeros((height, width), dtype=bool)
    areas: list[int] = []
    for y, x in zip(*np.nonzero(enclosed)):
        if visited[y, x]:
            continue
        visited[y, x] = True
        pending.append((int(y), int(x)))
        area = 0
        while pending:
            current_y, current_x = pending.popleft()
            area += 1
            for dy, dx in neighbours:
                next_y, next_x = current_y + dy, current_x + dx
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and enclosed[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    pending.append((next_y, next_x))
        areas.append(area)

    return sorted(areas, reverse=True)


def validate_frames(
    profile: Profile,
    input_dir: str | Path,
    tier_name: str,
    animation: str | None = None,
    report_path: str | Path | None = None,
    *,
    placement_mode: str = "per-frame-anchor",
) -> dict[str, Any]:
    if placement_mode not in {"per-frame-anchor", "shared-motion"}:
        raise ValueError(
            "placement_mode must be 'per-frame-anchor' or 'shared-motion'"
        )
    tier = profile.tier(tier_name)
    quality = profile.data["quality"]
    animation_contract = profile.animation(animation) if animation else {}
    min_alpha = int(quality.get("alphaThreshold", 20))
    background_quality = quality.get("background", {})
    edge_matte = quality.get("edgeMatte", {})
    edge_matte_min_rgb = int(edge_matte.get("minRgb", 235))
    edge_matte_spread = int(edge_matte.get("maxChannelSpread", 18))
    edge_matte_limit = edge_matte.get("maxPixels")
    if edge_matte_limit is not None:
        edge_matte_limit = int(edge_matte_limit)
    max_enclosed_transparent_pixels = background_quality.get("maxEnclosedTransparentPixels")
    if max_enclosed_transparent_pixels is not None:
        max_enclosed_transparent_pixels = int(max_enclosed_transparent_pixels)
    paths = frame_paths(input_dir)
    selection_error: str | None = None
    if animation:
        try:
            paths = select_requested_animation_paths(
                paths,
                animation,
                profile.data.get("animations", {}).keys(),
            )
        except ValueError as exc:
            paths = []
            selection_error = str(exc)
    metrics = []
    transparent_holes: list[dict[str, Any]] = []
    edge_matte_report: list[dict[str, Any]] = []
    errors: list[str] = [selection_error] if selection_error else []
    warnings: list[str] = []

    for path in paths:
        try:
            image = Image.open(path).convert("RGBA")
            metrics.append(_metric(path, image, min_alpha))
            hole_areas = enclosed_transparent_hole_areas(image, min_alpha)
            hole_pixels = sum(hole_areas)
            neutral_fringe = neutral_foreground_fringe_pixels(
                image,
                min_alpha=min_alpha,
                min_rgb=edge_matte_min_rgb,
                max_channel_spread=edge_matte_spread,
            )
            transparent_holes.append(
                {
                    "file": path.name,
                    "componentCount": len(hole_areas),
                    "pixelCount": hole_pixels,
                    "largestComponentPixels": hole_areas[0] if hole_areas else 0,
                }
            )
            edge_matte_report.append(
                {"file": path.name, "neutralPixels": neutral_fringe}
            )
            if edge_matte_limit is not None and neutral_fringe > edge_matte_limit:
                errors.append(
                    f"{path.name}: {neutral_fringe} neutral foreground matte pixels exceeds "
                    f"edgeMatte.maxPixels={edge_matte_limit}"
                )
            if (
                placement_mode != "shared-motion"
                and
                max_enclosed_transparent_pixels is not None
                and hole_pixels > max_enclosed_transparent_pixels
            ):
                errors.append(
                    f"{path.name}: {hole_pixels} enclosed transparent pixels across "
                    f"{len(hole_areas)} component(s) exceeds "
                    "background.maxEnclosedTransparentPixels="
                    f"{max_enclosed_transparent_pixels}"
                )
        except ValueError as exc:
            errors.append(str(exc))

    if tier.get("canvasPolicy", "fixed") == "fixed":
        expected = list(map(int, tier["canvas"]))
        for metric in metrics:
            if metric.canvas != expected:
                errors.append(f"{metric.file}: canvas {metric.canvas} != expected {expected}")

    content_min = tier.get("contentMin")
    if content_min:
        minimum_width, minimum_height = map(int, content_min)
        for metric in metrics:
            width, height = metric.contentSize
            if width < minimum_width or height < minimum_height:
                errors.append(
                    f"{metric.file}: content {metric.contentSize} is below "
                    f"contentMin={[minimum_width, minimum_height]}"
                )

    anchor_quality = dict(quality.get("anchor", {}))
    anchor_quality.update(animation_contract.get("anchor", {}))
    anchor_tolerance = int(anchor_quality.get("maxFootDrift", 1))
    anchor = tier.get("anchor")
    # A preserved rig owns its pixel coordinates. Applying the profile's
    # generated-frame ground-line gate here would reject valid hand-authored
    # RigSpec placement before the dedicated overflow checks get a chance to
    # report the real contract violation.
    if anchor and not tier.get("preservePlacement", False):
        expected_y = int(anchor[1])
        if placement_mode == "shared-motion":
            # The first authored pose must start on the profile ground line.
            # Later non-loop frames may intentionally lunge, recoil, or fall.
            if metrics and abs(metrics[0].foot[1] - expected_y) > anchor_tolerance:
                errors.append(
                    f"{metrics[0].file}: initial foot y={metrics[0].foot[1]} exceeds "
                    f"anchor y={expected_y} tolerance={anchor_tolerance} in shared-motion mode"
                )
            if bool(animation_contract.get("loop", True)) and metrics:
                feet = [metric.foot[1] for metric in metrics]
                if max(feet) - min(feet) > anchor_tolerance:
                    errors.append(
                        f"foot-line drift {max(feet) - min(feet)}px exceeds "
                        f"tolerance={anchor_tolerance}px in shared-motion mode"
                    )
        else:
            for metric in metrics:
                if abs(metric.foot[1] - expected_y) > anchor_tolerance:
                    errors.append(
                        f"{metric.file}: foot y={metric.foot[1]} exceeds anchor y={expected_y} "
                        f"tolerance={anchor_tolerance}"
                    )
    elif metrics:
        feet = [metric.foot[1] for metric in metrics]
        if max(feet) - min(feet) > anchor_tolerance:
            errors.append(
                f"foot-line drift {max(feet) - min(feet)}px exceeds "
                f"tolerance={anchor_tolerance}px"
            )

    max_colors = int(quality.get("palette", {}).get("maxColors", 256))
    for metric in metrics:
        if metric.colors > max_colors:
            errors.append(f"{metric.file}: {metric.colors} colors exceeds maxColors={max_colors}")

    heights = [metric.contentSize[1] for metric in metrics]
    widths = [metric.contentSize[0] for metric in metrics]
    identity = dict(quality.get("identity", {}))
    identity.update(animation_contract.get("identity", {}))
    height_drift = _drift(heights)
    width_drift = _drift(widths)
    if height_drift > float(identity.get("maxHeightDriftRatio", 1.0)):
        errors.append(f"content height drift {height_drift:.3f} exceeds limit")
    if width_drift > float(identity.get("maxWidthDriftRatio", 1.0)):
        warnings.append(f"content width drift {width_drift:.3f} exceeds soft limit")

    if animation:
        minimum = int(animation_contract.get("minFrames", 1))
        maximum = int(animation_contract.get("maxFrames", 999))
        if not minimum <= len(metrics) <= maximum:
            errors.append(f"animation {animation!r} has {len(metrics)} frames; expected {minimum}..{maximum}")

    result = {
        "ok": not errors,
        "profile": profile.id,
        "profileFingerprint": profile.fingerprint,
        "tier": tier_name,
        "animation": animation,
        "placementMode": placement_mode,
        "enclosedTransparencyPolicy": (
            "report-only" if placement_mode == "shared-motion" else "profile-gate"
        ),
        "input": str(Path(input_dir).expanduser().resolve()),
        "frameCount": len(metrics),
        "heightDriftRatio": round(height_drift, 4),
        "widthDriftRatio": round(width_drift, 4),
        "errors": errors,
        "warnings": warnings,
        "transparentHoles": transparent_holes,
        "edgeMatte": {
            "minRgb": edge_matte_min_rgb,
            "maxChannelSpread": edge_matte_spread,
            "maxPixels": edge_matte_limit,
            "frames": edge_matte_report,
        },
        "frames": [metric.__dict__ for metric in metrics],
    }
    if report_path:
        path = Path(report_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
