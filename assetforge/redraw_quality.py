from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .json_utils import strict_json_loads
from .path_safety import safe_output_child


DEFAULT_THRESHOLDS: dict[str, float] = {
    "background_foreground_distance": 32.0,
    "background_drift_distance": 28.0,
    "max_background_drift_ratio": 0.025,
    "identity_min_mask_iou": 0.78,
    "identity_min_color_similarity": 0.86,
    "identity_min_score": 0.82,
    "guide_color_distance": 24.0,
    "max_pose_guide_residual": 0.05,
    "min_cell_silhouette_iou": 0.55,
    "min_cell_occupancy_ratio": 0.65,
    "max_cell_occupancy_ratio": 1.45,
    "max_unused_foreground_ratio": 0.003,
}

_POSE_GUIDE_COLORS = np.asarray(
    ((70, 104, 138), (244, 238, 192)),
    dtype=np.float32,
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _regular_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _pixel_sha256(path: Path, *, label: str) -> str:
    try:
        with Image.open(path) as opened:
            rendered = opened.convert("RGB")
            rendered.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"{label} is not a readable image: {path}") from exc
    return hashlib.sha256(rendered.tobytes()).hexdigest()


def _manifest_child(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative to the dataset manifest: {value}")
    candidate = safe_output_child(root, *relative.parts, label=label)
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} not found: {candidate}")
    return candidate


def _load_rgb(path: Path, *, label: str) -> tuple[np.ndarray, str | None]:
    try:
        with Image.open(path) as opened:
            image_format = opened.format
            pixels = np.asarray(opened.convert("RGB"), dtype=np.float32)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"{label} is not a readable image: {path}") from exc
    return pixels, image_format


def _round(value: float) -> float:
    return round(float(value), 6)


def _distance_from_color(pixels: np.ndarray, color: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pixels - color, axis=2)


def _foreground_mask(pixels: np.ndarray, background: np.ndarray, threshold: float) -> np.ndarray:
    return _distance_from_color(pixels, background) > threshold


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    for y_offset in range(radius * 2 + 1):
        for x_offset in range(radius * 2 + 1):
            result |= padded[y_offset : y_offset + height, x_offset : x_offset + width]
    return result


def _cell(pixels: np.ndarray, index: int, columns: int, rows: int) -> np.ndarray:
    cell_width = pixels.shape[1] // columns
    cell_height = pixels.shape[0] // rows
    row, column = divmod(index, columns)
    return pixels[
        row * cell_height : (row + 1) * cell_height,
        column * cell_width : (column + 1) * cell_width,
    ]


def _infer_grid(width: int, height: int, required_cells: int) -> tuple[int, int]:
    # Paired redraw boards are 3x3 by contract. Prefer that layout when an old
    # manifest predates explicit columns/rows metadata; otherwise infer a
    # square-cell grid for non-square fixture/custom boards.
    if width == height and width % 3 == 0 and required_cells <= 9:
        return 3, 3
    candidates: list[tuple[int, int, int]] = []
    for columns in range(1, 13):
        for rows in range(1, 13):
            if width % columns or height % rows or columns * rows < required_cells:
                continue
            if width // columns != height // rows:
                continue
            candidates.append((columns * rows, columns, rows))
    if not candidates:
        raise ValueError(
            f"could not infer a square-cell board grid for {width}x{height} and "
            f"{required_cells} required cells; pass columns and rows explicitly"
        )
    _, columns, rows = min(candidates, key=lambda item: (item[0], abs(item[1] - item[2]), item[1]))
    return columns, rows


def _background_from_corners(*boards: np.ndarray) -> np.ndarray:
    corners: list[np.ndarray] = []
    for board in boards:
        corners.extend((board[0, 0], board[0, -1], board[-1, 0], board[-1, -1]))
    return np.median(np.asarray(corners, dtype=np.float32), axis=0)


def _thresholds(overrides: Mapping[str, float] | None) -> dict[str, float]:
    values = dict(DEFAULT_THRESHOLDS)
    if overrides is None:
        return values
    unknown = sorted(set(overrides) - set(values))
    if unknown:
        raise ValueError(f"unknown redraw quality thresholds: {', '.join(unknown)}")
    for name, raw_value in overrides.items():
        value = float(raw_value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"redraw quality threshold {name} must be a finite non-negative number")
        values[name] = value
    return values


def _manifest_inputs(manifest_path: str | Path, sample_id: str) -> dict[str, Any]:
    manifest = _regular_file(manifest_path, label="redraw dataset manifest")
    data = strict_json_loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        raise ValueError("redraw dataset manifest must contain a samples array")
    matches = [sample for sample in data["samples"] if isinstance(sample, dict) and sample.get("id") == sample_id]
    if not matches:
        raise ValueError(f"redraw dataset sample not found: {sample_id}")
    if len(matches) != 1:
        raise ValueError(f"redraw dataset sample id is not unique: {sample_id}")
    sample = matches[0]
    if sample.get("split") != "validation":
        raise ValueError(f"redraw quality manifest sample must be a validation holdout: {sample_id}")
    frame_count = sample.get("frameCount")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 1:
        raise ValueError(f"redraw dataset sample {sample_id} has an invalid frameCount")
    board = data.get("board")
    if not isinstance(board, dict):
        raise ValueError("redraw dataset manifest must contain board metadata")
    background = board.get("background")
    if (
        not isinstance(background, list)
        or len(background) != 3
        or any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255 for value in background)
    ):
        raise ValueError("redraw dataset board.background must contain three RGB integers")
    input_file = _manifest_child(manifest.parent, sample.get("input"), label="redraw sample input")
    target_file = _manifest_child(manifest.parent, sample.get("target"), label="redraw sample target")
    for path, digest_field, label in (
        (input_file, "inputPixelSha256", "redraw sample input"),
        (target_file, "targetPixelSha256", "redraw sample target"),
    ):
        declared = sample.get(digest_field)
        if not isinstance(declared, str) or not _SHA256.fullmatch(declared):
            raise ValueError(f"redraw dataset sample {sample_id} has an invalid {digest_field}")
        if _pixel_sha256(path, label=label) != declared:
            raise ValueError(f"redraw dataset sample {sample_id} {label} pixels do not match {digest_field}")
    return {
        "manifest": manifest,
        "input": input_file,
        "target": target_file,
        "frame_count": frame_count,
        "background": background,
        "columns": board.get("columns"),
        "rows": board.get("rows"),
    }


def evaluate_redraw_quality(
    *,
    generated_path: str | Path,
    manifest_path: str | Path | None = None,
    sample_id: str | None = None,
    input_path: str | Path | None = None,
    target_path: str | Path | None = None,
    columns: int | None = None,
    rows: int | None = None,
    expected_frames: int | None = None,
    background: tuple[int, int, int] | list[int] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Measure whether a generated paired-board is complete enough for review.

    Use either ``manifest_path`` plus ``sample_id`` or explicit input/target paths.
    Quality failures are returned as blockers; invalid contracts and unsafe paths raise.
    """

    manifest_mode = manifest_path is not None or sample_id is not None
    explicit_mode = input_path is not None or target_path is not None
    if manifest_mode and explicit_mode:
        raise ValueError("use either manifest sample lookup or explicit input/target paths, not both")
    manifest: Path | None = None
    if manifest_mode:
        if manifest_path is None or not sample_id:
            raise ValueError("manifest_path and sample_id are both required for manifest sample lookup")
        resolved = _manifest_inputs(manifest_path, sample_id)
        manifest = resolved["manifest"]
        resolved_input = resolved["input"]
        resolved_target = resolved["target"]
        if expected_frames is None:
            expected_frames = resolved["frame_count"]
        if background is None:
            background = resolved["background"]
        if columns is None and isinstance(resolved["columns"], int):
            columns = resolved["columns"]
        if rows is None and isinstance(resolved["rows"], int):
            rows = resolved["rows"]
    else:
        if input_path is None or target_path is None:
            raise ValueError("explicit input_path and target_path are both required")
        resolved_input = _regular_file(input_path, label="redraw input")
        resolved_target = _regular_file(target_path, label="redraw target")

    generated = _regular_file(generated_path, label="generated redraw")
    input_pixels, input_format = _load_rgb(resolved_input, label="redraw input")
    target_pixels, target_format = _load_rgb(resolved_target, label="redraw target")
    generated_pixels, generated_format = _load_rgb(generated, label="generated redraw")
    if input_format != "PNG" or target_format != "PNG":
        raise ValueError("redraw input and target must both be PNG images")
    if input_pixels.shape != target_pixels.shape:
        raise ValueError(
            f"redraw input and target dimensions differ: "
            f"{input_pixels.shape[1]}x{input_pixels.shape[0]} vs "
            f"{target_pixels.shape[1]}x{target_pixels.shape[0]}"
        )

    values = _thresholds(thresholds)
    height, width = target_pixels.shape[:2]
    if expected_frames is None:
        expected_frames = 8
    if not isinstance(expected_frames, int) or isinstance(expected_frames, bool) or expected_frames < 1:
        raise ValueError("expected_frames must be a positive integer")
    if columns is None and rows is None:
        columns, rows = _infer_grid(width, height, expected_frames + 1)
    elif not isinstance(columns, int) or isinstance(columns, bool) or not isinstance(rows, int) or isinstance(rows, bool):
        raise ValueError("columns and rows must be supplied together as positive integers")
    if columns < 1 or rows < 1 or width % columns or height % rows:
        raise ValueError(f"invalid {columns}x{rows} grid for {width}x{height} board")
    if expected_frames + 1 > columns * rows:
        raise ValueError(
            f"expected_frames={expected_frames} exceeds {columns}x{rows} board capacity after cell 0"
        )

    if background is None:
        background_array = _background_from_corners(input_pixels, target_pixels)
    else:
        if (
            len(background) != 3
            or any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255 for value in background)
        ):
            raise ValueError("background must contain three RGB integers")
        background_array = np.asarray(background, dtype=np.float32)

    blockers: list[str] = []
    format_passed = generated_format == "PNG"
    if not format_passed:
        blockers.append(f"generated image format is {generated_format or 'unknown'}, expected PNG")
    dimensions_passed = generated_pixels.shape == target_pixels.shape
    if not dimensions_passed:
        blockers.append(
            f"generated image is {generated_pixels.shape[1]}x{generated_pixels.shape[0]}, "
            f"expected {width}x{height}"
        )
        return {
            "ok": False,
            "passed": False,
            "sampleId": sample_id,
            "manifest": str(manifest) if manifest else None,
            "paths": {
                "input": str(resolved_input),
                "target": str(resolved_target),
                "generated": str(generated),
            },
            "board": {
                "size": [width, height],
                "columns": columns,
                "rows": rows,
                "expectedFrames": expected_frames,
            },
            "metrics": {
                "format": {"value": generated_format, "passed": format_passed},
                "dimensions": {
                    "value": [generated_pixels.shape[1], generated_pixels.shape[0]],
                    "expected": [width, height],
                    "passed": False,
                },
            },
            "thresholds": values,
            "blockers": blockers,
        }

    foreground_distance = values["background_foreground_distance"]
    input_mask = _foreground_mask(input_pixels, background_array, foreground_distance)
    target_mask = _foreground_mask(target_pixels, background_array, foreground_distance)
    generated_mask = _foreground_mask(generated_pixels, background_array, foreground_distance)

    identity_target = _cell(target_pixels, 0, columns, rows)
    identity_generated = _cell(generated_pixels, 0, columns, rows)
    identity_target_mask = _cell(target_mask, 0, columns, rows)
    identity_generated_mask = _cell(generated_mask, 0, columns, rows)
    identity_iou = _mask_iou(identity_target_mask, identity_generated_mask)
    identity_overlap = np.logical_and(identity_target_mask, identity_generated_mask)
    if identity_overlap.any():
        mean_identity_distance = float(
            np.linalg.norm(identity_target - identity_generated, axis=2)[identity_overlap].mean()
        )
        color_similarity = max(0.0, 1.0 - mean_identity_distance / np.sqrt(3.0 * 255.0**2))
    else:
        mean_identity_distance = float(np.sqrt(3.0 * 255.0**2))
        color_similarity = 0.0
    identity_score = 0.65 * identity_iou + 0.35 * color_similarity
    identity_passed = bool(
        identity_iou >= values["identity_min_mask_iou"]
        and color_similarity >= values["identity_min_color_similarity"]
        and identity_score >= values["identity_min_score"]
    )
    if identity_iou < values["identity_min_mask_iou"]:
        blockers.append(
            f"cell 0 identity silhouette IoU {_round(identity_iou)} is below "
            f"{values['identity_min_mask_iou']}"
        )
    if color_similarity < values["identity_min_color_similarity"]:
        blockers.append(
            f"cell 0 identity color similarity {_round(color_similarity)} is below "
            f"{values['identity_min_color_similarity']}"
        )
    if identity_score < values["identity_min_score"]:
        blockers.append(
            f"cell 0 combined identity score {_round(identity_score)} is below "
            f"{values['identity_min_score']}"
        )

    cell_reports: list[dict[str, Any]] = []
    guide_pixels_total = 0
    guide_residual_total = 0
    for index in range(1, expected_frames + 1):
        input_cell = _cell(input_pixels, index, columns, rows)
        target_cell = _cell(target_pixels, index, columns, rows)
        generated_cell = _cell(generated_pixels, index, columns, rows)
        input_cell_mask = _cell(input_mask, index, columns, rows)
        target_cell_mask = _cell(target_mask, index, columns, rows)
        generated_cell_mask = _cell(generated_mask, index, columns, rows)
        target_count = int(target_cell_mask.sum())
        generated_count = int(generated_cell_mask.sum())
        occupancy_ratio = generated_count / target_count if target_count else (1.0 if generated_count == 0 else float("inf"))
        silhouette_iou = _mask_iou(target_cell_mask, generated_cell_mask)
        guide_distances = np.linalg.norm(
            generated_cell[:, :, None, :] - _POSE_GUIDE_COLORS[None, None, :, :],
            axis=3,
        ).min(axis=2)
        input_guide_count = int(input_cell_mask.sum())
        guide_residual_count = int(
            np.logical_and(guide_distances <= values["guide_color_distance"], input_cell_mask).sum()
        )
        guide_residual = guide_residual_count / input_guide_count if input_guide_count else 0.0
        guide_pixels_total += input_guide_count
        guide_residual_total += guide_residual_count
        occupancy_passed = (
            values["min_cell_occupancy_ratio"]
            <= occupancy_ratio
            <= values["max_cell_occupancy_ratio"]
        )
        silhouette_passed = silhouette_iou >= values["min_cell_silhouette_iou"]
        guide_passed = guide_residual <= values["max_pose_guide_residual"]
        completed = bool(
            target_count > 0
            and generated_count > 0
            and occupancy_passed
            and silhouette_passed
            and guide_passed
        )
        if not completed:
            reasons: list[str] = []
            if generated_count == 0:
                reasons.append("no foreground")
            if not occupancy_passed:
                reasons.append(f"occupancy ratio {_round(occupancy_ratio)}")
            if not silhouette_passed:
                reasons.append(f"silhouette IoU {_round(silhouette_iou)}")
            if not guide_passed:
                reasons.append(f"pose-guide residual {_round(guide_residual)}")
            blockers.append(f"expected cell {index} is incomplete: {', '.join(reasons)}")
        cell_reports.append(
            {
                "index": index,
                "targetForegroundPixels": target_count,
                "generatedForegroundPixels": generated_count,
                "occupancyRatio": _round(occupancy_ratio),
                "silhouetteIoU": _round(silhouette_iou),
                "poseGuideResidual": _round(guide_residual),
                "completed": completed,
            }
        )

    pose_guide_residual = guide_residual_total / guide_pixels_total if guide_pixels_total else 0.0
    pose_guide_passed = pose_guide_residual <= values["max_pose_guide_residual"]
    if not pose_guide_passed:
        blockers.append(
            f"board pose-guide residual {_round(pose_guide_residual)} exceeds "
            f"{values['max_pose_guide_residual']}"
        )

    unused_reports: list[dict[str, Any]] = []
    unused_passed = True
    cell_area = (width // columns) * (height // rows)
    for index in range(expected_frames + 1, columns * rows):
        foreground_pixels = int(_cell(generated_mask, index, columns, rows).sum())
        foreground_ratio = foreground_pixels / cell_area
        blank = foreground_ratio <= values["max_unused_foreground_ratio"]
        unused_passed &= blank
        if not blank:
            blockers.append(
                f"unused cell {index} foreground ratio {_round(foreground_ratio)} exceeds "
                f"{values['max_unused_foreground_ratio']}"
            )
        unused_reports.append(
            {
                "index": index,
                "foregroundPixels": foreground_pixels,
                "foregroundRatio": _round(foreground_ratio),
                "blank": blank,
            }
        )

    stable_background = np.logical_not(_dilate(np.logical_or(input_mask, target_mask), radius=1))
    generated_background_distances = _distance_from_color(generated_pixels, background_array)
    if stable_background.any():
        stable_distances = generated_background_distances[stable_background]
        background_drift_ratio = float(
            (stable_distances > values["background_drift_distance"]).mean()
        )
        mean_background_distance = float(stable_distances.mean())
    else:
        background_drift_ratio = 1.0
        mean_background_distance = float("inf")
    background_passed = background_drift_ratio <= values["max_background_drift_ratio"]
    if not background_passed:
        blockers.append(
            f"board background drift ratio {_round(background_drift_ratio)} exceeds "
            f"{values['max_background_drift_ratio']}"
        )

    completed_count = sum(report["completed"] for report in cell_reports)
    passed = not blockers
    return {
        "ok": passed,
        "passed": passed,
        "sampleId": sample_id,
        "manifest": str(manifest) if manifest else None,
        "paths": {
            "input": str(resolved_input),
            "target": str(resolved_target),
            "generated": str(generated),
        },
        "board": {
            "size": [width, height],
            "columns": columns,
            "rows": rows,
            "cellSize": [width // columns, height // rows],
            "expectedFrames": expected_frames,
            "unusedCells": columns * rows - expected_frames - 1,
            "background": [int(round(value)) for value in background_array],
        },
        "metrics": {
            "format": {"value": generated_format, "passed": format_passed},
            "dimensions": {"value": [width, height], "expected": [width, height], "passed": True},
            "identity": {
                "maskIoU": _round(identity_iou),
                "meanForegroundColorDistance": _round(mean_identity_distance),
                "colorSimilarity": _round(color_similarity),
                "score": _round(identity_score),
                "passed": identity_passed,
            },
            "poseGuide": {
                "residualPixels": guide_residual_total,
                "guidePixels": guide_pixels_total,
                "residualRatio": _round(pose_guide_residual),
                "passed": pose_guide_passed,
            },
            "expectedCells": {
                "completed": completed_count,
                "required": expected_frames,
                "passed": completed_count == expected_frames,
                "cells": cell_reports,
            },
            "unusedCells": {"passed": unused_passed, "cells": unused_reports},
            "background": {
                "stablePixelCount": int(stable_background.sum()),
                "meanColorDistance": _round(mean_background_distance),
                "driftRatio": _round(background_drift_ratio),
                "passed": background_passed,
            },
        },
        "thresholds": values,
        "blockers": blockers,
    }


def evaluate_redraw_sample(
    manifest_path: str | Path,
    sample_id: str,
    generated_path: str | Path,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    return evaluate_redraw_quality(
        manifest_path=manifest_path,
        sample_id=sample_id,
        generated_path=generated_path,
        thresholds=thresholds,
    )


def evaluate_redraw_images(
    input_path: str | Path,
    target_path: str | Path,
    generated_path: str | Path,
    *,
    columns: int = 3,
    rows: int = 3,
    expected_frames: int = 8,
    background: tuple[int, int, int] | list[int] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    return evaluate_redraw_quality(
        input_path=input_path,
        target_path=target_path,
        generated_path=generated_path,
        columns=columns,
        rows=rows,
        expected_frames=expected_frames,
        background=background,
        thresholds=thresholds,
    )


__all__ = [
    "DEFAULT_THRESHOLDS",
    "evaluate_redraw_images",
    "evaluate_redraw_quality",
    "evaluate_redraw_sample",
]
