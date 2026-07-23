from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .frames import alpha_bbox, remove_corner_background
from .json_utils import strict_json_loads
from .path_safety import reset_output_directory, safe_output_child
from .redraw_quality import evaluate_redraw_sample


_SAFE_SAMPLE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_OUTPUT_MARKER = ".assetforge-output.json"
_OUTPUT_MARKER_CONTENT = {"owner": "assetforge", "schemaVersion": 1}


def _manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"redraw manifest must not be a symbolic link: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"redraw manifest not found: {resolved}")
    value = strict_json_loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("samples"), list):
        raise ValueError("redraw manifest must contain a samples array")
    return resolved, value


def _sample(value: dict[str, Any], sample_id: str) -> dict[str, Any]:
    if not _SAFE_SAMPLE_ID.fullmatch(sample_id):
        raise ValueError(f"unsafe redraw sample id: {sample_id!r}")
    matches = [
        sample
        for sample in value["samples"]
        if isinstance(sample, dict) and sample.get("id") == sample_id
    ]
    if len(matches) != 1:
        raise ValueError(f"redraw sample id must exist exactly once: {sample_id}")
    return matches[0]


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_size(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain two positive integers")
    return (
        _positive_integer(value[0], f"{label}[0]"),
        _positive_integer(value[1], f"{label}[1]"),
    )


def _delivery_contract(value: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    """Validate the geometry needed to split a generated paired board."""

    sample_id = sample.get("id")
    if not isinstance(sample_id, str) or not _SAFE_SAMPLE_ID.fullmatch(sample_id):
        raise ValueError(f"unsafe redraw sample id in manifest: {sample_id!r}")
    if sample.get("split") != "validation":
        raise ValueError(f"redraw delivery sample must be a validation holdout: {sample_id}")
    frame_count = _positive_integer(
        sample.get("frameCount"),
        f"redraw sample {sample_id} frameCount",
    )
    canvas_width, canvas_height = _positive_size(
        value.get("nativeCanvas"),
        "redraw manifest nativeCanvas",
    )
    board = value.get("board")
    if not isinstance(board, dict):
        raise ValueError("redraw manifest must contain board metadata")
    columns = _positive_integer(board.get("columns"), "redraw board columns")
    rows = _positive_integer(board.get("rows"), "redraw board rows")
    native_width, native_height = _positive_size(
        board.get("nativeSize"),
        "redraw board nativeSize",
    )
    training_width, training_height = _positive_size(
        board.get("trainingSize"),
        "redraw board trainingSize",
    )
    capacity = columns * rows - 1
    if frame_count > capacity:
        raise ValueError(
            f"redraw sample {sample_id} frameCount={frame_count} exceeds "
            f"{columns}x{rows} board capacity={capacity} after identity cell 0"
        )
    if native_width % columns or native_height % rows:
        raise ValueError("native redraw board is not divisible by its grid")
    if training_width % columns or training_height % rows:
        raise ValueError("training redraw board is not divisible by its grid")
    if training_width % native_width or training_height % native_height:
        raise ValueError("training redraw board is not an integer nearest-neighbor scale")
    scale_x = training_width // native_width
    scale_y = training_height // native_height
    if scale_x != scale_y:
        raise ValueError("redraw board uses different horizontal and vertical scales")

    native_cell_width = native_width // columns
    native_cell_height = native_height // rows
    training_cell_width = training_width // columns
    training_cell_height = training_height // rows
    if (
        training_cell_width != native_cell_width * scale_x
        or training_cell_height != native_cell_height * scale_y
    ):
        raise ValueError("redraw board cell scale does not match board scale")
    declared_cell_size = board.get("cellSize")
    if declared_cell_size is not None:
        declared = _positive_integer(declared_cell_size, "redraw board cellSize")
        if training_cell_width != training_cell_height or declared != training_cell_width:
            raise ValueError(
                "redraw board cellSize does not match the square training grid cell"
            )
    if canvas_width > native_cell_width or canvas_height > native_cell_height:
        raise ValueError("nativeCanvas does not fit inside one redraw board cell")
    return {
        "sampleId": sample_id,
        "frameCount": frame_count,
        "canvas": [canvas_width, canvas_height],
        "columns": columns,
        "rows": rows,
        "nativeSize": [native_width, native_height],
        "trainingSize": [training_width, training_height],
        "nativeCell": [native_cell_width, native_cell_height],
        "trainingCell": [training_cell_width, training_cell_height],
        "scale": scale_x,
    }


def _failed_board_report(sample_id: str, generated: Path, blocker: str) -> dict[str, Any]:
    return {
        "ok": False,
        "passed": False,
        "sampleId": sample_id,
        "paths": {"generated": str(generated)},
        "blockers": [blocker],
    }


def _generated_board_readable(path: Path) -> str | None:
    try:
        with Image.open(path) as opened:
            opened.verify()
    except (OSError, UnidentifiedImageError) as exc:
        return f"generated holdout board is not a readable image: {exc}"
    return None


def evaluate_redraw_holdout_batch(
    manifest_path: str | Path,
    generated_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate every validation sample from ``<sample-id>.png`` files.

    A missing, symbolic-link, unreadable, or quality-rejected generated board is
    reported against its holdout instead of aborting the remaining batch. Invalid
    dataset contracts still raise before any generated board is evaluated.
    """

    manifest, value = _manifest(manifest_path)
    generated_raw = Path(generated_dir).expanduser()
    if generated_raw.is_symlink():
        raise ValueError(f"generated board directory must not be a symbolic link: {generated_raw}")
    generated_root = generated_raw.resolve()
    if not generated_root.is_dir():
        raise FileNotFoundError(f"generated board directory not found: {generated_root}")

    validation_samples = sorted(
        (
            sample
            for sample in value["samples"]
            if isinstance(sample, dict) and sample.get("split") == "validation"
        ),
        key=lambda sample: str(sample.get("id")),
    )
    if not validation_samples:
        raise ValueError("redraw manifest contains no validation holdouts")
    contracts = [_delivery_contract(value, sample) for sample in validation_samples]
    validation_ids = [contract["sampleId"] for contract in contracts]
    if len(validation_ids) != len(set(validation_ids)):
        raise ValueError("redraw validation sample ids must be unique")

    reports: list[dict[str, Any]] = []
    passed = 0
    for sample in validation_samples:
        sample_id = str(sample["id"])
        raw_generated = generated_root / f"{sample_id}.png"
        if raw_generated.is_symlink():
            reports.append(
                _failed_board_report(
                    sample_id,
                    raw_generated,
                    "generated holdout board must not be a symbolic link",
                )
            )
            continue
        generated = safe_output_child(
            generated_root,
            f"{sample_id}.png",
            label="generated holdout board",
        )
        if not generated.is_file():
            reports.append(
                _failed_board_report(
                    sample_id,
                    generated,
                    "generated holdout board is missing",
                )
            )
            continue
        unreadable = _generated_board_readable(generated)
        if unreadable:
            reports.append(_failed_board_report(sample_id, generated, unreadable))
            continue
        try:
            report = evaluate_redraw_sample(manifest, sample_id, generated)
        except FileNotFoundError:
            # The producer may still be replacing files while QC starts. Report
            # that board as unavailable without losing results for other holdouts.
            report = _failed_board_report(
                sample_id,
                generated,
                "generated holdout board became unavailable during evaluation",
            )
        reports.append(report)
        passed += int(bool(report.get("ok")))
    return {
        "ok": passed == len(validation_samples),
        "manifest": str(manifest),
        "generatedDirectory": str(generated_root),
        "summary": {
            "passed": passed,
            "required": len(validation_samples),
            "failed": len(validation_samples) - passed,
        },
        "reports": reports,
    }


def _validate_output_destination(path: str | Path) -> Path:
    """Validate an export destination without changing a previous good output."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"redraw frame export is a symbolic link: {candidate}")
    output = candidate.resolve()
    if output.exists() and not output.is_dir():
        raise ValueError(f"redraw frame export is not a directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        return output

    entries = list(output.iterdir())
    for descendant in output.rglob("*"):
        if descendant.is_symlink():
            raise ValueError(f"redraw frame export contains a symbolic link: {descendant}")
    if not entries:
        return output
    marker = output / _OUTPUT_MARKER
    if marker.is_symlink() or not marker.is_file():
        raise ValueError(f"redraw frame export is non-empty and is not AssetForge-owned: {output}")
    try:
        marker_value = strict_json_loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"redraw frame export has an unreadable ownership marker: {marker}") from exc
    if marker_value != _OUTPUT_MARKER_CONTENT:
        raise ValueError(f"redraw frame export has an invalid ownership marker: {marker}")
    return output


def _publish_staged_output(stage: Path, output: Path) -> None:
    """Atomically replace an output tree, rolling back the previous tree on error."""

    backup: Path | None = None
    if output.exists():
        backup = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=output.parent)
        )
        backup.rmdir()
        os.replace(output, backup)
    try:
        os.replace(stage, output)
    except BaseException as exc:
        rollback_error: BaseException | None = None
        if backup is not None and backup.exists():
            try:
                os.replace(backup, output)
            except BaseException as rollback_exc:  # pragma: no cover - catastrophic filesystem failure
                rollback_error = rollback_exc
        if rollback_error is not None:
            raise RuntimeError(
                "redraw frame export failed and rollback also failed; "
                f"previous output is preserved at {backup}: {rollback_error}"
            ) from exc
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def _write_staged_frames(
    stage: Path,
    prepared: list[Image.Image],
    export_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], Path]:
    files: list[dict[str, Any]] = []
    for index, frame in enumerate(prepared):
        filename = f"frame_{index:02d}.png"
        path = safe_output_child(stage, filename, label="redraw frame")
        frame.save(path, format="PNG", compress_level=1)
        with Image.open(path) as opened:
            opened.load()
            if opened.format != "PNG" or opened.size != frame.size:
                raise RuntimeError(f"staged redraw frame verification failed: {path}")
        files.append(
            {
                "index": index,
                "path": filename,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bbox": list(alpha_bbox(frame, 0) or ()),
            }
        )
    export_manifest["frames"] = files
    manifest_output = safe_output_child(stage, "frames.json", label="redraw frame manifest")
    manifest_output.write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Verify that the descriptor is self-contained before publishing the tree.
    descriptor = strict_json_loads(manifest_output.read_text(encoding="utf-8"))
    for item in descriptor["frames"]:
        relative = Path(item["path"])
        if relative.is_absolute() or len(relative.parts) != 1:
            raise RuntimeError("redraw frame manifest contains a non-portable frame path")
        frame_path = safe_output_child(stage, *relative.parts, label="redraw frame manifest path")
        if hashlib.sha256(frame_path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"redraw frame manifest hash verification failed: {frame_path}")
    return files, manifest_output


def export_redraw_board_frames(
    manifest_path: str | Path,
    sample_id: str,
    generated_path: str | Path,
    output_dir: str | Path,
    *,
    background_tolerance: int = 42,
) -> dict[str, Any]:
    """Split only a passing held-out board into native transparent PNG frames."""

    if not isinstance(background_tolerance, int) or isinstance(background_tolerance, bool):
        raise TypeError("background_tolerance must be an integer")
    if not 0 <= background_tolerance <= 765:
        raise ValueError("background_tolerance must be between 0 and 765")

    manifest, value = _manifest(manifest_path)
    sample = _sample(value, sample_id)
    contract = _delivery_contract(value, sample)
    generated_raw = Path(generated_path).expanduser()
    if generated_raw.is_symlink():
        raise ValueError(f"generated redraw board must not be a symbolic link: {generated_raw}")
    generated = generated_raw.resolve()
    if not generated.is_file():
        raise FileNotFoundError(f"generated redraw board not found: {generated}")
    source_bytes = generated.read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()

    quality = evaluate_redraw_sample(manifest, sample_id, generated)
    if not quality["ok"]:
        raise ValueError("redraw board failed quality gate: " + "; ".join(quality["blockers"]))
    if hashlib.sha256(generated.read_bytes()).hexdigest() != source_digest:
        raise RuntimeError("generated redraw board changed during quality evaluation")

    training_width, training_height = contract["trainingSize"]
    training_cell_width, training_cell_height = contract["trainingCell"]
    native_cell_width, native_cell_height = contract["nativeCell"]
    canvas_width, canvas_height = contract["canvas"]
    columns = contract["columns"]
    prepared: list[Image.Image] = []
    try:
        with Image.open(io.BytesIO(source_bytes)) as opened:
            board_image = opened.convert("RGB")
            board_image.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"generated redraw board is not a readable image: {generated}") from exc
    if board_image.size != (training_width, training_height):
        raise ValueError("generated redraw board dimensions changed after quality evaluation")
    for board_index in range(1, contract["frameCount"] + 1):
        row, column = divmod(board_index, columns)
        cell = board_image.crop(
            (
                column * training_cell_width,
                row * training_cell_height,
                (column + 1) * training_cell_width,
                (row + 1) * training_cell_height,
            )
        )
        cell = cell.resize(
            (native_cell_width, native_cell_height),
            Image.Resampling.NEAREST,
        )
        left = (native_cell_width - canvas_width) // 2
        top = (native_cell_height - canvas_height) // 2
        frame = cell.crop((left, top, left + canvas_width, top + canvas_height))
        frame = remove_corner_background(frame, background_tolerance)
        if frame.size != (canvas_width, canvas_height):
            raise RuntimeError(f"redraw frame cell {board_index} has an invalid canvas")
        if alpha_bbox(frame, 0) is None:
            raise ValueError(
                f"redraw frame cell {board_index} became empty after background removal"
            )
        prepared.append(frame)
    if len(prepared) != contract["frameCount"]:
        raise RuntimeError(
            f"redraw frame extraction produced {len(prepared)} frames; "
            f"expected {contract['frameCount']}"
        )

    output = _validate_output_destination(output_dir)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    published = False
    try:
        reset_output_directory(stage, label="staged redraw frame export")
        export_manifest = {
            "schemaVersion": 1,
            "sampleId": sample_id,
            "character": sample.get("character"),
            "direction": sample.get("direction"),
            "clip": sample.get("clip"),
            "loop": sample.get("loop"),
            "canvas": contract["canvas"],
            "frameCount": contract["frameCount"],
            "sourceDataset": manifest.name,
            "sourceDatasetSha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "sourceBoard": generated.name,
            "sourceBoardSha256": source_digest,
            "qualityPassed": True,
            "frames": [],
        }
        files, _ = _write_staged_frames(stage, prepared, export_manifest)
        _publish_staged_output(stage, output)
        published = True
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)

    manifest_output = safe_output_child(output, "frames.json", label="redraw frame manifest")
    frame_paths = [
        str(safe_output_child(output, item["path"], label="redraw frame"))
        for item in files
    ]
    return {
        "ok": True,
        "output": str(output),
        "manifest": str(manifest_output),
        "sampleId": sample_id,
        "frameCount": len(files),
        "canvas": contract["canvas"],
        "frames": frame_paths,
    }


__all__ = ["evaluate_redraw_holdout_batch", "export_redraw_board_frames"]
