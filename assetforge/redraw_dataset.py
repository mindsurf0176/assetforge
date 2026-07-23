from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from PIL import Image, ImageChops, ImageFilter

from .json_utils import strict_json_loads
from .path_safety import reset_output_directory, safe_output_child


_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_NUMBER = re.compile(r"(\d+)")
_OUTPUT_MARKER = ".assetforge-output.json"
_OUTPUT_MARKER_CONTENT = {"owner": "assetforge", "schemaVersion": 1}
_MFLUX_EDIT_PROMPTS = (
    "Redraw every pose-guide cell as the same pixel-art creature shown in cell 0; preserve the grid, palette, scale, and blank cells.",
    "Use cell 0 as the exact identity reference and replace each pose guide with that creature in the indicated pose; keep the spritesheet layout unchanged.",
    "Complete this animation board by rendering the reference creature in every guided pose while preserving its pixel style, colors, proportions, and outline.",
    "Turn the blue pose guides into finished frames of the creature from cell 0; retain the original grid geometry and leave unused cells blank.",
    "Create a coherent pixel-art animation sheet from this identity-and-pose board, matching the first cell's creature exactly in all completed cells.",
    "Render the same creature from cell 0 over every pose silhouette, with consistent anatomy, palette, pixel density, and spritesheet alignment.",
    "Finish each guided animation frame using cell 0 for character identity; do not move, crop, add, or remove any spritesheet cells.",
    "Replace only the pose guides with polished pixel-art frames of the reference creature, preserving the board background, layout, and empty cells.",
)


def _schema() -> dict[str, Any]:
    path = Path(__file__).with_name("schemas") / "redraw-dataset-spec.schema.json"
    return strict_json_loads(path.read_text(encoding="utf-8"))


def _load_spec(path: str | Path) -> tuple[Path, dict[str, Any]]:
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(f"redraw dataset spec not found: {spec_path}")
    data = strict_json_loads(spec_path.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(_schema()).iter_errors(data), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.path) or "$"
        raise ValueError(f"invalid redraw dataset spec at {location}: {first.message}")
    character_ids = [entry["id"] for entry in data["characters"]]
    if len(character_ids) != len(set(character_ids)):
        raise ValueError("redraw dataset character ids must be unique")
    unknown = sorted(set(data.get("validationCharacters", [])) - set(character_ids))
    if unknown:
        raise ValueError(f"validationCharacters contains unknown ids: {', '.join(unknown)}")
    validation = set(data["validationCharacters"])
    if validation == set(character_ids):
        raise ValueError("validationCharacters must leave at least one character in the train split")
    board = data["board"]
    native_width, native_height = (int(value) for value in data["nativeCanvas"])
    cell_size = int(board["cellSize"])
    if cell_size < native_width or cell_size < native_height:
        raise ValueError("board.cellSize must contain the complete nativeCanvas")
    training_width = cell_size * int(board["columns"]) * int(board["trainingScale"])
    training_height = cell_size * int(board["rows"]) * int(board["trainingScale"])
    if (
        training_width < 64
        or training_height < 64
        or training_width > 4096
        or training_height > 4096
        or training_width % 16
        or training_height % 16
    ):
        raise ValueError(
            "scaled redraw board dimensions must each be 64..4096 pixels and divisible by 16"
        )
    capacity = int(data["board"]["columns"]) * int(data["board"]["rows"]) - 1
    for character in data["characters"]:
        clip_ids = [clip["id"] for clip in character["clips"]]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError(f"{character['id']} clip ids must be unique")
        for clip in character["clips"]:
            if int(clip["expectedFrames"]) > capacity:
                raise ValueError(
                    f"{character['id']}:{clip['id']} needs {clip['expectedFrames']} frame cells; "
                    f"board capacity is {capacity}"
                )
    return spec_path, data


def _natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in _NUMBER.split(path.name)]


def _resolve_root(spec_path: Path, value: str) -> Path:
    raw = Path(value).expanduser()
    root = raw if raw.is_absolute() else spec_path.parent / raw
    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"redraw source root not found: {resolved}")
    return resolved


def _resolve_source(root: Path, pattern: str, *, direction: str, clip: str) -> Path:
    rendered = pattern.format(direction=direction, clip=clip)
    candidate = (root / rendered).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"redraw source path escapes character root: {rendered}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"redraw source image not found: {candidate}")
    return candidate


def _resolve_frames(root: Path, pattern: str, *, direction: str, clip: str) -> list[Path]:
    rendered = pattern.format(direction=direction, clip=clip)
    pattern_path = root / rendered
    parent = pattern_path.parent.resolve()
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"redraw frame pattern escapes character root: {rendered}") from exc
    frames = sorted(parent.glob(pattern_path.name), key=_natural_key)
    for frame in frames:
        resolved = frame.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"redraw frame path escapes character root: {frame}") from exc
        if frame.is_symlink() or not frame.is_file():
            raise ValueError(f"redraw frame must be a regular file: {frame}")
    return frames


def _read_rgba(path: Path, expected_size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGBA")
    if image.size != expected_size:
        raise ValueError(f"redraw source must be {expected_size[0]}x{expected_size[1]}: {path} is {image.size}")
    return image


def _place(board: Image.Image, image: Image.Image, index: int, columns: int, cell_size: int) -> None:
    if image.width > cell_size or image.height > cell_size:
        raise ValueError(
            f"board cell {cell_size}x{cell_size} cannot contain image {image.width}x{image.height}"
        )
    column = index % columns
    row = index // columns
    x = column * cell_size + (cell_size - image.width) // 2
    y = row * cell_size + (cell_size - image.height) // 2
    board.alpha_composite(image, (x, y))


def pose_guide(image: Image.Image, alpha_threshold: int = 20) -> Image.Image:
    """Build a style-free silhouette/edge guide from one complete RGBA frame."""

    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A").point(lambda value: 255 if value > alpha_threshold else 0)
    outer = alpha.filter(ImageFilter.MaxFilter(3))
    inner = alpha.filter(ImageFilter.MinFilter(3))
    edge = ImageChops.difference(outer, inner)
    guide = Image.new("RGBA", rgba.size, (70, 104, 138, 0))
    guide.putalpha(alpha)
    edge_layer = Image.new("RGBA", rgba.size, (244, 238, 192, 0))
    edge_layer.putalpha(edge)
    guide.alpha_composite(edge_layer)
    return guide


def _save_board(image: Image.Image, path: Path, scale: int) -> str:
    rendered = image.convert("RGB")
    if scale != 1:
        rendered = rendered.resize(
            (rendered.width * scale, rendered.height * scale),
            Image.Resampling.NEAREST,
        )
    pixel_digest = hashlib.sha256(rendered.tobytes()).hexdigest()
    rendered.save(path, format="PNG", compress_level=1)
    return pixel_digest


def _atomic_copy(source: Path, destination: Path) -> None:
    """Copy one generated training artifact without exposing a partial file."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"MFLUX export destination already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"MFLUX export temporary path already exists: {temporary}")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(destination: Path, value: str) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"MFLUX export destination already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"MFLUX export temporary path already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as writer:
            writer.write(value)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_replaceable_output(path: str | Path) -> Path:
    """Validate an output root without mutating its last successful build."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"redraw dataset output is a symbolic link: {candidate}")
    resolved = candidate.resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"redraw dataset output is not a directory: {resolved}")
    if not resolved.exists():
        return resolved
    entries = list(resolved.iterdir())
    if not entries:
        return resolved
    for descendant in resolved.rglob("*"):
        if descendant.is_symlink():
            raise ValueError(
                f"redraw dataset output contains a symbolic link: {descendant}"
            )
    marker = resolved / _OUTPUT_MARKER
    if marker.is_symlink() or not marker.is_file():
        raise ValueError(
            "redraw dataset output is non-empty and is not marked as "
            f"AssetForge-owned: {resolved}"
        )
    try:
        marker_value = strict_json_loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"redraw dataset output has an unreadable AssetForge ownership marker: {marker}"
        ) from exc
    if marker_value != _OUTPUT_MARKER_CONTENT:
        raise ValueError(
            f"redraw dataset output has an invalid AssetForge ownership marker: {marker}"
        )
    return resolved


def _promote_staged_output(staging: Path, destination: Path) -> Path:
    """Atomically promote a complete sibling build and restore on swap failure."""

    backup = destination.with_name(f".{destination.name}.assetforge-backup")
    if backup.exists() or backup.is_symlink():
        raise FileExistsError(f"redraw dataset backup path already exists: {backup}")
    moved_previous = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved_previous = True
        os.replace(staging, destination)
    except BaseException:
        if moved_previous and not destination.exists() and backup.exists():
            os.replace(backup, destination)
        raise
    if moved_previous:
        shutil.rmtree(backup)
    return destination


def _export_mflux_split(
    output_root: Path,
    samples: list[dict[str, Any]],
    *,
    source_split: str,
    export_split: str,
) -> dict[str, Any]:
    export_dir = safe_output_child(output_root, "mflux", export_split, label="MFLUX export")
    export_dir.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, Any]] = []
    selected = sorted(
        (sample for sample in samples if sample["split"] == source_split),
        key=lambda sample: sample["id"],
    )
    number_width = max(4, len(str(len(selected))))
    for zero_index, sample in enumerate(selected):
        number = zero_index + 1
        prefix = f"{number:0{number_width}d}"
        prompt_index = zero_index % len(_MFLUX_EDIT_PROMPTS)
        input_relative = f"mflux/{export_split}/{prefix}_in.png"
        target_relative = f"mflux/{export_split}/{prefix}_out.png"
        prompt_relative = f"mflux/{export_split}/{prefix}_in.txt"
        input_path = safe_output_child(output_root, *input_relative.split("/"), label="MFLUX input")
        target_path = safe_output_child(output_root, *target_relative.split("/"), label="MFLUX target")
        prompt_path = safe_output_child(output_root, *prompt_relative.split("/"), label="MFLUX prompt")
        source_input = safe_output_child(output_root, *sample["input"].split("/"), label="redraw input")
        source_target = safe_output_child(output_root, *sample["target"].split("/"), label="redraw target")
        _atomic_copy(source_input, input_path)
        _atomic_copy(source_target, target_path)
        _atomic_write_text(prompt_path, _MFLUX_EDIT_PROMPTS[prompt_index] + "\n")
        entries.append(
            {
                "index": number,
                "sample": sample["id"],
                "input": input_relative,
                "target": target_relative,
                "prompt": prompt_relative,
                "promptIndex": prompt_index,
            }
        )
    return {
        "path": f"mflux/{export_split}",
        "sampleCount": len(entries),
        "entries": entries,
    }


def build_redraw_dataset(spec: str | Path, output: str | Path) -> dict[str, Any]:
    spec_path, data = _load_spec(spec)
    native_size = tuple(int(value) for value in data["nativeCanvas"])
    board_config = data["board"]
    columns = int(board_config["columns"])
    rows = int(board_config["rows"])
    cell_size = int(board_config["cellSize"])
    scale = int(board_config["trainingScale"])
    background = tuple(int(value) for value in board_config["background"]) + (255,)
    alpha_threshold = int(data.get("alphaThreshold", 20))
    board_size = (columns * cell_size, rows * cell_size)
    validation = set(data["validationCharacters"])
    prepared_samples: list[dict[str, Any]] = []
    sample_ids: set[str] = set()

    # Resolve, count, and decode every source before touching a previous valid
    # generated dataset. Source drift must fail without erasing the last build.
    for character in data["characters"]:
        character_id = character["id"]
        if not _SAFE_ID.fullmatch(character_id):
            raise ValueError(f"unsafe character id: {character_id!r}")
        root = _resolve_root(spec_path, character["root"])
        split = "validation" if character_id in validation else "train"
        for direction in character["directions"]:
            if not _SAFE_ID.fullmatch(direction):
                raise ValueError(f"unsafe redraw direction: {direction!r}")
            reference_path = _resolve_source(
                root,
                character["referencePattern"],
                direction=direction,
                clip="reference",
            )
            reference = _read_rgba(reference_path, native_size)
            for clip in character["clips"]:
                clip_id = clip["id"]
                frame_paths = _resolve_frames(
                    root,
                    clip["framePattern"],
                    direction=direction,
                    clip=clip_id,
                )
                expected = int(clip["expectedFrames"])
                if len(frame_paths) != expected:
                    raise ValueError(
                        f"{character_id}:{direction}:{clip_id} expected {expected} frames, "
                        f"found {len(frame_paths)}"
                    )
                frames = [_read_rgba(path, native_size) for path in frame_paths]
                sample_id = f"{character_id}__{direction}__{clip_id}"
                if sample_id in sample_ids:
                    raise ValueError(f"redraw dataset sample id collision: {sample_id}")
                sample_ids.add(sample_id)
                prepared_samples.append(
                    {
                        "id": sample_id,
                        "split": split,
                        "character": character_id,
                        "direction": direction,
                        "clip": clip_id,
                        "loop": bool(clip["loop"]),
                        "root": root,
                        "referencePath": reference_path,
                        "reference": reference,
                        "framePaths": frame_paths,
                        "frames": frames,
                    }
                )

    destination = _validate_replaceable_output(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.assetforge-staging-",
            dir=destination.parent,
        )
    ).resolve()
    output_root = reset_output_directory(staging, label="redraw dataset staging output")
    try:
        samples: list[dict[str, Any]] = []

        for prepared in prepared_samples:
            input_board = Image.new("RGBA", board_size, background)
            output_board = Image.new("RGBA", board_size, background)
            _place(input_board, prepared["reference"], 0, columns, cell_size)
            _place(output_board, prepared["reference"], 0, columns, cell_size)
            for index, frame in enumerate(prepared["frames"], start=1):
                _place(input_board, pose_guide(frame, alpha_threshold), index, columns, cell_size)
                _place(output_board, frame, index, columns, cell_size)

            sample_id = prepared["id"]
            split = prepared["split"]
            sample_dir = safe_output_child(
                output_root,
                "samples",
                split,
                sample_id,
                label="redraw sample",
            )
            sample_dir.mkdir(parents=True, exist_ok=False)
            input_path = sample_dir / "input.png"
            target_path = sample_dir / "target.png"
            input_digest = _save_board(input_board, input_path, scale)
            target_digest = _save_board(output_board, target_path, scale)
            root = prepared["root"]
            samples.append(
                {
                    "id": sample_id,
                    "split": split,
                    "character": prepared["character"],
                    "direction": prepared["direction"],
                    "clip": prepared["clip"],
                    "loop": prepared["loop"],
                    "frameCount": len(prepared["frames"]),
                    "input": str(input_path.relative_to(output_root)),
                    "target": str(target_path.relative_to(output_root)),
                    "inputPixelSha256": input_digest,
                    "targetPixelSha256": target_digest,
                    "source": {
                        "rootId": prepared["character"],
                        "reference": prepared["referencePath"].relative_to(root).as_posix(),
                        "frames": [
                            path.relative_to(root).as_posix()
                            for path in prepared["framePaths"]
                        ],
                    },
                }
            )

        mflux_train = _export_mflux_split(
            output_root,
            samples,
            source_split="train",
            export_split="train",
        )
        mflux_holdout = _export_mflux_split(
            output_root,
            samples,
            source_split="validation",
            export_split="holdout",
        )

        manifest = {
            "schemaVersion": 1,
            "id": data["id"],
            "purpose": "local-full-frame-redraw-training",
            "spec": {
                "name": spec_path.name,
                "sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            },
            "nativeCanvas": list(native_size),
            "board": {
                "nativeSize": list(board_size),
                "trainingSize": [board_size[0] * scale, board_size[1] * scale],
                "columns": columns,
                "rows": rows,
                "cellSize": cell_size * scale,
                "layout": (
                    "cell-0 identity reference; cells 1..N pose guides or complete target frames"
                ),
                "background": list(background[:3]),
            },
            "sampleCount": len(samples),
            "splits": {
                "train": sum(sample["split"] == "train" for sample in samples),
                "validation": sum(
                    sample["split"] == "validation" for sample in samples
                ),
            },
            "characters": [entry["id"] for entry in data["characters"]],
            "mflux": {
                "format": "paired-edit-lora-flat-v1",
                "promptCount": len(_MFLUX_EDIT_PROMPTS),
                "train": mflux_train,
                "holdout": mflux_holdout,
            },
            "samples": samples,
        }
        manifest_path = output_root / "dataset.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _promote_staged_output(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    manifest_path = destination / "dataset.json"
    return {
        "ok": True,
        "output": str(destination),
        "manifest": str(manifest_path),
        "sampleCount": manifest["sampleCount"],
        "splits": manifest["splits"],
        "trainingSize": manifest["board"]["trainingSize"],
        "mflux": {
            "train": mflux_train["sampleCount"],
            "holdout": mflux_holdout["sampleCount"],
            "promptCount": len(_MFLUX_EDIT_PROMPTS),
        },
    }


__all__ = ["build_redraw_dataset", "pose_guide"]
