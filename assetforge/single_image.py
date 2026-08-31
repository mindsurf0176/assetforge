"""One-image animation orchestration and redraw-board preparation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

from .frames import (
    apply_palette,
    build_shared_palette,
    make_contact_sheet,
    remove_corner_background,
    remove_neutral_foreground_fringe,
    remove_neutral_edge_halo,
)
from .local_animation import run_local_animation
from .mflux_backend import paired_board_edit_plan, run_mflux_plan
from .path_safety import reset_output_directory, safe_output_child
from .redraw_dataset import pose_guide


def _place(board: Image.Image, image: Image.Image, index: int, columns: int, cell: int) -> None:
    x = (index % columns) * cell + (cell - image.width) // 2
    y = (index // columns) * cell + (cell - image.height) // 2
    board.alpha_composite(image, (x, y))


def _clip_prompt(clip: str) -> str:
    motion = {
        "idle": "subtle breathing and cloth sway, combat-ready stance",
        "walk": "clear readable walking cycle, stable weapon, alternating legs",
        "aim": "steady aiming pose with minimal breathing sway, weapon rigid",
        "attack": "anticipation, decisive attack, readable impact effect, recoil, recovery",
        "hit": "brief readable hit flinch, preserve weapon and costume identity",
        "death": "stagger, collapse, final grounded pose, no extra body parts",
    }.get(clip, f"clear readable {clip} animation pose")
    return (
        "Use cell 0 as the exact character identity reference. Replace every pose guide "
        f"with a complete full-body pixel-art frame showing {motion}. Preserve face, "
        "hair, costume, equipment, proportions, outline weight, palette, side-view angle, "
        "and grid alignment. Keep effects readable and attached to the action. Do not add "
        "characters, scenery, text, labels, or camera movement. Render every finished "
        "sprite on a genuinely transparent RGBA background: no white matte, gray/black "
        "backdrop, checkerboard, glow fringe, or semi-transparent light halo."
    )


def _normalize_redrawn_frames(
    generated: dict[str, list[Path]],
    output: Path,
    *,
    padding: int = 3,
    max_colors: int = 191,
) -> dict[str, Any]:
    """Make generated full-frame results behave like a stable game sprite set."""

    decoded: dict[str, list[Image.Image]] = {}
    boxes: list[tuple[int, int, int, int]] = []
    for clip, paths in generated.items():
        decoded[clip] = []
        for path in paths:
            with Image.open(path) as opened:
                image = remove_neutral_foreground_fringe(
                    remove_neutral_edge_halo(remove_corner_background(opened, 42))
                )
            box = image.getchannel("A").getbbox()
            if box is None:
                raise ValueError(f"generated frame has no foreground: {path}")
            decoded[clip].append(image)
            boxes.append(box)
    if not boxes:
        raise ValueError("no generated frames to normalize")
    content_width = max(box[2] - box[0] for box in boxes)
    content_height = max(box[3] - box[1] for box in boxes)
    canvas = (content_width + padding * 2, content_height + padding * 2)
    fitted: list[Image.Image] = []
    for images in decoded.values():
        for image in images:
            box = image.getchannel("A").getbbox()
            assert box is not None
            cropped = image.crop(box)
            fitted.append(cropped)
    palette = build_shared_palette(fitted, max_colors, 20)
    normalized_root = safe_output_child(output, "normalized", label="normalized redraw output")
    reset_output_directory(normalized_root, label="normalized redraw output")
    report: dict[str, Any] = {"canvas": list(canvas), "padding": padding, "maxColors": max_colors, "clips": {}}
    cursor = 0
    all_paths: list[Path] = []
    for clip, images in decoded.items():
        clip_dir = safe_output_child(normalized_root, clip, label="normalized redraw clip")
        clip_dir.mkdir(parents=True, exist_ok=False)
        metrics: list[dict[str, Any]] = []
        paths: list[Path] = []
        for index, image in enumerate(images):
            box = image.getchannel("A").getbbox()
            assert box is not None
            cropped = apply_palette(image.crop(box), palette)
            frame = Image.new("RGBA", canvas, (0, 0, 0, 0))
            x = (canvas[0] - cropped.width) // 2
            y = canvas[1] - padding - cropped.height
            frame.alpha_composite(cropped, (x, y))
            path = safe_output_child(clip_dir, f"{clip}_{index:02d}.png", label="normalized redraw frame")
            frame.save(path)
            paths.append(path)
            all_paths.append(path)
            alpha_box = frame.getchannel("A").getbbox()
            assert alpha_box is not None
            colors = len(set(frame.getdata()))
            metrics.append({"file": str(path), "bbox": list(alpha_box), "colors": colors})
            cursor += 1
        contact = make_contact_sheet(paths, clip_dir / "_contact.png", scale=4)
        report["clips"][clip] = {"frames": metrics, "contactSheet": str(contact)}
    contact = make_contact_sheet(all_paths, normalized_root / "_contact.png", scale=4)
    report["contactSheet"] = str(contact)
    report["quality"] = "production-candidate"
    report_path = safe_output_child(normalized_root, "quality-report.json", label="redraw quality report")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    return report


def build_single_image_animation(
    *,
    reference: str | Path,
    output: str | Path,
    character: str,
    archetype: str,
    direction: str = "east",
    clips: list[str] | None = None,
    frame_overrides: dict[str, int] | None = None,
    height: int = 192,
    resample: str = "nearest",
    execute: bool = False,
    executable: str | Path | None = None,
    model: str = "flux2-klein-4b",
    base_model: str | None = None,
    model_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    lora: list[str | Path] | None = None,
    lora_scale: float = 1.0,
    seed: int = 42,
    steps: int | None = None,
    guidance: float = 1.0,
    quantize: int | None = None,
    low_ram: bool = True,
    mlx_cache_limit_gib: float | None = 2.5,
    minimum_free_gib: float = 6.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build local frames and identity-plus-pose boards from one image."""

    destination = Path(output).expanduser().resolve()
    if destination.is_symlink():
        raise ValueError(f"single-image output must not be a symbolic link: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"single-image output is not a directory: {destination}")
    work = destination / "cutout"
    result = run_local_animation(
        work_dir=work,
        character=character,
        direction=direction,
        clips=clips,
        frame_overrides=frame_overrides,
        reference=reference,
        archetype=archetype,
        height=height,
        resample=resample,
    )
    if not result.get("ok"):
        return result

    rig_path = Path(result["rig"])
    rig = json.loads(rig_path.read_text(encoding="utf-8"))
    canvas_width, canvas_height = map(int, rig["canvas"])
    cell = int(math.ceil(max(canvas_width, canvas_height) / 16) * 16)
    columns = 4
    boards = safe_output_child(destination, "redraw-boards", label="single-image redraw boards")
    reset_output_directory(boards, label="single-image redraw boards")
    board_entries: list[dict[str, Any]] = []
    reference_path = rig_path.parent / "reference-normalized.png"
    with Image.open(reference_path) as opened:
        identity = opened.convert("RGBA")

    for clip_name, clip_data in result["raw"]["clips"].items():
        frame_paths = [Path(value["file"]) for value in clip_data["frames"]]
        rows = math.ceil((len(frame_paths) + 1) / columns)
        board = Image.new("RGBA", (columns * cell, rows * cell), (35, 39, 47, 255))
        _place(board, identity, 0, columns, cell)
        for index, frame_path in enumerate(frame_paths, start=1):
            with Image.open(frame_path) as opened:
                frame = opened.convert("RGBA")
            _place(board, pose_guide(frame), index, columns, cell)
        board_path = safe_output_child(boards, f"{clip_name}.png", label="single-image redraw board")
        board.save(board_path, format="PNG", compress_level=1)
        board_entries.append({
            "clip": clip_name,
            "frameCount": len(frame_paths),
            "input": str(board_path),
            "layout": {"columns": columns, "rows": rows, "cellSize": cell},
            "identityCell": 0,
            "status": "ready-for-full-frame-redraw",
        })

    redraw_output = safe_output_child(destination, "redrawn", label="single-image redrawn output")
    redraw_output.mkdir(parents=True, exist_ok=True)
    inference: list[dict[str, Any]] = []
    generated_frames: dict[str, list[Path]] = {}
    for entry in board_entries:
        generated_board = safe_output_child(
            redraw_output, f"{entry['clip']}-board.png", label="single-image generated board"
        )
        if not execute:
            inference.append({
                "clip": entry["clip"],
                "board": entry["input"],
                "generatedBoard": str(generated_board),
                "ready": None,
                "blockers": [],
                "status": "planned",
            })
            continue
        plan = paired_board_edit_plan(
            entry["input"],
            generated_board,
            prompt=_clip_prompt(entry["clip"]),
            executable=executable,
            model=model,
            base_model=base_model,
            model_path=model_path,
            cache_dir=cache_dir,
            lora=lora,
            lora_scale=lora_scale,
            seed=seed,
            steps=steps,
            guidance=guidance,
            quantize=quantize,
            low_ram=low_ram,
            mlx_cache_limit_gib=mlx_cache_limit_gib,
            overwrite=overwrite,
            minimum_free_gib=minimum_free_gib,
        )
        item: dict[str, Any] = {
            "clip": entry["clip"],
            "board": entry["input"],
            "generatedBoard": str(generated_board),
            "ready": plan["ready"],
            "blockers": plan["blockers"],
            "status": "planned",
        }
        if execute and plan["ready"]:
            generated = run_mflux_plan(plan, execute=True)
            item.update(generated)
            item["status"] = "generated"
            frames_dir = safe_output_child(
                redraw_output, entry["clip"], label="single-image generated frames"
            )
            reset_output_directory(frames_dir, label="single-image generated frames")
            with Image.open(generated_board) as opened:
                board = opened.convert("RGBA")
            columns = int(entry["layout"]["columns"])
            cell_size = int(entry["layout"]["cellSize"])
            for frame_index in range(1, int(entry["frameCount"]) + 1):
                x = (frame_index % columns) * cell_size
                y = (frame_index // columns) * cell_size
                frame = board.crop((x, y, x + cell_size, y + cell_size))
                frame = remove_neutral_foreground_fringe(
                    remove_neutral_edge_halo(remove_corner_background(frame, 42))
                )
                frame.save(
                    safe_output_child(
                        frames_dir,
                        f"{entry['clip']}_{frame_index - 1:02d}.png",
                        label="single-image generated frame",
                    )
                )
            generated_frames[entry["clip"]] = sorted(frames_dir.glob("*.png"))
            item["frames"] = str(frames_dir)
        inference.append(item)

    normalization: dict[str, Any] | None = None
    if generated_frames:
        normalization = _normalize_redrawn_frames(generated_frames, redraw_output)

    manifest = {
        "schemaVersion": 1,
        "kind": "assetforge-single-image-animation",
        "quality": "production-candidate" if normalization else "coarse",
        "character": character,
        "direction": direction,
        "reference": str(Path(reference).expanduser().resolve()),
        "cutout": result,
        "redrawBoards": board_entries,
        "inference": inference,
        "normalization": normalization,
        "limitations": [
            "cutout frames preserve visible pixels but do not reconstruct occluded anatomy",
            "redraw boards require a full-frame generative backend for production-quality frames",
            "approve identity, silhouette, alignment, and loop continuity before export",
        ],
    }
    manifest_path = safe_output_child(destination, "single-image-manifest.json", label="single-image manifest")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    inference_ready = bool(execute and inference and all(item["ready"] for item in inference))
    return {
        "ok": inference_ready if execute else True,
        "mode": "single-image",
        "quality": "production-candidate" if normalization else "coarse",
        "output": str(destination),
        "cutout": str(work),
        "redrawBoards": str(boards),
        "manifest": str(manifest_path),
        "clips": board_entries,
        "inference": inference,
        "inferenceReady": inference_ready,
    }
