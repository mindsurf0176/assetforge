from __future__ import annotations

import json
import math
import re
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from .path_safety import safe_output_child
from .profile import Profile


@dataclass
class FrameMetric:
    file: str
    canvas: list[int]
    bbox: list[int]
    contentSize: list[int]
    foot: list[int]
    opaquePixels: int
    colors: int


def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def frame_paths(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"frame directory not found: {root}")
    paths = [
        path
        for path in root.glob("*.png")
        if not path.name.startswith("_") and path.name not in {"contact.png", "preview.png"}
    ]
    paths.sort(key=natural_key)
    if not paths:
        raise ValueError(f"no PNG frames found in {root}")
    return paths


def select_requested_animation_paths(
    paths: Iterable[Path],
    animation: str,
    known_animations: Iterable[str],
) -> list[Path]:
    """Select one named clip while preserving generic provider frame names.

    Raw providers commonly emit neutral names such as ``frame_0.png`` or
    ``pose_0.png``. Those files belong to the animation requested by the
    ingest command. Conversely, silently treating ``attack_*.png`` as a walk
    clip is always a caller error, so a directory containing another known
    animation prefix must fail instead of falling back.
    """

    candidates = list(paths)
    matching = [
        path
        for path in candidates
        if path.stem == animation or path.stem.startswith(f"{animation}_")
    ]
    if matching:
        return matching

    known = tuple(known_animations)
    conflicting = [
        path
        for path in candidates
        if any(
            path.stem == name or path.stem.startswith(f"{name}_")
            for name in known
            if name != animation
        )
    ]
    if conflicting:
        names = ", ".join(path.name for path in conflicting[:5])
        if len(conflicting) > 5:
            names += f", ... ({len(conflicting)} total)"
        raise ValueError(
            f"no PNG frames found for requested animation {animation!r}; "
            f"input contains other known animation frames: {names}"
        )
    return candidates


def alpha_bbox(image: Image.Image, min_alpha: int) -> tuple[int, int, int, int] | None:
    alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
    ys, xs = np.nonzero(alpha > min_alpha)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def remove_corner_background(image: Image.Image, tolerance: int) -> Image.Image:
    rgba = np.asarray(image.convert("RGBA")).copy()
    height, width = rgba.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("image must have a positive width and height")
    if (rgba[:, :, 3] < 250).any():
        return Image.fromarray(rgba)
    near_y, near_x = min(1, height - 1), min(1, width - 1)
    far_y, far_x = max(0, height - 2), max(0, width - 2)
    corners = np.array(
        [
            rgba[near_y, near_x, :3],
            rgba[near_y, far_x, :3],
            rgba[far_y, near_x, :3],
            rgba[far_y, far_x, :3],
        ],
        dtype=np.int16,
    )
    background = np.median(corners, axis=0)
    distance = np.abs(rgba[:, :, :3].astype(np.int16) - background).sum(axis=2)
    candidate = distance <= tolerance
    connected = np.zeros(candidate.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    height, width = candidate.shape

    def seed(y: int, x: int) -> None:
        if candidate[y, x] and not connected[y, x]:
            connected[y, x] = True
            queue.append((y, x))

    for x in range(width):
        seed(0, x)
        seed(height - 1, x)
    for y in range(1, height - 1):
        seed(y, 0)
        seed(y, width - 1)

    while queue:
        y, x = queue.popleft()
        for next_y, next_x in (
            (y - 1, x),
            (y + 1, x),
            (y, x - 1),
            (y, x + 1),
            (y - 1, x - 1),
            (y - 1, x + 1),
            (y + 1, x - 1),
            (y + 1, x + 1),
        ):
            if (
                0 <= next_y < height
                and 0 <= next_x < width
                and candidate[next_y, next_x]
                and not connected[next_y, next_x]
            ):
                connected[next_y, next_x] = True
                queue.append((next_y, next_x))

    rgba[connected, 3] = 0
    return Image.fromarray(rgba)


def repair_small_enclosed_transparent_components(
    image: Image.Image,
    max_component_pixels: int,
    min_alpha: int,
) -> Image.Image:
    """Fill only tiny transparent islands that are enclosed by opaque sprite pixels.

    Eight-neighbour connectivity matches the validation gate: diagonal access to the
    canvas edge makes transparency exterior background, not a repair candidate. Each
    repair uses the most common exact RGBA value on the component boundary, so no new
    palette color is invented. Ties prefer the more opaque, lighter existing color.
    """

    rgba = np.asarray(image.convert("RGBA")).copy()
    if max_component_pixels <= 0:
        return Image.fromarray(rgba)

    transparent = rgba[:, :, 3] <= min_alpha
    if not transparent.any():
        return Image.fromarray(rgba)

    height, width = transparent.shape
    exterior = np.zeros((height, width), dtype=bool)
    pending: deque[tuple[int, int]] = deque()
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
    for start_y, start_x in zip(*np.nonzero(enclosed)):
        if visited[start_y, start_x]:
            continue

        component: list[tuple[int, int]] = []
        visited[start_y, start_x] = True
        pending.append((int(start_y), int(start_x)))
        while pending:
            y, x = pending.popleft()
            component.append((y, x))
            for dy, dx in neighbours:
                next_y, next_x = y + dy, x + dx
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and enclosed[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    pending.append((next_y, next_x))

        if len(component) > max_component_pixels:
            continue

        boundary: set[tuple[int, int]] = set()
        for y, x in component:
            for dy, dx in neighbours:
                next_y, next_x = y + dy, x + dx
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and rgba[next_y, next_x, 3] > min_alpha
                ):
                    boundary.add((next_y, next_x))
        if not boundary:
            continue

        counts: dict[tuple[int, int, int, int], int] = {}
        for y, x in sorted(boundary):
            color = tuple(int(value) for value in rgba[y, x])
            counts[color] = counts.get(color, 0) + 1

        def color_rank(color: tuple[int, int, int, int]) -> tuple[int, int, int, tuple[int, int, int, int]]:
            red, green, blue, alpha = color
            luma = 299 * red + 587 * green + 114 * blue
            return counts[color], alpha, luma, color

        material = max(counts, key=color_rank)
        for y, x in component:
            rgba[y, x] = material

    return Image.fromarray(rgba)


def _opaque_rgb(images: Iterable[Image.Image], min_alpha: int, max_samples: int = 200_000) -> np.ndarray:
    chunks = []
    for image in images:
        rgba = np.asarray(image.convert("RGBA"))
        pixels = rgba[rgba[:, :, 3] > min_alpha, :3]
        if len(pixels):
            chunks.append(pixels)
    if not chunks:
        return np.empty((0, 3), dtype=np.uint8)
    pixels = np.concatenate(chunks, axis=0)
    if len(pixels) > max_samples:
        step = max(1, math.ceil(len(pixels) / max_samples))
        pixels = pixels[::step][:max_samples]
    return pixels


def build_shared_palette(images: list[Image.Image], colors: int, min_alpha: int) -> Image.Image | None:
    pixels = _opaque_rgb(images, min_alpha)
    if not len(pixels):
        return None
    if len(np.unique(pixels, axis=0)) <= colors:
        return None
    strip = Image.fromarray(pixels.reshape(1, len(pixels), 3))
    return strip.quantize(colors=max(2, min(colors, 256)), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def apply_palette(image: Image.Image, palette: Image.Image | None) -> Image.Image:
    if palette is None:
        return image.convert("RGBA")
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    rgb = rgba.convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE).convert("RGB")
    rgb.putalpha(alpha)
    return rgb


def _resampling(name: str) -> Image.Resampling:
    if name == "nearest":
        return Image.Resampling.NEAREST
    if name == "box":
        return Image.Resampling.BOX
    return Image.Resampling.LANCZOS


def _resize_to_fit(
    image: Image.Image,
    max_size: tuple[int, int],
    filtering: str,
    allow_upscale: bool,
    downscale_filtering: str = "lanczos",
) -> Image.Image:
    width, height = image.size
    scale = min(max_size[0] / width, max_size[1] / height)
    if scale >= 1 and not allow_upscale:
        return image
    if abs(scale - 1) < 1e-9:
        return image
    resample = _resampling(downscale_filtering if scale < 1 else filtering)
    return image.resize((max(1, round(width * scale)), max(1, round(height * scale))), resample)


def _metric(path: Path, image: Image.Image, min_alpha: int) -> FrameMetric:
    box = alpha_bbox(image, min_alpha)
    if box is None:
        raise ValueError(f"frame has no foreground pixels: {path}")
    rgba = np.asarray(image.convert("RGBA"))
    opaque = rgba[:, :, 3] > min_alpha
    colors = len(np.unique(rgba[opaque, :3], axis=0)) if opaque.any() else 0
    return FrameMetric(
        file=path.name,
        canvas=[image.width, image.height],
        bbox=list(box),
        contentSize=[box[2] - box[0], box[3] - box[1]],
        foot=[(box[0] + box[2] - 1) // 2, box[3] - 1],
        opaquePixels=int(opaque.sum()),
        colors=int(colors),
    )


def make_contact_sheet(paths: list[Path], output: Path, scale: int = 4) -> Path:
    images = [Image.open(path).convert("RGBA") for path in paths]
    tile_w = max(image.width for image in images)
    tile_h = max(image.height for image in images)
    columns = min(6, len(images))
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGBA", (tile_w * columns, tile_h * rows), (27, 32, 39, 255))
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        x = (index % columns) * tile_w
        y = (index // columns) * tile_h
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=(62, 72, 83, 255))
        sheet.alpha_composite(image, (x + (tile_w - image.width) // 2, y + tile_h - image.height))
    if scale > 1:
        sheet = sheet.resize((sheet.width * scale, sheet.height * scale), Image.Resampling.NEAREST)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.convert("RGB").save(output)
    return output


def ingest_frames(
    profile: Profile,
    input_dir: str | Path,
    output_dir: str | Path,
    tier_name: str,
    animation: str,
    direction: str,
    *,
    placement_mode: str = "per-frame-anchor",
    palette_override: Image.Image | None = None,
    source_anchor: tuple[int, int] | list[int] | None = None,
    source_bounds: tuple[int, int, int, int] | list[int] | None = None,
) -> dict[str, Any]:
    tier = profile.tier(tier_name)
    policy = tier.get("canvasPolicy", "fixed")
    preserve_placement = bool(tier.get("preservePlacement", False))
    if placement_mode not in {"per-frame-anchor", "shared-motion"}:
        raise ValueError(
            "placement_mode must be 'per-frame-anchor' or 'shared-motion'"
        )
    preserve_motion = placement_mode == "shared-motion"
    if preserve_placement and policy != "fixed":
        raise ValueError(
            f"tier {tier_name!r}: preservePlacement requires canvasPolicy='fixed'"
        )
    quality = profile.data["quality"]
    min_alpha = int(quality.get("alphaThreshold", 20))
    background_quality = quality.get("background", {})
    tolerance = int(background_quality.get("tolerance", 42))
    max_repair_pixels = int(background_quality.get("maxRepairableEnclosedComponentPixels", 0))
    if preserve_motion:
        # Spaces enclosed by articulated limbs, wings, or tails are legitimate
        # cutout geometry, not generator pinholes.
        max_repair_pixels = 0
    padding = int(tier.get("padding", 2))
    filtering = tier.get("filtering", "nearest")
    downscale_filtering = tier.get("downscaleFiltering", "lanczos")
    allow_upscale = bool(tier.get("allowUpscale", False))
    sources = select_requested_animation_paths(
        frame_paths(input_dir),
        animation,
        profile.data.get("animations", {}).keys(),
    )

    prepared: list[Image.Image] = []
    for path in sources:
        image = remove_corner_background(Image.open(path), tolerance)
        image = repair_small_enclosed_transparent_components(image, max_repair_pixels, min_alpha)
        box = alpha_bbox(image, min_alpha)
        if box is None:
            raise ValueError(f"frame has no foreground pixels after background removal: {path}")
        if preserve_placement:
            expected_canvas = tuple(map(int, tier["canvas"]))
            if image.size != expected_canvas:
                raise ValueError(
                    f"{path.name}: source canvas {list(image.size)} must equal configured canvas "
                    f"{list(expected_canvas)} when preservePlacement=true"
                )
            prepared.append(image)
        elif preserve_motion:
            prepared.append(image)
        else:
            prepared.append(image.crop(box))

    if preserve_motion:
        source_sizes = {image.size for image in prepared}
        if len(source_sizes) != 1:
            raise ValueError(
                "shared-motion placement requires every source frame to use one canvas"
            )

    content_max = tier.get("contentMax")
    if content_max and not preserve_placement and not preserve_motion:
        prepared = [
            _resize_to_fit(
                image,
                (int(content_max[0]), int(content_max[1])),
                filtering,
                False,
                downscale_filtering,
            )
            for image in prepared
        ]

    if policy == "fixed":
        canvas_w, canvas_h = map(int, tier["canvas"])
    else:
        canvas_w = max(image.width for image in prepared) + padding * 2
        canvas_h = max(image.height for image in prepared) + padding * 2
    anchor = tier.get("anchor", [canvas_w // 2, canvas_h - padding - 1])
    anchor_x, anchor_y = map(int, anchor)
    max_size = (max(1, canvas_w - padding * 2), max(1, anchor_y - padding + 1))

    shared_offset: tuple[int, int] | None = None
    fitted_source_anchor: tuple[int, int] | None = None
    if preserve_motion and source_anchor is not None:
        source_anchor_x, source_anchor_y = map(int, source_anchor)
        source_width, source_height = prepared[0].size
        if not (0 <= source_anchor_x < source_width and 0 <= source_anchor_y < source_height):
            raise ValueError(
                f"source anchor {list(source_anchor)} lies outside source canvas "
                f"{[source_width, source_height]}"
            )
        if source_bounds is None:
            bounds_left, bounds_top, bounds_right, bounds_bottom = 0, 0, source_width, source_height
        else:
            if len(source_bounds) != 4:
                raise ValueError("source_bounds must be [left, top, right, bottom]")
            bounds_left, bounds_top, bounds_right, bounds_bottom = map(int, source_bounds)
            if not (
                0 <= bounds_left < bounds_right <= source_width
                and 0 <= bounds_top < bounds_bottom <= source_height
            ):
                raise ValueError(
                    f"source bounds {list(source_bounds)} lie outside source canvas "
                    f"{[source_width, source_height]}"
                )
        capacities = []
        extents = (
            (source_anchor_x - bounds_left, anchor_x - padding),
            (bounds_right - 1 - source_anchor_x, canvas_w - padding - 1 - anchor_x),
            (source_anchor_y - bounds_top, anchor_y - padding),
            # A profile foot anchor is intentionally close to the bottom edge.
            # Bottom-side motion may use the remaining pixels below it; applying
            # top padding here can collapse an otherwise valid motion envelope.
            (bounds_bottom - 1 - source_anchor_y, canvas_h - 1 - anchor_y),
        )
        for extent, capacity in extents:
            if extent > 0:
                capacities.append(max(0.0, capacity) / extent)
        scale = min(capacities or [1.0])
        if not allow_upscale:
            scale = min(scale, 1.0)
        content_width = bounds_right - bounds_left
        content_height = bounds_bottom - bounds_top
        if scale <= 0 or min(content_width * scale, content_height * scale) < 4:
            raise ValueError(
                "shared motion envelope cannot fit the profile canvas without "
                "collapsing below 4 pixels"
            )
        target_size = (
            max(1, round(source_width * scale)),
            max(1, round(source_height * scale)),
        )
        sampling = _resampling(downscale_filtering if scale < 1 else filtering)
        fitted = [image.resize(target_size, sampling) for image in prepared]
        fitted_source_anchor = (
            round(source_anchor_x * scale),
            round(source_anchor_y * scale),
        )
        shared_offset = (
            anchor_x - fitted_source_anchor[0],
            anchor_y - fitted_source_anchor[1],
        )
    else:
        fitted = (
            prepared
            if preserve_placement
            else [
                _resize_to_fit(image, max_size, filtering, allow_upscale, downscale_filtering)
                for image in prepared
            ]
        )
    palette_cfg = quality.get("palette", {})
    palette = palette_override
    if palette is None and palette_cfg.get("lockAcrossClip", True):
        palette = build_shared_palette(fitted, int(palette_cfg.get("maxColors", 32)), min_alpha)

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("*.png"):
        if old.name.startswith(f"{animation}_") or old.name == "_contact.png":
            old.unlink()

    digits = max(2, len(str(max(0, len(fitted) - 1))))
    paths: list[Path] = []
    metrics: list[FrameMetric] = []
    for index, image in enumerate(fitted):
        image = repair_small_enclosed_transparent_components(
            image,
            max_repair_pixels,
            min_alpha,
        )
        image = apply_palette(image, palette)
        if preserve_placement:
            canvas = image
        else:
            canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            if shared_offset is not None:
                x, y = shared_offset
            else:
                x = anchor_x - image.width // 2
                y = anchor_y - image.height + 1
            canvas.alpha_composite(image, (x, y))
        path = safe_output_child(
            output,
            f"{animation}_{index:0{digits}d}.png",
            label="normalized frame",
        )
        canvas.save(path)
        paths.append(path)
        metrics.append(_metric(path, canvas, min_alpha))

    contact = make_contact_sheet(
        paths,
        safe_output_child(output, "_contact.png", label="normalized contact sheet"),
        scale=int(tier.get("previewScale", 4)),
    )
    manifest = {
        "schemaVersion": 1,
        "profile": profile.id,
        "profileFingerprint": profile.fingerprint,
        "kind": profile.kind,
        "tier": tier_name,
        "animation": animation,
        "direction": direction,
        "canvasPolicy": policy,
        "preservePlacement": preserve_placement,
        "placementMode": placement_mode,
        "sourceAnchor": list(source_anchor) if source_anchor is not None else None,
        "sourceBounds": list(source_bounds) if source_bounds is not None else None,
        "fittedSourceAnchor": list(fitted_source_anchor) if fitted_source_anchor is not None else None,
        "canvas": [canvas_w, canvas_h],
        "anchor": [anchor_x, anchor_y],
        "source": str(Path(input_dir).expanduser().resolve()),
        "output": str(output),
        "frames": [asdict(metric) for metric in metrics],
        "contactSheet": str(contact),
    }
    safe_output_child(output, "manifest.json", label="normalized manifest").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
