from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .clip_library import clips_for
from .frames import remove_corner_background
from .json_utils import strict_json_loads
from .path_safety import reset_output_directory, safe_output_child
from .rig_core import RigError, render_frame, validate_production_part_contract, write_rig


ARCHETYPES = ("biped-side", "quadruped-side", "winged-quadruped-side")


_BIPED_SLOTS: dict[str, tuple[str, int, tuple[float, float]]] = {
    "cape": ("cape", -1, (0.5, 0.06)),
    "thigh_b": ("hip_b", 0, (0.5, 0.06)),
    "shin_b": ("kn_b", 0, (0.5, 0.06)),
    "leg_b": ("hip_b", 0, (0.5, 0.06)),
    "legs": ("hip_c", 0, (0.5, 0.06)),
    "uarm_b": ("sh_b", 1, (0.5, 0.06)),
    "farm_b": ("el_b", 1, (0.5, 0.06)),
    "arm_b": ("sh_b", 1, (0.5, 0.06)),
    "hair_b": ("hair_b", 1, (0.5, 0.08)),
    "torso": ("spine", 2, (0.5, 0.62)),
    "head": ("neck", 3, (0.5, 0.82)),
    "thigh_f": ("hip_f", 4, (0.5, 0.06)),
    "shin_f": ("kn_f", 4, (0.5, 0.06)),
    "leg_f": ("hip_f", 4, (0.5, 0.06)),
    "uarm_f": ("sh_f", 5, (0.5, 0.06)),
    "farm_f": ("el_f", 5, (0.5, 0.06)),
    "arm_f": ("sh_f", 5, (0.5, 0.06)),
    "weapon": ("grip", 6, (0.25, 0.5)),
}


_QUADRUPED_SLOTS: dict[str, tuple[str, int, tuple[float, float]]] = {
    "tail": ("tail", -1, (0.88, 0.46)),
    "hindleg_b": ("hindleg_b", 0, (0.5, 0.07)),
    "foreleg_b": ("foreleg_b", 0, (0.5, 0.07)),
    "wing_b": ("wing_b", 1, (0.16, 0.72)),
    "body": ("body", 2, (0.5, 0.5)),
    "wing_f": ("wing_f", 3, (0.16, 0.72)),
    "hindleg_f": ("hindleg_f", 4, (0.5, 0.07)),
    "foreleg_f": ("foreleg_f", 4, (0.5, 0.07)),
    "head": ("head", 5, (0.15, 0.58)),
}


_BIPED_MIRRORS = (
    ("uarm_f", "uarm_b"),
    ("farm_f", "farm_b"),
    ("arm_f", "arm_b"),
    ("thigh_f", "thigh_b"),
    ("shin_f", "shin_b"),
    ("leg_f", "leg_b"),
)


_QUADRUPED_MIRRORS = (
    ("foreleg_f", "foreleg_b"),
    ("hindleg_f", "hindleg_b"),
    ("wing_f", "wing_b"),
)


def archetypes() -> tuple[str, ...]:
    return ARCHETYPES


def _clip_archetype(archetype: str) -> str:
    return "biped_side" if archetype == "biped-side" else "quadruped_side"


def _slot_contract(archetype: str) -> dict[str, tuple[str, int, tuple[float, float]]]:
    if archetype == "biped-side":
        return _BIPED_SLOTS
    if archetype in {"quadruped-side", "winged-quadruped-side"}:
        return _QUADRUPED_SLOTS
    raise RigError(f"unknown archetype {archetype!r}; available: {', '.join(ARCHETYPES)}")


def _canonical_slot(archetype: str, name: str) -> str:
    token = name.strip().lower().replace("-", "_")
    aliases = {
        "body": "torso",
        "front_arm": "arm_f",
        "back_arm": "arm_b",
        "front_leg": "leg_f",
        "back_leg": "leg_b",
    }
    if archetype == "biped-side":
        return aliases.get(token, token)
    quadruped_aliases = {
        "torso": "body",
        "front_leg_f": "foreleg_f",
        "front_leg_b": "foreleg_b",
        "back_leg_f": "hindleg_f",
        "back_leg_b": "hindleg_b",
        "rearleg_f": "hindleg_f",
        "rearleg_b": "hindleg_b",
        "wing": "wing_f",
        "leg_front_f": "foreleg_f",
        "leg_front_b": "foreleg_b",
        "leg_back_f": "hindleg_f",
        "leg_back_b": "hindleg_b",
    }
    return quadruped_aliases.get(token, token)


def _tight_part(path: Path, background_tolerance: int = 42) -> Image.Image:
    with Image.open(path) as opened:
        image = remove_corner_background(opened, background_tolerance)
    box = image.getchannel("A").getbbox()
    if box is None:
        raise RigError(f"part image has no foreground pixels: {path.name}")
    return image.crop(box)


def load_named_parts(
    directory: str | Path,
    archetype: str,
    *,
    background_tolerance: int = 42,
) -> dict[str, Image.Image]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"part directory not found: {root}")
    contract = _slot_contract(archetype)
    images: dict[str, Image.Image] = {}
    for path in sorted(root.glob("*.png")):
        slot = _canonical_slot(archetype, path.stem)
        if slot not in contract:
            raise RigError(
                f"unknown part filename {path.name!r} for {archetype}; "
                f"available slots: {', '.join(sorted(contract))}"
            )
        if slot in images:
            raise RigError(f"duplicate part slot {slot!r}")
        images[slot] = _tight_part(path, background_tolerance)
    if not images:
        raise RigError(f"no PNG parts found in {root}")
    return images


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 8-connected components with row runs and union-find."""

    height, width = mask.shape
    previous = np.zeros_like(mask)
    previous[:, 1:] = mask[:, :-1]
    following = np.zeros_like(mask)
    following[:, :-1] = mask[:, 1:]
    start_rows, start_columns = np.nonzero(mask & ~previous)
    end_rows, end_columns = np.nonzero(mask & ~following)
    run_count = len(start_rows)
    labels = np.zeros((height, width), np.int32)
    if not run_count:
        return labels, 0
    parents = list(range(run_count))

    def find(index: int) -> int:
        root = index
        while parents[root] != root:
            root = parents[root]
        while parents[index] != root:
            parents[index], index = root, parents[index]
        return root

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    row_first = np.searchsorted(start_rows, np.arange(height + 1))
    for row in range(1, height):
        upper, upper_end = int(row_first[row - 1]), int(row_first[row])
        lower, lower_end = int(row_first[row]), int(row_first[row + 1])
        while upper < upper_end and lower < lower_end:
            if end_columns[upper] + 1 < start_columns[lower]:
                upper += 1
            elif end_columns[lower] + 1 < start_columns[upper]:
                lower += 1
            else:
                union(upper, lower)
                if end_columns[upper] < end_columns[lower]:
                    upper += 1
                else:
                    lower += 1
    root_labels: dict[int, int] = {}
    for index in range(run_count):
        root = find(index)
        root_labels.setdefault(root, len(root_labels) + 1)
        labels[
            start_rows[index],
            start_columns[index] : end_columns[index] + 1,
        ] = root_labels[root]
    return labels, len(root_labels)


def _sheet_foreground(
    image: Image.Image,
    background_tolerance: int,
) -> tuple[np.ndarray, np.ndarray]:
    rgba = np.asarray(image.convert("RGBA")).copy()
    if rgba[:, :, 3].min() < 255:
        return rgba, rgba[:, :, 3] > 0
    cleaned = np.asarray(remove_corner_background(image, background_tolerance)).copy()
    return cleaned, cleaned[:, :, 3] > 0


def _draw_component_contact(entries: list[dict[str, Any]], output: Path) -> None:
    columns = max(1, math.ceil(math.sqrt(len(entries))))
    rows = math.ceil(len(entries) / columns)
    cell = 220
    sheet = Image.new("RGBA", (columns * cell, rows * cell), (35, 39, 47, 255))
    draw = ImageDraw.Draw(sheet)
    for index, entry in enumerate(entries):
        x = (index % columns) * cell
        y = (index // columns) * cell
        draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(83, 92, 105, 255))
        image = entry["image"]
        scale = min((cell - 24) / image.width, (cell - 40) / image.height, 4)
        resampling = Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.NEAREST
        preview = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            resampling,
        )
        sheet.alpha_composite(preview, (x + (cell - preview.width) // 2, y + 6 + (cell - 40 - preview.height) // 2))
        draw.rectangle((x + 4, y + 4, x + 78, y + 27), fill=(255, 211, 70, 255))
        draw.text((x + 9, y + 8), entry["id"], fill=(20, 20, 22, 255))
        draw.text((x + 6, y + cell - 18), f"{image.width}x{image.height}", fill=(230, 233, 238, 255))
    sheet.save(output)


def extract_part_sheet(
    sheet_path: str | Path,
    output_dir: str | Path,
    *,
    background_tolerance: int = 42,
    min_area_ratio: float = 0.0005,
) -> dict[str, Any]:
    source = Path(sheet_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"part sheet not found: {source}")
    if not 0 < min_area_ratio < 1:
        raise RigError("min_area_ratio must be between 0 and 1")
    with Image.open(source) as opened:
        image = opened.convert("RGBA")
    rgba, foreground = _sheet_foreground(image, background_tolerance)
    if not foreground.any():
        raise RigError("part sheet contains no foreground pixels")
    if float(foreground.mean()) > 0.9:
        raise RigError("part sheet background could not be separated; use a flat border-connected background")
    dilated = np.asarray(
        Image.fromarray(foreground.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))
    ) > 0
    labels, component_count = _label_mask(dilated)
    if not component_count:
        raise RigError("no disconnected parts found in the sheet")

    components: list[dict[str, Any]] = []
    minimum_area = image.width * image.height * min_area_ratio
    for component_id in range(1, component_count + 1):
        mask = foreground & (labels == component_id)
        ys, xs = np.nonzero(mask)
        if len(xs) < minimum_area:
            continue
        left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        crop = rgba[top:bottom, left:right].copy()
        crop_mask = mask[top:bottom, left:right]
        crop[:, :, 3] = np.where(crop_mask, crop[:, :, 3], 0)
        components.append(
            {
                "area": int(len(xs)),
                "bbox": [left, top, right - left, bottom - top],
                "image": Image.fromarray(crop),
            }
        )
    if not components:
        raise RigError("all detected sheet components were below the minimum area")
    components.sort(key=lambda item: (-item["area"], item["bbox"][1], item["bbox"][0]))

    output = reset_output_directory(output_dir, label="part-sheet extraction output")
    blobs = safe_output_child(output, "blobs", label="extracted parts directory")
    blobs.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for index, component in enumerate(components, start=1):
        identifier = f"blob_{index:02d}"
        target = safe_output_child(blobs, f"{identifier}.png", label="extracted part")
        component["image"].save(target)
        entries.append({"id": identifier, "image": component["image"]})
        metadata.append(
            {
                "id": identifier,
                "file": f"blobs/{identifier}.png",
                "bbox": component["bbox"],
                "area": component["area"],
                "size": list(component["image"].size),
            }
        )
    contact = safe_output_child(output, "contact-sheet.png", label="parts contact sheet")
    _draw_component_contact(entries, contact)
    result = {
        "schemaVersion": 1,
        "source": source.name,
        "canvas": list(image.size),
        "count": len(metadata),
        "backgroundTolerance": background_tolerance,
        "minAreaRatio": min_area_ratio,
        "components": metadata,
        "contactSheet": str(contact),
    }
    report = safe_output_child(output, "parts-manifest.json", label="parts manifest")
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["manifest"] = str(report)
    result["directory"] = str(blobs)
    return result


def _canonical_blob(value: str) -> str:
    match = re.fullmatch(r"(?i)blob[_-]?0*(\d+)", value.strip())
    return f"blob_{int(match.group(1)):02d}" if match else value.strip()


def load_sheet_parts(
    sheet_path: str | Path,
    mapping_path: str | Path,
    work_dir: str | Path,
    archetype: str,
    *,
    background_tolerance: int = 42,
    min_area_ratio: float = 0.0005,
) -> tuple[dict[str, Image.Image], dict[str, Any]]:
    extraction = extract_part_sheet(
        sheet_path,
        work_dir,
        background_tolerance=background_tolerance,
        min_area_ratio=min_area_ratio,
    )
    mapping_file = Path(mapping_path).expanduser().resolve()
    if not mapping_file.is_file():
        raise FileNotFoundError(f"part mapping not found: {mapping_file}")
    try:
        mapping = strict_json_loads(mapping_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        raise RigError(f"invalid mapping JSON: {exc}") from exc
    if not isinstance(mapping, dict) or not mapping:
        raise RigError("part mapping must be a non-empty JSON object")
    contract = _slot_contract(archetype)
    available = {entry["id"] for entry in extraction["components"]}
    parts: dict[str, Image.Image] = {}
    used_blobs: set[str] = set()
    for raw_blob, raw_slot in mapping.items():
        blob = _canonical_blob(str(raw_blob))
        if blob not in available:
            raise RigError(f"mapping references unknown component {raw_blob!r}")
        if blob in used_blobs:
            raise RigError(f"mapping aliases component {blob!r} more than once")
        used_blobs.add(blob)
        if not isinstance(raw_slot, str):
            raise RigError(f"mapping value for {raw_blob!r} must be a slot name")
        if raw_slot.strip().upper() == "IGNORE":
            continue
        slot = _canonical_slot(archetype, raw_slot)
        if slot not in contract:
            raise RigError(f"unknown slot {raw_slot!r} for {archetype}")
        if slot in parts:
            raise RigError(f"multiple sheet components map to slot {slot!r}")
        with Image.open(Path(extraction["directory"]) / f"{blob}.png") as image:
            parts[slot] = image.convert("RGBA").copy()
    unmapped = sorted(available - used_blobs)
    if unmapped:
        raise RigError(
            "mapping must assign every extracted component to a slot or IGNORE; "
            f"unmapped: {', '.join(unmapped)}"
        )
    if not parts:
        raise RigError("mapping did not select any usable parts")
    return parts, extraction


def _dark_mirror(image: Image.Image) -> Image.Image:
    mirrored = np.asarray(ImageOps.mirror(image), dtype=np.float32).copy()
    mirrored[:, :, :3] *= 0.85
    return Image.fromarray(np.clip(mirrored, 0, 255).astype(np.uint8))


def _add_mirrors(parts: dict[str, Image.Image], archetype: str, no_mirror: bool) -> list[str]:
    if no_mirror:
        return []
    generated: list[str] = []
    pairs = _BIPED_MIRRORS if archetype == "biped-side" else _QUADRUPED_MIRRORS
    for front, back in pairs:
        if front in parts and back not in parts:
            parts[back] = _dark_mirror(parts[front])
            generated.append(back)
        elif back in parts and front not in parts:
            parts[front] = _dark_mirror(parts[back])
            generated.append(front)
    return generated


def _scaled_parts(
    parts: dict[str, Image.Image],
    scale: float,
    resample: Image.Resampling,
) -> dict[str, Image.Image]:
    return {
        name: image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            resample,
        )
        for name, image in parts.items()
    }


def _first_height(parts: dict[str, Image.Image], *names: str) -> float:
    for name in names:
        if name in parts:
            return float(parts[name].height)
    return 0.0


def _build_biped_geometry(
    parts: dict[str, Image.Image],
    height: int,
    resample: Image.Resampling,
) -> tuple[dict[str, Image.Image], dict[str, Any], tuple[int, int]]:
    if "torso" not in parts or "head" not in parts:
        raise RigError("biped-side requires torso.png and head.png")
    if any(name in parts for name in ("thigh_f", "thigh_b", "shin_f", "shin_b")):
        leg_est = _first_height(parts, "thigh_f", "thigh_b") + _first_height(parts, "shin_f", "shin_b")
    elif any(name in parts for name in ("leg_f", "leg_b")):
        leg_est = _first_height(parts, "leg_f", "leg_b")
    else:
        leg_est = _first_height(parts, "legs")
    estimate = parts["head"].height + parts["torso"].height + leg_est
    if estimate <= 0:
        raise RigError("could not estimate biped height")
    scale = (height * 0.66) / estimate
    scaled = _scaled_parts(parts, scale, resample)
    width = int(round(height * 0.92 / 2) * 2)

    def part_height(*names: str, factor: float = 1.0) -> float:
        return _first_height(scaled, *names) * factor

    if any(name in scaled for name in ("thigh_f", "thigh_b", "shin_f", "shin_b")):
        leg_length = part_height("thigh_f", "thigh_b", factor=0.88) + part_height("shin_f", "shin_b", factor=0.88)
    elif any(name in scaled for name in ("leg_f", "leg_b")):
        leg_length = part_height("leg_f", "leg_b", factor=0.92)
    else:
        leg_length = part_height("legs", factor=0.92)
    torso_width, torso_height = scaled["torso"].size
    root_y = height * 0.82 - leg_length * 0.92
    parents = {
        "spine": "root",
        "cape": "spine",
        "neck": "spine",
        "hair_b": "neck",
        "sh_f": "spine",
        "el_f": "sh_f",
        "sh_b": "spine",
        "el_b": "sh_b",
        "hip_f": "root",
        "kn_f": "hip_f",
        "hip_b": "root",
        "kn_b": "hip_b",
        "hip_c": "root",
        "grip": "el_f" if "farm_f" in scaled else "sh_f" if "arm_f" in scaled else "spine",
    }
    offsets = {
        "spine": [0.0, 0.0],
        "cape": [0.0, -torso_height * 0.48],
        "neck": [0.0, -torso_height * 0.55],
        "hair_b": [0.0, -part_height("head", factor=0.60)],
        "sh_f": [torso_width * 0.06, -torso_height * 0.50],
        "el_f": [0.0, part_height("uarm_f", factor=0.88)],
        "sh_b": [-torso_width * 0.06, -torso_height * 0.50],
        "el_b": [0.0, part_height("uarm_b", factor=0.88)],
        "hip_f": [torso_width * 0.10, 0.0],
        "kn_f": [0.0, part_height("thigh_f", factor=0.88)],
        "hip_b": [-torso_width * 0.10, 0.0],
        "kn_b": [0.0, part_height("thigh_b", factor=0.88)],
        "hip_c": [0.0, 0.0],
        "grip": [
            0.0,
            part_height("farm_f", factor=0.90)
            if "farm_f" in scaled
            else part_height("arm_f", factor=0.75),
        ],
    }
    contract = _BIPED_SLOTS
    needed = {"root", "spine"}
    for name in scaled:
        needed.add(contract[name][0])
    stack = list(needed)
    while stack:
        joint = stack.pop()
        parent = parents.get(joint)
        if parent and parent not in needed:
            needed.add(parent)
            stack.append(parent)
    order = ("root", "spine", "cape", "neck", "hair_b", "sh_f", "el_f", "sh_b", "el_b", "hip_f", "kn_f", "hip_b", "kn_b", "hip_c", "grip")
    skeleton: dict[str, Any] = {}
    for joint in order:
        if joint not in needed:
            continue
        if joint == "root":
            skeleton[joint] = {"parent": None, "offset": [round(width / 2, 3), round(root_y, 3)]}
        else:
            skeleton[joint] = {
                "parent": parents[joint],
                "offset": [round(offsets[joint][0], 3), round(offsets[joint][1], 3)],
            }
    return scaled, skeleton, (width, height)


def _build_quadruped_geometry(
    parts: dict[str, Image.Image],
    height: int,
    resample: Image.Resampling,
) -> tuple[dict[str, Image.Image], dict[str, Any], tuple[int, int]]:
    if "body" not in parts or "head" not in parts:
        raise RigError("quadruped archetypes require body.png and head.png")
    width = int(round(height * 1.28 / 2) * 2)
    leg_est = max(
        (_first_height(parts, name) for name in ("foreleg_f", "foreleg_b", "hindleg_f", "hindleg_b")),
        default=0.0,
    )
    tail_width = parts["tail"].width if "tail" in parts else 0
    estimated_width = parts["body"].width + parts["head"].width * 0.72 + tail_width * 0.58
    estimated_height = parts["body"].height * 0.75 + leg_est
    scale = min((width * 0.72) / max(1, estimated_width), (height * 0.60) / max(1, estimated_height))
    scaled = _scaled_parts(parts, scale, resample)
    body_width, body_height = scaled["body"].size
    leg_length = max(
        (_first_height(scaled, name) for name in ("foreleg_f", "foreleg_b", "hindleg_f", "hindleg_b")),
        default=0.0,
    )
    ground = height * 0.82
    body_y = ground - leg_length * 0.9 - body_height * 0.24
    skeleton: dict[str, Any] = {
        "root": {"parent": None, "offset": [round(width * 0.50, 3), round(body_y, 3)]},
        "body": {"parent": "root", "offset": [0.0, 0.0]},
        "neck": {"parent": "body", "offset": [round(body_width * 0.42, 3), round(-body_height * 0.18, 3)]},
        "head": {"parent": "neck", "offset": [0.0, 0.0]},
    }
    optional_joints = {
        "tail": ("body", [-body_width * 0.42, -body_height * 0.10]),
        "foreleg_f": ("body", [body_width * 0.30, body_height * 0.24]),
        "foreleg_b": ("body", [body_width * 0.27, body_height * 0.22]),
        "hindleg_f": ("body", [-body_width * 0.30, body_height * 0.24]),
        "hindleg_b": ("body", [-body_width * 0.27, body_height * 0.22]),
        "wing_f": ("body", [-body_width * 0.08, -body_height * 0.22]),
        "wing_b": ("body", [-body_width * 0.12, -body_height * 0.18]),
    }
    for part_name, (parent, offset) in optional_joints.items():
        if part_name in scaled:
            skeleton[part_name] = {
                "parent": parent,
                "offset": [round(offset[0], 3), round(offset[1], 3)],
            }
    return scaled, skeleton, (width, height)


def _bind_pose(
    rig: dict[str, Any],
    output: Path,
    resample: Image.Resampling = Image.Resampling.BICUBIC,
) -> tuple[Path, Path]:
    bind = render_frame(
        rig,
        {"fps": 1, "loop": False, "tracks": {}},
        0.0,
        tuple(map(int, rig["canvas"])),
        resample=resample,
    )
    bind_path = safe_output_child(output, "bindpose.png", label="bind pose")
    bind.save(bind_path)
    debug = Image.new("RGBA", bind.size, (226, 229, 235, 255))
    debug.alpha_composite(bind)
    draw = ImageDraw.Draw(debug)
    from .rig_core import world_transforms

    transforms = world_transforms(rig["skeleton"], {"loop": False, "tracks": {}}, 0.0)
    for name, (_angle, x, y) in transforms.items():
        radius = 4 if name != "root" else 6
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(230, 54, 62, 220))
        draw.text((x + radius + 2, y - radius), name, fill=(32, 35, 42, 255))
    debug_path = safe_output_child(output, "rig-overlay.png", label="rig overlay")
    debug.save(debug_path)
    return bind_path, debug_path


def build_rig(
    parts: dict[str, Image.Image],
    output_dir: str | Path,
    *,
    archetype: str,
    character: str,
    height: int = 512,
    source_mode: str = "artist-parts",
    source_name: str | None = None,
    no_mirror: bool = False,
    clips: list[str] | None = None,
    resample: str = "bicubic",
    direction: str = "east",
) -> dict[str, Any]:
    if archetype not in ARCHETYPES:
        raise RigError(f"unknown archetype {archetype!r}; available: {', '.join(ARCHETYPES)}")
    if height < 64:
        raise RigError("rig height must be at least 64 pixels")
    if resample not in {"nearest", "bicubic"}:
        raise RigError("resample must be nearest or bicubic")
    sampling = Image.Resampling.NEAREST if resample == "nearest" else Image.Resampling.BICUBIC
    contract = _slot_contract(archetype)
    normalized: dict[str, Image.Image] = {}
    for raw_name, image in parts.items():
        name = _canonical_slot(archetype, raw_name)
        if name not in contract:
            raise RigError(f"unknown part slot {raw_name!r} for {archetype}")
        if name in normalized:
            raise RigError(f"duplicate part slot {name!r}")
        converted = image.convert("RGBA")
        box = converted.getchannel("A").getbbox()
        if box is None:
            raise RigError(f"part {name!r} has no foreground pixels")
        normalized[name] = converted.crop(box)
    generated_parts = _add_mirrors(normalized, archetype, no_mirror)
    validate_production_part_contract(normalized, archetype)
    if archetype == "biped-side":
        scaled, skeleton, canvas = _build_biped_geometry(normalized, height, sampling)
    else:
        scaled, skeleton, canvas = _build_quadruped_geometry(normalized, height, sampling)

    output = reset_output_directory(output_dir, label="compiled rig output")
    parts_output = safe_output_child(output, "parts", label="rig parts directory")
    parts_output.mkdir(parents=True, exist_ok=True)
    parts_json: dict[str, Any] = {}
    for name, image in sorted(scaled.items(), key=lambda item: (contract[item[0]][1], item[0])):
        joint, z, pivot_ratio = contract[name]
        target = safe_output_child(parts_output, f"{name}.png", label="rig part")
        image.save(target)
        parts_json[name] = {
            "image": f"parts/{name}.png",
            "pivot": [round(image.width * pivot_ratio[0], 3), round(image.height * pivot_ratio[1], 3)],
            "joint": joint,
            "z": z,
            "_image": image,
        }
    clip_pack = clips_for(
        _clip_archetype(archetype),
        canvas[1],
        skeleton.keys(),
        requested=clips,
    )
    source = {
        "mode": source_mode,
        "quality": "production",
        "occlusionSynthesis": False,
    }
    rig: dict[str, Any] = {
        "schemaVersion": 1,
        "id": character,
        "direction": direction,
        "archetype": archetype,
        "canvas": list(canvas),
        "source": source,
        "parts": parts_json,
        "skeleton": skeleton,
        "clips": clip_pack,
    }
    if source_name:
        rig["source"]["reference"] = Path(source_name).name
    rig_path = safe_output_child(output, "rig.json", label="RigSpec output")
    write_rig(rig, rig_path)
    bind_path, overlay_path = _bind_pose(rig, output, sampling)
    report = {
        "schemaVersion": 1,
        "quality": "production",
        "archetype": archetype,
        "character": character,
        "direction": direction,
        "canvas": list(canvas),
        "partCount": len(parts_json),
        "parts": sorted(parts_json),
        "mirroredParts": sorted(generated_parts),
        "resample": resample,
        "clips": {
            name: {"fps": clip["fps"], "loop": clip["loop"], "tracks": len(clip["tracks"])}
            for name, clip in clip_pack.items()
        },
        "rig": str(rig_path),
        "bindPose": str(bind_path),
        "overlay": str(overlay_path),
    }
    report_path = safe_output_child(output, "rig-report.json", label="rig report")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    return report


def _mask_part(
    canvas: Image.Image,
    mask: np.ndarray,
    pivot_world: tuple[int, int],
) -> tuple[Image.Image, list[int]] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    pivot_x, pivot_y = pivot_world
    left = min(int(xs.min()), pivot_x)
    top = min(int(ys.min()), pivot_y)
    right = max(int(xs.max()) + 1, pivot_x + 1)
    bottom = max(int(ys.max()) + 1, pivot_y + 1)
    rgba = np.asarray(canvas.convert("RGBA"))[top:bottom, left:right].copy()
    local_mask = mask[top:bottom, left:right]
    rgba[:, :, 3] = np.where(local_mask, rgba[:, :, 3], 0)
    return Image.fromarray(rgba), [pivot_x - left, pivot_y - top]


def _world_point(
    box: tuple[int, int, int, int],
    normalized: tuple[float, float],
) -> tuple[int, int]:
    left, top, right, bottom = box
    return (
        round(left + (right - left - 1) * normalized[0]),
        round(top + (bottom - top - 1) * normalized[1]),
    )


def _partition_foreground(
    alpha: np.ndarray,
    box: tuple[int, int, int, int],
    seeds: dict[str, tuple[float, float, float, float]],
) -> dict[str, np.ndarray]:
    """Assign every foreground pixel to one preset semantic seed.

    The weighted Voronoi partition is intentionally deterministic. It does not
    invent occluded pixels; the resulting rig is therefore marked coarse.
    """

    ys, xs = np.nonzero(alpha > 0)
    if not len(xs):
        raise RigError("reference contains no foreground pixels")
    left, top, right, bottom = box
    normalized_x = (xs - left) / max(1, right - left - 1)
    normalized_y = (ys - top) / max(1, bottom - top - 1)
    names = list(seeds)
    scores = []
    for name in names:
        seed_x, seed_y, weight_x, weight_y = seeds[name]
        scores.append(
            ((normalized_x - seed_x) / weight_x) ** 2
            + ((normalized_y - seed_y) / weight_y) ** 2
        )
    labels = np.argmin(np.stack(scores, axis=1), axis=1)
    masks = {name: np.zeros(alpha.shape, dtype=bool) for name in names}
    for index, name in enumerate(names):
        selected = labels == index
        masks[name][ys[selected], xs[selected]] = True
    return masks


def _alpha_iou(left: Image.Image, right: Image.Image) -> float:
    left_alpha = np.asarray(left.convert("RGBA"))[:, :, 3] > 0
    right_alpha = np.asarray(right.convert("RGBA"))[:, :, 3] > 0
    union = left_alpha | right_alpha
    if not union.any():
        return 1.0
    return float((left_alpha & right_alpha).sum() / union.sum())


def autorig_reference(
    reference_path: str | Path,
    output_dir: str | Path,
    *,
    archetype: str,
    character: str,
    height: int = 512,
    clips: list[str] | None = None,
    background_tolerance: int = 42,
    resample: str = "bicubic",
    direction: str = "east",
) -> dict[str, Any]:
    """Create an assisted coarse rig from one assembled character image.

    This path preserves visible pixels and emits editable diagnostics, but it
    cannot reconstruct hidden limbs or unseen surfaces. Production work should
    use :func:`build_rig` with separated parts or a disconnected part sheet.
    """

    if archetype not in ARCHETYPES:
        raise RigError(f"unknown archetype {archetype!r}; available: {', '.join(ARCHETYPES)}")
    if height < 64:
        raise RigError("rig height must be at least 64 pixels")
    if resample not in {"nearest", "bicubic"}:
        raise RigError("resample must be nearest or bicubic")
    sampling = Image.Resampling.NEAREST if resample == "nearest" else Image.Resampling.BICUBIC
    source = Path(reference_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"reference image not found: {source}")
    with Image.open(source) as opened:
        cleaned = remove_corner_background(opened, background_tolerance)
    source_box = cleaned.getchannel("A").getbbox()
    if source_box is None:
        raise RigError("reference contains no foreground after background removal")
    cropped = cleaned.crop(source_box)
    canvas_width = int(round(height * (0.92 if archetype == "biped-side" else 1.28) / 2) * 2)
    target_width = canvas_width * 0.72
    target_height = height * 0.68
    scale = min(target_width / cropped.width, target_height / cropped.height)
    scale = max(scale, 1 / max(cropped.width, cropped.height))
    resized = cropped.resize(
        (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))),
        sampling,
    )
    canvas = Image.new("RGBA", (canvas_width, height), (0, 0, 0, 0))
    x = (canvas_width - resized.width) // 2
    y = round(height * 0.82 - resized.height)
    canvas.alpha_composite(resized, (x, y))
    content_box = canvas.getchannel("A").getbbox()
    if content_box is None:
        raise RigError("scaled reference rendered empty")
    alpha = np.asarray(canvas)[:, :, 3]

    if archetype == "biped-side":
        seeds = {
            "head": (0.50, 0.12, 0.28, 0.16),
            "torso": (0.50, 0.39, 0.24, 0.22),
            "arm_b": (0.32, 0.42, 0.20, 0.25),
            "arm_f": (0.68, 0.42, 0.20, 0.25),
            "leg_b": (0.40, 0.78, 0.22, 0.32),
            "leg_f": (0.60, 0.78, 0.22, 0.32),
        }
        joints_normalized = {
            "root": (0.50, 0.57),
            "neck": (0.50, 0.23),
            "sh_b": (0.38, 0.31),
            "sh_f": (0.62, 0.31),
            "hip_b": (0.44, 0.56),
            "hip_f": (0.56, 0.56),
        }
        points = {name: _world_point(content_box, value) for name, value in joints_normalized.items()}
        root_x, root_y = points["root"]
        skeleton = {
            "root": {"parent": None, "offset": [root_x, root_y]},
            "spine": {"parent": "root", "offset": [0, 0]},
            "neck": {"parent": "spine", "offset": [points["neck"][0] - root_x, points["neck"][1] - root_y]},
            "sh_b": {"parent": "spine", "offset": [points["sh_b"][0] - root_x, points["sh_b"][1] - root_y]},
            "sh_f": {"parent": "spine", "offset": [points["sh_f"][0] - root_x, points["sh_f"][1] - root_y]},
            "hip_b": {"parent": "root", "offset": [points["hip_b"][0] - root_x, points["hip_b"][1] - root_y]},
            "hip_f": {"parent": "root", "offset": [points["hip_f"][0] - root_x, points["hip_f"][1] - root_y]},
        }
        part_contract = {
            "head": ("neck", points["neck"], 3),
            "torso": ("spine", points["root"], 2),
            "arm_b": ("sh_b", points["sh_b"], 1),
            "arm_f": ("sh_f", points["sh_f"], 5),
            "leg_b": ("hip_b", points["hip_b"], 0),
            "leg_f": ("hip_f", points["hip_f"], 4),
        }
    else:
        seeds = {
            "tail": (0.10, 0.46, 0.22, 0.22),
            "body": (0.47, 0.44, 0.28, 0.25),
            "head": (0.82, 0.34, 0.22, 0.22),
            "hindleg_f": (0.36, 0.76, 0.24, 0.31),
            "foreleg_f": (0.66, 0.76, 0.24, 0.31),
        }
        joints_normalized = {
            "root": (0.48, 0.48),
            "neck": (0.70, 0.38),
            "tail": (0.24, 0.46),
            "hindleg_f": (0.36, 0.56),
            "foreleg_f": (0.64, 0.56),
        }
        if archetype == "winged-quadruped-side":
            seeds["wing_f"] = (0.48, 0.18, 0.28, 0.20)
            joints_normalized["wing_f"] = (0.45, 0.32)
        points = {name: _world_point(content_box, value) for name, value in joints_normalized.items()}
        root_x, root_y = points["root"]
        skeleton = {
            "root": {"parent": None, "offset": [root_x, root_y]},
            "body": {"parent": "root", "offset": [0, 0]},
            "neck": {"parent": "body", "offset": [points["neck"][0] - root_x, points["neck"][1] - root_y]},
            "head": {"parent": "neck", "offset": [0, 0]},
            "tail": {"parent": "body", "offset": [points["tail"][0] - root_x, points["tail"][1] - root_y]},
            "hindleg_f": {"parent": "body", "offset": [points["hindleg_f"][0] - root_x, points["hindleg_f"][1] - root_y]},
            "foreleg_f": {"parent": "body", "offset": [points["foreleg_f"][0] - root_x, points["foreleg_f"][1] - root_y]},
        }
        part_contract = {
            "tail": ("tail", points["tail"], -1),
            "body": ("body", points["root"], 2),
            "head": ("head", points["neck"], 5),
            "hindleg_f": ("hindleg_f", points["hindleg_f"], 4),
            "foreleg_f": ("foreleg_f", points["foreleg_f"], 4),
        }
        if archetype == "winged-quadruped-side":
            skeleton["wing_f"] = {
                "parent": "body",
                "offset": [points["wing_f"][0] - root_x, points["wing_f"][1] - root_y],
            }
            part_contract["wing_f"] = ("wing_f", points["wing_f"], 3)

    masks = _partition_foreground(alpha, content_box, seeds)
    output = reset_output_directory(output_dir, label="coarse rig output")
    parts_output = safe_output_child(output, "parts", label="coarse rig parts directory")
    parts_output.mkdir(parents=True, exist_ok=True)
    parts_json: dict[str, Any] = {}
    empty_parts: list[str] = []
    for name, (joint, pivot_world, z) in part_contract.items():
        extracted = _mask_part(canvas, masks[name], pivot_world)
        if extracted is None:
            empty_parts.append(name)
            continue
        image, pivot = extracted
        image.save(
            safe_output_child(parts_output, f"{name}.png", label="coarse rig part")
        )
        parts_json[name] = {
            "image": f"parts/{name}.png",
            "pivot": pivot,
            "joint": joint,
            "z": z,
            "_image": image,
        }
    required = {"head", "torso"} if archetype == "biped-side" else {"head", "body"}
    missing_required = sorted(required - set(parts_json))
    if missing_required:
        raise RigError(f"coarse auto-rig produced empty required parts: {', '.join(missing_required)}")
    usable_joints = {
        part["joint"] for part in parts_json.values()
    } | {"root", "spine" if archetype == "biped-side" else "body", "neck"}
    skeleton = {name: joint for name, joint in skeleton.items() if name in usable_joints}
    # Parent closure is already present for every retained joint.
    clip_pack = clips_for(
        _clip_archetype(archetype),
        height,
        skeleton.keys(),
        requested=clips,
    )
    rig: dict[str, Any] = {
        "schemaVersion": 1,
        "id": character,
        "direction": direction,
        "archetype": archetype,
        "canvas": [canvas_width, height],
        "source": {
            "mode": "auto-segmented-flat",
            "quality": "coarse",
            "occlusionSynthesis": False,
            "reference": source.name,
        },
        "parts": parts_json,
        "skeleton": skeleton,
        "clips": clip_pack,
    }
    rig_path = safe_output_child(output, "rig.json", label="coarse RigSpec output")
    write_rig(rig, rig_path)
    bind_path, overlay_path = _bind_pose(rig, output, sampling)
    with Image.open(bind_path) as bind:
        rest_iou = _alpha_iou(canvas, bind)
    canvas.save(
        safe_output_child(output, "reference-normalized.png", label="normalized reference")
    )
    warnings = [
        "coarse auto-rig cannot reconstruct occluded limbs or unseen surfaces",
        "use separated production parts before approving large walk, attack, or death motion",
        "alpha reconstruction IoU does not measure semantic part accuracy",
    ]
    if min(resized.size) < 96:
        warnings.append("reference detail is below 96px; seams and face detail may be unstable")
    if empty_parts:
        warnings.append(f"empty optional regions were omitted: {', '.join(empty_parts)}")
    report = {
        "schemaVersion": 1,
        "quality": "coarse",
        "archetype": archetype,
        "character": character,
        "canvas": [canvas_width, height],
        "referenceSize": list(cleaned.size),
        "normalizedReferenceSize": list(resized.size),
        "restAlphaReconstructionIoU": round(rest_iou, 6),
        "semanticConfidence": "unscored",
        "occlusionSynthesis": False,
        "resample": resample,
        "direction": direction,
        "parts": sorted(parts_json),
        "warnings": warnings,
        "rig": str(rig_path),
        "bindPose": str(bind_path),
        "overlay": str(overlay_path),
    }
    report_path = safe_output_child(output, "autorig-report.json", label="coarse rig report")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    return report
