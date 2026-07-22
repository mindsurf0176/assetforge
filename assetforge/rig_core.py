from __future__ import annotations

import hashlib
import json
import math
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from jsonschema import Draft202012Validator
from PIL import Image, ImageDraw, ImageOps

from .json_utils import strict_json_loads
from .path_safety import reset_output_directory, safe_output_child


class RigError(ValueError):
    """Raised when a rig contract is unsafe or internally inconsistent."""


_PRODUCTION_PART_JOINTS: dict[str, dict[str, str]] = {
    "biped-side": {
        "cape": "cape",
        "thigh_b": "hip_b",
        "shin_b": "kn_b",
        "leg_b": "hip_b",
        "uarm_b": "sh_b",
        "farm_b": "el_b",
        "arm_b": "sh_b",
        "hair_b": "hair_b",
        "torso": "spine",
        "head": "neck",
        "thigh_f": "hip_f",
        "shin_f": "kn_f",
        "leg_f": "hip_f",
        "uarm_f": "sh_f",
        "farm_f": "el_f",
        "arm_f": "sh_f",
        "weapon": "grip",
    },
    "quadruped-side": {
        "tail": "tail",
        "hindleg_b": "hindleg_b",
        "foreleg_b": "foreleg_b",
        "body": "body",
        "hindleg_f": "hindleg_f",
        "foreleg_f": "foreleg_f",
        "head": "head",
    },
    "winged-quadruped-side": {
        "tail": "tail",
        "hindleg_b": "hindleg_b",
        "foreleg_b": "foreleg_b",
        "wing_b": "wing_b",
        "body": "body",
        "wing_f": "wing_f",
        "hindleg_f": "hindleg_f",
        "foreleg_f": "foreleg_f",
        "head": "head",
    },
}


def validate_production_part_contract(
    parts: dict[str, Any],
    archetype: str,
    *,
    require_joint_bindings: bool = False,
) -> None:
    """Enforce the semantic slots required for a deployable cutout rig.

    The same contract is used while compiling source art and while loading a
    hand-authored RigSpec. This prevents an edited JSON file from claiming
    ``production`` with a single unarticulated image.
    """

    expected_joints = _PRODUCTION_PART_JOINTS.get(archetype)
    if expected_joints is None:
        raise RigError(f"unsupported production archetype {archetype!r}")
    names = set(parts)
    unsupported = sorted(names - set(expected_joints))
    missing: list[str]
    if archetype == "biped-side":
        missing = sorted({"head", "torso"} - names)
        if names & {"arm_f", "arm_b"} and names & {
            "uarm_f", "uarm_b", "farm_f", "farm_b"
        }:
            raise RigError("biped-side cannot mix single-piece and segmented arm slots")
        if names & {"leg_f", "leg_b"} and names & {
            "thigh_f", "thigh_b", "shin_f", "shin_b"
        }:
            raise RigError("biped-side cannot mix single-piece and segmented leg slots")
        arm_ready = {"arm_f", "arm_b"} <= names or {
            "uarm_f", "farm_f", "uarm_b", "farm_b"
        } <= names
        leg_ready = {"leg_f", "leg_b"} <= names or {
            "thigh_f", "shin_f", "thigh_b", "shin_b"
        } <= names
        if not arm_ready:
            missing.append("complete front/back arm set")
        if not leg_ready:
            missing.append("complete front/back leg set")
    else:
        missing = sorted({"head", "body"} - names)
        if not {"foreleg_f", "foreleg_b"} <= names:
            missing.append("front/back foreleg set")
        if not {"hindleg_f", "hindleg_b"} <= names:
            missing.append("front/back hindleg set")
        if archetype == "winged-quadruped-side" and not {"wing_f", "wing_b"} <= names:
            missing.append("front/back wing set")
    if unsupported:
        missing.append(f"unsupported slots: {', '.join(unsupported)}")
    if missing:
        raise RigError(
            f"{archetype} production rig is incomplete: {', '.join(missing)}. "
            "Provide the missing art or allow mirrored back-side limbs."
        )
    if require_joint_bindings:
        binding_errors = [
            f"{name}->{part.get('joint')!r} (expected {expected_joints[name]!r})"
            for name, part in parts.items()
            if part.get("joint") != expected_joints[name]
        ]
        if binding_errors:
            raise RigError(
                f"{archetype} production rig has invalid semantic joint bindings: "
                + ", ".join(binding_errors)
            )


def _schema_validate(rig: dict[str, Any]) -> None:
    schema_path = files("assetforge.schemas").joinpath("rig-spec.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(rig), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.absolute_path)) or "$"
        raise RigError(f"RigSpec validation failed at {location}: {error.message}")


def _pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise RigError(f"{label} must be a two-item array")
    try:
        pair = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise RigError(f"{label} values must be numbers") from exc
    if not all(math.isfinite(number) for number in pair):
        raise RigError(f"{label} values must be finite numbers")
    return pair


def _safe_relative_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise RigError(f"{label} must be a non-empty relative path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or value.startswith(("~", "\\")):
        raise RigError(f"{label} must stay inside the rig directory: {value!r}")
    return candidate


def _validate_track(keys: Any, label: str) -> None:
    if not isinstance(keys, list) or not keys:
        raise RigError(f"{label} must contain at least one keyframe")
    previous = -math.inf
    for index, key in enumerate(keys):
        time, _value = _pair(key, f"{label}[{index}]")
        if not 0 <= time <= 1:
            raise RigError(f"{label}[{index}] time must be between 0 and 1")
        if time < previous:
            raise RigError(f"{label} keyframe times must be sorted")
        previous = time


def validate_rig(rig: dict[str, Any], rig_root: str | Path | None = None) -> dict[str, Any]:
    """Validate graph integrity, part references, and executable clip tracks.

    JSON Schema checks shape. This function checks the relationships JSON Schema
    cannot express: safe paths, a single acyclic skeleton, valid part joints, and
    pivots that lie inside their source images.
    """

    if not isinstance(rig, dict):
        raise RigError("rig must be a JSON object")
    contract = {
        key: value
        for key, value in rig.items()
        if key != "_image"
    }
    contract["parts"] = {
        name: {key: value for key, value in part.items() if key != "_image"}
        for name, part in rig.get("parts", {}).items()
    } if isinstance(rig.get("parts"), dict) else rig.get("parts")
    _schema_validate(contract)
    canvas = _pair(rig.get("canvas"), "canvas")
    if canvas[0] < 1 or canvas[1] < 1:
        raise RigError("canvas dimensions must be positive")
    skeleton = rig.get("skeleton")
    parts = rig.get("parts")
    if not isinstance(skeleton, dict) or not skeleton:
        raise RigError("skeleton must be a non-empty object")
    if not isinstance(parts, dict) or not parts:
        raise RigError("parts must be a non-empty object")
    if rig["source"]["quality"] == "production":
        validate_production_part_contract(
            parts,
            rig["archetype"],
            require_joint_bindings=True,
        )

    roots: list[str] = []
    for name, joint in skeleton.items():
        if not isinstance(name, str) or not name:
            raise RigError("joint names must be non-empty strings")
        if not isinstance(joint, dict):
            raise RigError(f"joint {name!r} must be an object")
        _pair(joint.get("offset"), f"skeleton.{name}.offset")
        parent = joint.get("parent")
        if parent is None:
            roots.append(name)
        elif parent not in skeleton:
            raise RigError(f"joint {name!r} references missing parent {parent!r}")
    if len(roots) != 1:
        raise RigError(f"skeleton must have exactly one root; found {len(roots)}")

    state: dict[str, int] = {}

    def visit(name: str) -> None:
        marker = state.get(name, 0)
        if marker == 1:
            raise RigError(f"skeleton contains a cycle at joint {name!r}")
        if marker == 2:
            return
        state[name] = 1
        parent = skeleton[name].get("parent")
        if parent is not None:
            visit(parent)
        state[name] = 2

    for joint_name in skeleton:
        visit(joint_name)

    root_path = Path(rig_root).expanduser().resolve() if rig_root is not None else None
    for name, part in parts.items():
        if not isinstance(part, dict):
            raise RigError(f"part {name!r} must be an object")
        if part.get("joint") not in skeleton:
            raise RigError(f"part {name!r} references missing joint {part.get('joint')!r}")
        relative = _safe_relative_path(part.get("image"), f"parts.{name}.image")
        pivot = _pair(part.get("pivot"), f"parts.{name}.pivot")
        if not isinstance(part.get("z", 0), (int, float)):
            raise RigError(f"parts.{name}.z must be a number")
        if root_path is not None:
            image_path = (root_path / Path(*relative.parts)).resolve()
            try:
                image_path.relative_to(root_path)
            except ValueError as exc:
                raise RigError(f"part {name!r} escapes the rig directory") from exc
            if not image_path.is_file():
                raise RigError(f"part image not found: {image_path}")
            with Image.open(image_path) as image:
                if not (0 <= pivot[0] <= image.width and 0 <= pivot[1] <= image.height):
                    raise RigError(
                        f"part {name!r} pivot {list(pivot)} is outside image {list(image.size)}"
                    )

    clips = rig.get("clips", {})
    if not isinstance(clips, dict):
        raise RigError("clips must be an object")
    for clip_name, clip in clips.items():
        if not isinstance(clip, dict):
            raise RigError(f"clip {clip_name!r} must be an object")
        fps = clip.get("fps")
        if (
            not isinstance(fps, (int, float))
            or not math.isfinite(float(fps))
            or fps <= 0
        ):
            raise RigError(f"clip {clip_name!r} fps must be positive")
        if not isinstance(clip.get("loop"), bool):
            raise RigError(f"clip {clip_name!r} must declare loop=true or loop=false")
        if clip.get("ease", "linear") not in {"linear", "smooth"}:
            raise RigError(f"clip {clip_name!r} has unsupported ease")
        tracks = clip.get("tracks")
        if not isinstance(tracks, dict):
            raise RigError(f"clip {clip_name!r} tracks must be an object")
        for joint_name, channels in tracks.items():
            if joint_name not in skeleton:
                raise RigError(f"clip {clip_name!r} references missing joint {joint_name!r}")
            if not isinstance(channels, dict) or not channels:
                raise RigError(f"clip {clip_name!r} joint {joint_name!r} has no channels")
            for channel, keys in channels.items():
                if channel not in {"rotation", "offset_x", "offset_y"}:
                    raise RigError(
                        f"clip {clip_name!r} joint {joint_name!r} has unsupported channel {channel!r}"
                    )
                _validate_track(keys, f"clips.{clip_name}.tracks.{joint_name}.{channel}")
    return rig


def load_rig(path: str | Path) -> tuple[dict[str, Any], Path]:
    rig_path = Path(path).expanduser().resolve()
    if not rig_path.is_file():
        raise FileNotFoundError(f"rig file not found: {rig_path}")
    try:
        rig = strict_json_loads(rig_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        raise RigError(f"invalid rig JSON: {exc}") from exc
    validate_rig(rig, rig_path.parent)
    for name, part in rig["parts"].items():
        relative = _safe_relative_path(part["image"], f"parts.{name}.image")
        image_path = rig_path.parent.joinpath(*relative.parts)
        with Image.open(image_path) as image:
            part["_image"] = image.convert("RGBA").copy()
    return rig, rig_path


def serializable_rig(rig: dict[str, Any]) -> dict[str, Any]:
    clean = json.loads(json.dumps(rig, default=lambda _value: None))
    for part in clean.get("parts", {}).values():
        part.pop("_image", None)
    return clean


def write_rig(rig: dict[str, Any], path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    clean = serializable_rig(rig)
    validate_rig(clean, target.parent)
    target.write_text(json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def sample_track(
    keys: list[list[float]] | None,
    time: float,
    *,
    default: float = 0.0,
    ease: str = "linear",
    loop: bool = True,
) -> float:
    if not keys:
        return default
    if len(keys) == 1:
        return float(keys[0][1])
    time = time % 1.0 if loop else min(1.0, max(0.0, time))
    if not loop and time >= 1.0:
        return float(keys[-1][1])
    if time <= keys[0][0]:
        return float(keys[0][1])
    for index in range(len(keys) - 1):
        time0, value0 = keys[index]
        time1, value1 = keys[index + 1]
        if time0 <= time <= time1:
            fraction = 0.0 if time1 == time0 else (time - time0) / (time1 - time0)
            if ease == "smooth":
                fraction = 0.5 - 0.5 * math.cos(math.pi * fraction)
            return float(value0 + (value1 - value0) * fraction)
    return float(keys[-1][1])


def world_transforms(
    skeleton: dict[str, Any],
    clip: dict[str, Any],
    time: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> dict[str, tuple[float, float, float]]:
    tracks = clip.get("tracks", {})
    ease = clip.get("ease", "linear")
    loop = bool(clip.get("loop", True))
    solved: dict[str, tuple[float, float, float]] = {}

    def solve(name: str) -> tuple[float, float, float]:
        if name in solved:
            return solved[name]
        joint = skeleton[name]
        channels = tracks.get(name, {})
        angle = sample_track(channels.get("rotation"), time, ease=ease, loop=loop)
        offset_x, offset_y = map(float, joint.get("offset", [0, 0]))
        offset_x += sample_track(channels.get("offset_x"), time, ease=ease, loop=loop)
        offset_y += sample_track(channels.get("offset_y"), time, ease=ease, loop=loop)
        parent = joint.get("parent")
        if parent is None:
            solved[name] = (angle, offset_x + origin[0], offset_y + origin[1])
        else:
            parent_angle, parent_x, parent_y = solve(parent)
            radians = math.radians(parent_angle)
            rotated_x = offset_x * math.cos(radians) + offset_y * math.sin(radians)
            rotated_y = -offset_x * math.sin(radians) + offset_y * math.cos(radians)
            solved[name] = (
                parent_angle + angle,
                parent_x + rotated_x,
                parent_y + rotated_y,
            )
        return solved[name]

    for joint_name in skeleton:
        solve(joint_name)
    return solved


def _rotate_about_pivot(
    image: Image.Image,
    angle: float,
    pivot: tuple[float, float],
    resample: Image.Resampling,
) -> tuple[Image.Image, tuple[int, int]]:
    diameter = int(math.ceil(math.hypot(image.width, image.height))) * 2 + 2
    center = diameter // 2
    canvas = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    canvas.alpha_composite(
        image,
        (int(round(center - pivot[0])), int(round(center - pivot[1]))),
    )
    if angle % 360:
        canvas = canvas.rotate(angle, resample=resample, center=(center, center))
    return canvas, (center, center)


def render_frame(
    rig: dict[str, Any],
    clip: dict[str, Any],
    time: float,
    size: tuple[int, int] | None = None,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    resample: Image.Resampling = Image.Resampling.BICUBIC,
) -> Image.Image:
    width, height = size or tuple(map(int, rig["canvas"]))
    transforms = world_transforms(rig["skeleton"], clip, time, origin)
    canvas = Image.new("RGBA", (int(width), int(height)), (0, 0, 0, 0))
    for _name, part in sorted(rig["parts"].items(), key=lambda item: (item[1].get("z", 0), item[0])):
        angle, world_x, world_y = transforms[part["joint"]]
        rotated, pivot = _rotate_about_pivot(
            part["_image"],
            angle,
            tuple(map(float, part["pivot"])),
            resample,
        )
        canvas.alpha_composite(
            rotated,
            (int(round(world_x - pivot[0])), int(round(world_y - pivot[1]))),
        )
    return canvas


def frame_times(frame_count: int, loop: bool) -> list[float]:
    if frame_count < 1:
        raise RigError("frame count must be at least 1")
    if frame_count == 1:
        return [0.0]
    if loop:
        return [index / frame_count for index in range(frame_count)]
    return [index / (frame_count - 1) for index in range(frame_count)]


def minimum_useful_frame_count(clip: dict[str, Any]) -> int:
    """Return the minimum sample count that cannot skip authored motion.

    A recovering non-loop clip often has the same pose at t=0 and t=1, with
    all of the attack or recoil stored in interior keys. Sampling only those
    endpoints would emit duplicate stills while claiming to be an animation.
    """

    if clip.get("loop", True):
        return 1
    has_motion = False
    needs_interior_sample = False
    for channels in clip.get("tracks", {}).values():
        for keys in channels.values():
            values = [float(key[1]) for key in keys]
            if not values or all(math.isclose(value, values[0]) for value in values[1:]):
                continue
            has_motion = True
            if (
                len(values) > 2
                and math.isclose(values[0], values[-1])
                and any(not math.isclose(value, values[0]) for value in values[1:-1])
            ):
                needs_interior_sample = True
    if needs_interior_sample:
        return 3
    if has_motion:
        return 2
    return 1


def validate_clip_frame_count(clip_name: str, clip: dict[str, Any], frame_count: int) -> None:
    minimum = minimum_useful_frame_count(clip)
    if frame_count < minimum:
        reason = (
            "its authored motion occurs between matching start and recovery poses"
            if minimum == 3
            else "it contains non-loop motion"
        )
        raise RigError(
            f"clip {clip_name!r} requires at least {minimum} frames because {reason}; "
            f"got {frame_count}"
        )


def render_clip(
    rig: dict[str, Any],
    clip: dict[str, Any],
    frame_count: int,
    size: tuple[int, int] | None = None,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    resample: Image.Resampling = Image.Resampling.BICUBIC,
) -> list[Image.Image]:
    validate_clip_frame_count("animation", clip, frame_count)
    return [
        render_frame(rig, clip, time, size, origin=origin, resample=resample)
        for time in frame_times(frame_count, bool(clip.get("loop", True)))
    ]


def _expanded_box(
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, box[0] - padding),
        max(0, box[1] - padding),
        min(size[0], box[2] + padding),
        min(size[1], box[3] + padding),
    )


def _union_box(boxes: Iterable[tuple[int, int, int, int] | None]) -> tuple[int, int, int, int]:
    present = [box for box in boxes if box is not None]
    if not present:
        raise RigError("all rendered frames are empty")
    return (
        min(box[0] for box in present),
        min(box[1] for box in present),
        max(box[2] for box in present),
        max(box[3] for box in present),
    )


def _contact_sheet(images: list[Image.Image], labels: list[str], output: Path) -> None:
    if not images:
        return
    columns = min(6, len(images))
    rows = math.ceil(len(images) / columns)
    tile_width = max(image.width for image in images)
    tile_height = max(image.height for image in images) + 16
    sheet = Image.new("RGBA", (columns * tile_width, rows * tile_height), (28, 32, 40, 255))
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        draw.rectangle((x, y, x + tile_width - 1, y + tile_height - 1), outline=(68, 76, 88, 255))
        sheet.alpha_composite(image, (x + (tile_width - image.width) // 2, y + tile_height - 16 - image.height))
        draw.text((x + 3, y + tile_height - 14), label, fill=(230, 233, 238, 255))
    sheet.save(output)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_animation_set(
    rig: dict[str, Any],
    output_dir: str | Path,
    clip_names: list[str],
    frame_counts: dict[str, int],
    *,
    resample: str = "bicubic",
    make_gifs: bool = True,
    mirror_x: bool = False,
    timing_source: str = "rig",
) -> dict[str, Any]:
    """Render all clips with one shared motion bound and deterministic canvas.

    A generous temporary canvas prevents attack/death motion from being clipped.
    The final crop is computed once across every requested clip, so scale and world
    placement do not jump when the animation changes.
    """

    if not clip_names:
        raise RigError("at least one clip is required")
    canonical_order = {name: index for index, name in enumerate(("idle", "walk", "attack", "hit", "death"))}
    clip_names = sorted(
        dict.fromkeys(clip_names),
        key=lambda name: (canonical_order.get(name, len(canonical_order)), name),
    )
    missing = [name for name in clip_names if name not in rig.get("clips", {})]
    if missing:
        raise RigError(f"rig is missing requested clips: {', '.join(missing)}")
    for clip_name in clip_names:
        validate_clip_frame_count(
            clip_name,
            rig["clips"][clip_name],
            int(frame_counts[clip_name]),
        )
    source_width, source_height = map(int, rig["canvas"])
    motion_padding = max(source_width, source_height)
    render_size = (source_width + motion_padding * 2, source_height + motion_padding * 2)
    origin = (float(motion_padding), float(motion_padding))
    sampling = Image.Resampling.NEAREST if resample == "nearest" else Image.Resampling.BICUBIC

    reference_clip = "idle" if "idle" in clip_names else clip_names[0]
    reference_frame = render_frame(
        rig,
        rig["clips"][reference_clip],
        0.0,
        render_size,
        origin=origin,
        resample=sampling,
    )
    if mirror_x:
        reference_frame = ImageOps.mirror(reference_frame)
    reference_box = reference_frame.getchannel("A").getbbox()
    if reference_box is None:
        raise RigError("reference pose rendered empty")
    reference_ground = reference_box[3] - 1

    def render_sample(
        clip_name: str,
        clip: dict[str, Any],
        frame_index: int,
        time: float,
    ) -> Image.Image:
        frame = render_frame(rig, clip, time, render_size, origin=origin, resample=sampling)
        if mirror_x:
            frame = ImageOps.mirror(frame)
        box = frame.getchannel("A").getbbox()
        if box is None:
            raise RigError(
                f"clip {clip_name!r} frame {frame_index} rendered empty; "
                "motion moved every part outside the temporary canvas"
            )
        if box[0] == 0 or box[1] == 0 or box[2] == render_size[0] or box[3] == render_size[1]:
            raise RigError(
                f"clip {clip_name!r} frame {frame_index} touches the temporary canvas edge; "
                "increase the RigSpec canvas or reduce part size/motion"
            )
        if clip.get("grounded", False):
            offset_y = reference_ground - (box[3] - 1)
            if offset_y:
                grounded = Image.new("RGBA", render_size, (0, 0, 0, 0))
                grounded.alpha_composite(frame, (0, offset_y))
                frame = grounded
                box = frame.getchannel("A").getbbox()
                if box is None or box[1] == 0 or box[3] == render_size[1]:
                    raise RigError(
                        f"clip {clip_name!r} frame {frame_index} overflows after ground locking"
                    )
        return frame

    boxes: list[tuple[int, int, int, int]] = []
    for clip_name in clip_names:
        clip = rig["clips"][clip_name]
        count = int(frame_counts[clip_name])
        for frame_index, time in enumerate(
            frame_times(count, bool(clip.get("loop", True)))
        ):
            frame = render_sample(clip_name, clip, frame_index, time)
            box = frame.getchannel("A").getbbox()
            assert box is not None
            boxes.append(box)
    union = _union_box(boxes)
    crop_padding = max(2, round(max(union[2] - union[0], union[3] - union[1]) * 0.035))
    crop = _expanded_box(union, render_size, crop_padding)
    reference_anchor = [
        (reference_box[0] + reference_box[2] - 1) // 2 - crop[0],
        reference_box[3] - 1 - crop[1],
    ]
    content_bounds = [
        union[0] - crop[0],
        union[1] - crop[1],
        union[2] - crop[0],
        union[3] - crop[1],
    ]
    # Anchor normalization to the reference standing pose. The union's lowest
    # pixel may belong to a fall/death frame and would make every normal clip
    # float above the runtime ground line.
    motion_anchor = list(reference_anchor)

    output = reset_output_directory(output_dir, label="raw animation output")
    manifest_clips: dict[str, Any] = {}
    preview_images: list[Image.Image] = []
    preview_labels: list[str] = []
    for clip_name in clip_names:
        clip = rig["clips"][clip_name]
        count = int(frame_counts[clip_name])
        frames = []
        for frame_index, time in enumerate(
            frame_times(count, bool(clip.get("loop", True)))
        ):
            frame = render_sample(clip_name, clip, frame_index, time)
            frames.append(frame.crop(crop))
        clip_dir = safe_output_child(output, clip_name, label="clip output")
        clip_dir.mkdir(parents=True, exist_ok=True)
        for old in clip_dir.glob("*.png"):
            old.unlink()
        digits = max(2, len(str(max(0, count - 1))))
        files: list[dict[str, Any]] = []
        for index, frame in enumerate(frames):
            target = safe_output_child(
                clip_dir,
                f"{clip_name}_{index:0{digits}d}.png",
                label="frame output",
            )
            frame.save(target)
            files.append({"file": str(target), "sha256": _sha256(target)})
        gif_path: str | None = None
        if make_gifs and frames:
            gif = safe_output_child(output, f"{clip_name}.gif", label="GIF preview")
            duration = max(20, round(1000 / float(clip["fps"])))
            save_options = {
                "save_all": True,
                "append_images": frames[1:],
                "duration": [duration] * len(frames),
                "disposal": 2,
                "optimize": False,
            }
            if clip.get("loop", True):
                save_options["loop"] = 0
            frames[0].save(gif, **save_options)
            gif_path = str(gif)
        preview_images.extend(frames)
        preview_labels.extend(f"{clip_name} {index + 1}/{count}" for index in range(count))
        manifest_clips[clip_name] = {
            "fps": clip["fps"],
            "loop": bool(clip.get("loop", True)),
            "grounded": bool(clip.get("grounded", False)),
            "frameCount": count,
            "directory": str(clip_dir),
            "frames": files,
            "gif": gif_path,
        }

    contact = safe_output_child(output, "contact-sheet.png", label="contact sheet")
    _contact_sheet(preview_images, preview_labels, contact)
    manifest = {
        "schemaVersion": 1,
        "renderer": "local-cutout-v1",
        "timingSource": timing_source,
        "fit": "shared-motion-bounds",
        "canvas": [crop[2] - crop[0], crop[3] - crop[1]],
        "sourceCanvas": [source_width, source_height],
        "motionBounds": list(crop),
        "referenceClip": reference_clip,
        "referenceAnchor": reference_anchor,
        "contentBounds": content_bounds,
        "motionAnchor": motion_anchor,
        "resample": resample,
        "mirroredX": mirror_x,
        "output": str(output),
        "contactSheet": str(contact),
        "clips": manifest_clips,
    }
    manifest_path = safe_output_child(output, "animation-manifest.json", label="animation manifest")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest
