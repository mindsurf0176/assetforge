from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from PIL import Image

from .exporters import deploy_animation_direction, export_assets, join_resource_prefix
from .frames import build_shared_palette, frame_paths, ingest_frames
from .json_utils import strict_json_loads
from .path_safety import reset_output_directory, safe_output_child
from .profile import Profile, load_profile
from .rig_build import autorig_reference, build_rig, load_named_parts, load_sheet_parts
from .rig_core import RigError, load_rig, render_animation_set
from .validation import enclosed_transparent_hole_areas, validate_frames


DEFAULT_FRAME_COUNTS = {
    "idle": 6,
    "walk": 8,
    "attack": 6,
    "hit": 4,
    "death": 8,
}

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CLIP_ORDER = {name: index for index, name in enumerate(DEFAULT_FRAME_COUNTS)}


def _require_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise RigError(
            f"{label} must match {_IDENTIFIER.pattern!r}; got {value!r}"
        )
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def parse_clips(value: str | list[str] | None) -> list[str]:
    if value is None:
        return list(DEFAULT_FRAME_COUNTS)
    values = value.split(",") if isinstance(value, str) else value
    result = list(dict.fromkeys(name.strip() for name in values if name.strip()))
    if not result:
        raise RigError("at least one clip is required")
    return result


def _canonical_clips(values: list[str]) -> list[str]:
    return sorted(
        dict.fromkeys(values),
        key=lambda name: (_CLIP_ORDER.get(name, len(_CLIP_ORDER)), name),
    )


def parse_frame_counts(value: str | None) -> dict[str, int]:
    if not value:
        return {}
    result: dict[str, int] = {}
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise RigError(f"invalid frame override {token!r}; expected clip=count")
        name, raw_count = (part.strip() for part in token.split("=", 1))
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise RigError(f"invalid frame count {raw_count!r} for {name!r}") from exc
        if count < 1:
            raise RigError(f"frame count for {name!r} must be at least 1")
        result[name] = count
    return result


def _resolved_frame_counts(
    clips: list[str],
    overrides: dict[str, int],
    profile: Profile | None,
) -> dict[str, int]:
    unknown_overrides = sorted(set(overrides) - set(clips))
    if unknown_overrides:
        raise RigError(f"frame overrides target unrequested clips: {', '.join(unknown_overrides)}")
    result: dict[str, int] = {}
    for clip in clips:
        count = overrides.get(clip, DEFAULT_FRAME_COUNTS.get(clip, 6))
        if profile is not None:
            contract = profile.animation(clip)
            minimum = int(contract.get("minFrames", 1))
            maximum = int(contract.get("maxFrames", max(minimum, count)))
            if clip not in overrides:
                count = max(minimum, min(maximum, count))
            elif not minimum <= count <= maximum:
                raise RigError(
                    f"animation {clip!r} frame override {count} violates profile range "
                    f"{minimum}..{maximum}"
                )
        result[clip] = count
    return result


def load_animation_spec(path: str | Path) -> tuple[dict[str, Any], Path]:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"animation spec not found: {target}")
    try:
        data = strict_json_loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        raise RigError(f"invalid AnimationSpec JSON: {exc}") from exc
    schema = json.loads(
        files("assetforge.schemas").joinpath("animation-spec.schema.json").read_text(encoding="utf-8")
    )
    errors = sorted(
        Draft202012Validator(schema).iter_errors(data),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.absolute_path)) or "$"
        raise RigError(f"AnimationSpec validation failed at {location}: {error.message}")
    rig_reference = Path(data["rig"])
    if rig_reference.is_absolute() or ".." in rig_reference.parts:
        raise RigError("AnimationSpec rig path must stay inside the spec directory")
    return data, target


def prepare_rig(
    work_dir: str | Path,
    *,
    character: str,
    archetype: str | None,
    rig_path: str | Path | None = None,
    parts_dir: str | Path | None = None,
    part_sheet: str | Path | None = None,
    reference: str | Path | None = None,
    mapping: str | Path | None = None,
    height: int = 512,
    clips: list[str] | None = None,
    no_mirror: bool = False,
    resample: str = "bicubic",
    direction: str = "east",
) -> tuple[dict[str, Any], Path, dict[str, Any] | None]:
    selected = [value is not None for value in (rig_path, parts_dir, part_sheet, reference)]
    if sum(selected) != 1:
        raise RigError("choose exactly one input: --rig, --parts, --part-sheet, or --reference")
    if mapping is not None and part_sheet is None:
        raise RigError("--mapping can only be used with --part-sheet")
    if no_mirror and parts_dir is None and part_sheet is None:
        raise RigError("--no-mirror can only be used with --parts or --part-sheet")
    if rig_path is not None:
        if archetype is not None:
            raise RigError("--archetype cannot override a compiled --rig")
        rig, source = load_rig(rig_path)
        return rig, source, None
    if archetype is None:
        raise RigError("--archetype is required when building a rig from art")
    work = Path(work_dir).expanduser().resolve()
    rig_output = safe_output_child(work, "rig", label="rig output")
    extraction: dict[str, Any] | None = None
    if reference is not None:
        report = autorig_reference(
            reference,
            rig_output,
            archetype=archetype,
            character=character,
            height=height,
            clips=clips,
            resample=resample,
            direction=direction,
        )
        rig, source = load_rig(report["rig"])
        return rig, source, report
    if parts_dir is not None:
        parts = load_named_parts(parts_dir, archetype)
        mode = "artist-parts"
        source_name = Path(parts_dir).name
    else:
        if mapping is None:
            raise RigError("--mapping is required with --part-sheet")
        parts, extraction = load_sheet_parts(
            part_sheet,
            mapping,
            safe_output_child(work, "sheet-extraction", label="sheet extraction output"),
            archetype,
        )
        mode = "part-sheet"
        source_name = Path(part_sheet).name
    report = build_rig(
        parts,
        rig_output,
        archetype=archetype,
        character=character,
        height=height,
        source_mode=mode,
        source_name=source_name,
        no_mirror=no_mirror,
        clips=clips,
        resample=resample,
        direction=direction,
    )
    rig, source = load_rig(report["rig"])
    return rig, source, extraction


def _character_palette(
    rendered: dict[str, Any],
    profile: Profile,
) -> Image.Image | None:
    palette_config = profile.data["quality"].get("palette", {})
    if not palette_config.get("lockAcrossClip", True):
        return None
    images: list[Image.Image] = []
    for clip in rendered["clips"].values():
        for path in frame_paths(clip["directory"]):
            with Image.open(path) as image:
                images.append(image.convert("RGBA").copy())
    quality = profile.data["quality"]
    return build_shared_palette(
        images,
        int(palette_config.get("maxColors", 32)),
        int(quality.get("alphaThreshold", 20)),
    )


def _validate_production_part_alpha(rig: dict[str, Any], profile: Profile) -> None:
    if rig["source"]["quality"] != "production":
        return
    quality = profile.data["quality"]
    configured = quality.get("partAlpha", {}).get("maxEnclosedTransparentPixels")
    if configured is None:
        return
    limit = int(configured)
    min_alpha = int(quality.get("alphaThreshold", 20))
    failures: list[str] = []
    for name, part in rig["parts"].items():
        areas = enclosed_transparent_hole_areas(part["_image"], min_alpha)
        pixels = sum(areas)
        if pixels > limit:
            failures.append(f"{name}.png={pixels}px/{len(areas)} component(s)")
    if failures:
        raise RigError(
            "production source parts violate the profile enclosed-alpha gate: "
            + ", ".join(failures)
        )


def _profile_timed_rig(
    rig: dict[str, Any],
    clips: list[str],
    profile: Profile,
) -> dict[str, Any]:
    """Use profile FPS while rejecting incompatible loop semantics."""

    effective = dict(rig)
    effective["clips"] = {name: dict(clip) for name, clip in rig["clips"].items()}
    for clip_name in clips:
        if clip_name not in rig["clips"]:
            raise RigError(f"rig is missing requested clip {clip_name!r}")
        rig_clip = rig["clips"][clip_name]
        contract = profile.animation(clip_name)
        rig_loop = bool(rig_clip.get("loop", True))
        profile_loop = bool(contract.get("loop", False))
        if rig_loop != profile_loop:
            raise RigError(
                f"clip {clip_name!r} loop contract mismatch: "
                f"RigSpec={rig_loop}, profile={profile_loop}"
            )
        effective["clips"][clip_name]["fps"] = float(contract["fps"])
    return effective


def _export_extension(profile: Profile) -> str:
    engine = profile.data.get("export", {}).get("engine")
    return ".tres" if engine == "godot" else ".json"


def run_local_animation(
    *,
    work_dir: str | Path,
    character: str,
    direction: str,
    clips: list[str] | None,
    frame_overrides: dict[str, int] | None = None,
    rig_path: str | Path | None = None,
    parts_dir: str | Path | None = None,
    part_sheet: str | Path | None = None,
    reference: str | Path | None = None,
    mapping: str | Path | None = None,
    archetype: str | None = None,
    height: int = 512,
    resample: str = "bicubic",
    profile_name: str | None = None,
    tier: str | None = None,
    resource_prefix: str | None = None,
    deploy_dir: str | Path | None = None,
    no_mirror: bool = False,
) -> dict[str, Any]:
    _require_identifier(character, "character")
    _require_identifier(direction, "direction")
    if resample not in {"nearest", "bicubic"}:
        raise RigError("resample must be nearest or bicubic")
    profile = load_profile(profile_name) if profile_name else None
    if clips is None:
        clips = list(DEFAULT_FRAME_COUNTS)
        if profile is not None:
            available = profile.data.get("animations", {})
            clips = [name for name in clips if name in available]
            if not clips:
                raise RigError(
                    f"profile {profile.id!r} shares no built-in local animation clips"
                )
    for clip in clips:
        _require_identifier(clip, "clip")
    clips = _canonical_clips(clips)
    if profile is not None and not tier:
        raise RigError("--tier is required when --profile is used")
    if tier is not None and profile is None:
        raise RigError("--tier requires --profile")
    if resource_prefix is not None and profile is None:
        raise RigError("--resource-prefix requires --profile")
    if deploy_dir is not None and profile is None:
        raise RigError("--deploy-dir requires --profile")
    if reference is not None and deploy_dir is not None:
        raise RigError(
            "--reference creates a coarse rig and cannot be used with --deploy-dir"
        )
    if profile is not None:
        for clip in clips:
            profile.animation(clip)
        directions = profile.data.get("directions", [])
        mirrored_directions = profile.data.get("mirrorDirections", {})
        if direction not in directions and direction not in mirrored_directions:
            raise RigError(
                f"direction {direction!r} is not supported by profile {profile.id!r}: "
                f"{', '.join([*directions, *mirrored_directions])}"
            )
    if deploy_dir is not None and resource_prefix is None:
        raise RigError("--deploy-dir requires --resource-prefix so the destination can be verified")

    frame_counts = _resolved_frame_counts(clips, frame_overrides or {}, profile)

    work = Path(work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    rig, source, extraction = prepare_rig(
        work,
        character=character,
        archetype=archetype,
        rig_path=rig_path,
        parts_dir=parts_dir,
        part_sheet=part_sheet,
        reference=reference,
        mapping=mapping,
        height=height,
        clips=clips,
        no_mirror=no_mirror,
        resample=resample,
        direction=direction,
    )
    if rig["id"] != character:
        raise RigError(
            f"compiled rig character {rig['id']!r} does not match requested "
            f"character {character!r}"
        )
    if profile is not None:
        _validate_production_part_alpha(rig, profile)
        rig = _profile_timed_rig(rig, clips, profile)
    rig_direction = rig["direction"]
    mirror_x = False
    if rig_direction != direction:
        mirror_source = (
            profile.data.get("mirrorDirections", {}).get(direction)
            if profile is not None
            else None
        )
        if mirror_source != rig_direction:
            raise RigError(
                f"compiled rig direction {rig_direction!r} cannot be relabeled as "
                f"{direction!r}; build direction-specific art or use a profile mirror direction"
            )
        mirror_x = True
    if deploy_dir is not None and rig["source"]["quality"] == "coarse":
        raise RigError(
            "coarse single-reference rigs cannot deploy into a game project; "
            "approve separated production parts first"
        )
    raw_output = safe_output_child(work, "raw", direction, label="raw animation output")
    raw = render_animation_set(
        rig,
        raw_output,
        clips,
        frame_counts,
        resample=resample,
        mirror_x=mirror_x,
        timing_source=f"profile:{profile.id}" if profile is not None else "rig",
    )
    for stage, label in (
        ("normalized", "normalized animation output"),
        ("reports", "validation report output"),
    ):
        previous_stage = safe_output_child(work, stage, direction, label=label)
        reset_output_directory(previous_stage, label=label)
    previous_exports = safe_output_child(work, "exports", label="export output")
    reset_output_directory(previous_exports, label="export output")
    result: dict[str, Any] = {
        "ok": True,
        "provider": "local-cutout",
        "quality": rig["source"]["quality"],
        "character": character,
        "direction": direction,
        "rig": str(source),
        "frameCounts": frame_counts,
        "raw": raw,
        "extraction": extraction,
        "profile": profile.id if profile else None,
        "tier": tier,
        "normalized": {},
        "validation": {},
        "exports": {},
    }
    if profile is None:
        manifest = safe_output_child(work, "character-manifest.json", label="character manifest")
        result["manifest"] = str(manifest)
        _write_json_atomic(manifest, result)
        return result

    palette = _character_palette(raw, profile)
    failures: list[str] = []
    for clip in clips:
        normalized_dir = safe_output_child(
            work,
            "normalized",
            direction,
            clip,
            label="normalized animation output",
        )
        normalized = ingest_frames(
            profile,
            raw["clips"][clip]["directory"],
            normalized_dir,
            tier,
            clip,
            direction,
            placement_mode="shared-motion",
            palette_override=palette,
            source_anchor=raw["motionAnchor"],
            source_bounds=raw["contentBounds"],
        )
        report_path = safe_output_child(
            work,
            "reports",
            direction,
            f"{clip}.json",
            label="validation report",
        )
        validation = validate_frames(
            profile,
            normalized_dir,
            tier,
            clip,
            report_path,
            placement_mode="shared-motion",
        )
        result["normalized"][clip] = normalized
        result["validation"][clip] = validation
        if not validation["ok"]:
            failures.append(clip)
    if failures:
        result["ok"] = False
        result["stage"] = "validate"
        result["failedClips"] = failures
        manifest = safe_output_child(work, "character-manifest.json", label="character manifest")
        result["manifest"] = str(manifest)
        _write_json_atomic(manifest, result)
        return result

    exports_dir = safe_output_child(work, "exports", label="export output")
    exports_dir.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        extension = _export_extension(profile)
        export_path = safe_output_child(
            exports_dir,
            f"{character}_{direction}_{clip}{extension}",
            label="engine export",
        )
        prefix = None
        if resource_prefix:
            prefix = join_resource_prefix(resource_prefix, character, direction, clip)
        result["exports"][clip] = export_assets(
            profile,
            result["normalized"][clip]["output"],
            export_path,
            character,
            tier,
            clip,
            direction,
            prefix,
            None,
        )
    if deploy_dir is not None:
        assert profile is not None and resource_prefix is not None
        manifest = safe_output_child(work, "character-manifest.json", label="character manifest")

        def finalize_deployment(deployment: dict[str, Any]) -> None:
            result["deployment"] = deployment
            for deployed_clip in clips:
                clip_deployment = deployment["clips"][deployed_clip]
                result["exports"][deployed_clip].update(
                    {
                        "deploymentMode": "explicit",
                        "deployDir": clip_deployment["deployDir"],
                        "artifactRoot": str(profile.project_root),
                        "localReferencePaths": clip_deployment["localReferencePaths"],
                        "verifiedReferences": clip_deployment["verifiedReferences"],
                        "transactional": True,
                        "directionManifest": deployment["manifest"],
                    }
                )
            result["manifest"] = str(manifest)
            _write_json_atomic(manifest, result)

        deploy_animation_direction(
            profile,
            {
                clip: frame_paths(result["normalized"][clip]["output"])
                for clip in clips
            },
            character,
            direction,
            resource_prefix,
            deploy_dir,
            finalize_deployment,
        )
        return result
    manifest = safe_output_child(work, "character-manifest.json", label="character manifest")
    result["manifest"] = str(manifest)
    _write_json_atomic(manifest, result)
    return result


def run_animation_spec(
    spec_path: str | Path,
    work_override: str | Path | None = None,
) -> dict[str, Any]:
    spec, path = load_animation_spec(spec_path)
    work = (
        Path(work_override).expanduser().resolve()
        if work_override
        else safe_output_child(
            path.parent,
            "build",
            spec["id"],
            label="AnimationSpec build output",
        )
    )
    rig = (path.parent / spec["rig"]).resolve()
    try:
        rig.relative_to(path.parent)
    except ValueError as exc:
        raise RigError("AnimationSpec rig path escapes the spec directory") from exc
    clips = list(spec["clips"])
    frames = {name: int(config["frames"]) for name, config in spec["clips"].items()}
    profile_name = spec.get("profile")
    if profile_name:
        profile_reference = Path(profile_name)
        looks_like_path = (
            profile_reference.suffix.lower() == ".json"
            or len(profile_reference.parts) > 1
            or profile_name.startswith(".")
        )
        if looks_like_path:
            if profile_reference.is_absolute() or ".." in profile_reference.parts:
                raise RigError("AnimationSpec profile path must stay inside the spec directory")
            resolved_profile = (path.parent / profile_reference).resolve()
            try:
                resolved_profile.relative_to(path.parent)
            except ValueError as exc:
                raise RigError("AnimationSpec profile path escapes the spec directory") from exc
            profile_name = str(resolved_profile)
    result = run_local_animation(
        work_dir=work,
        character=spec["character"],
        direction=spec.get("direction", "east"),
        clips=clips,
        frame_overrides=frames,
        rig_path=rig,
        resample=spec["render"].get("resample", "bicubic"),
        profile_name=profile_name,
        tier=spec.get("tier"),
    )
    result["animationSpec"] = {
        "id": spec["id"],
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if result.get("manifest"):
        _write_json_atomic(Path(result["manifest"]), result)
    return result
