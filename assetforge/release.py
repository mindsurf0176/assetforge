from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from . import __version__
from .frames import frame_paths, select_requested_animation_paths
from .json_utils import strict_json_loads
from .profile import Profile, validate_profile_data
from .validation import validate_frames


RELEASE_SCHEMA_VERSION = 1


def _validate_release_manifest_schema(data: dict[str, Any]) -> None:
    schema = json.loads(
        files("assetforge.schemas")
        .joinpath("release-manifest.schema.json")
        .read_text(encoding="utf-8")
    )
    errors = sorted(
        Draft202012Validator(schema).iter_errors(data),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.absolute_path)) or "$"
        raise ValueError(f"release manifest schema failed at {location}: {error.message}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_fingerprint(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _safe_relative(root: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a relative path inside the release: {value!r}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the release root: {value!r}") from exc
    return candidate


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _clip_input(root: Path, clip: str, known_clips: list[str]) -> tuple[Path, list[Path], str]:
    candidates = (
        (root / clip / "frames", "clip-frames"),
        (root / clip, "clip"),
    )
    for candidate, layout in candidates:
        if candidate.is_symlink():
            raise ValueError(f"release input crosses a symbolic link: {candidate}")
        if not candidate.is_dir():
            continue
        try:
            paths = frame_paths(candidate)
        except ValueError:
            continue
        return candidate, paths, layout

    if root.is_symlink():
        raise ValueError(f"release input is a symbolic link: {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"release input directory not found: {root}")
    paths = frame_paths(root)
    selected = select_requested_animation_paths(paths, clip, known_clips)
    if not selected or not any(
        path.stem == clip or path.stem.startswith(f"{clip}_") for path in selected
    ):
        raise ValueError(f"release input has no frames for animation {clip!r}: {root}")
    return root, selected, "flat"


def _require_transparent_pngs(paths: list[Path]) -> None:
    from PIL import Image

    for path in paths:
        try:
            with Image.open(path) as image:
                if "A" not in image.getbands():
                    raise ValueError(f"{path.name}: production release frames must have an alpha channel")
                alpha_min, _ = image.getchannel("A").getextrema()
                if alpha_min != 0:
                    raise ValueError(f"{path.name}: production release frame has no transparent pixels")
                image.load()
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"unable to read release frame {path}: {exc}") from exc


def _stage_swap(stage: Path, output: Path, overwrite: bool) -> None:
    if output.is_symlink():
        raise ValueError(f"release output must not be a symbolic link: {output}")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"release output already exists: {output}; pass --overwrite to replace it"
        )

    backup: Path | None = None
    if output.exists():
        backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.previous-", dir=output.parent))
        backup.rmdir()
        os.replace(output, backup)
    try:
        os.replace(stage, output)
    except BaseException:
        if backup is not None and not output.exists():
            os.replace(backup, output)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def package_release(
    profile: Profile,
    input_root: str | Path,
    output: str | Path,
    *,
    character: str,
    direction: str,
    tier: str,
    clips: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate and package one character direction into a portable release tree."""

    root = Path(input_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("release output must not be inside the release input")
    if not character or not direction:
        raise ValueError("release character and direction are required")
    profile.tier(tier)

    known_clips = list(profile.data.get("animations", {}).keys())
    selected_clips = clips or known_clips
    if not selected_clips:
        raise ValueError(f"profile {profile.id!r} defines no animations")
    unknown = sorted(set(selected_clips) - set(known_clips))
    if unknown:
        raise ValueError(f"release clips are not defined by profile: {', '.join(unknown)}")

    clip_records: dict[str, dict[str, Any]] = {}
    for clip in selected_clips:
        source_dir, paths, layout = _clip_input(root, clip, known_clips)
        _require_transparent_pngs(paths)
        validation = validate_frames(
            profile,
            source_dir,
            tier,
            clip,
            placement_mode="shared-motion",
        )
        if not validation["ok"]:
            raise ValueError(
                f"release validation failed for {clip!r}: "
                + "; ".join(validation["errors"])
            )
        contract = profile.animation(clip)
        clip_records[clip] = {
            "source": source_dir,
            "paths": paths,
            "layout": layout,
            "fps": float(contract["fps"]),
            "loop": bool(contract["loop"]),
            "validation": {
                "frameCount": validation["frameCount"],
                "heightDriftRatio": validation["heightDriftRatio"],
                "widthDriftRatio": validation["widthDriftRatio"],
                "warnings": list(validation["warnings"]),
                "transparentHoles": list(validation["transparentHoles"]),
            },
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        profile_target = stage / "profile.json"
        shutil.copy2(profile.path, profile_target)
        profile_sha256 = _sha256(profile_target)
        files: list[dict[str, Any]] = []
        packaged_clips: dict[str, dict[str, Any]] = {}
        for clip in selected_clips:
            record = clip_records[clip]
            clip_target = stage / "frames" / clip
            clip_target.mkdir(parents=True, exist_ok=True)
            packaged_frames: list[dict[str, str]] = []
            for index, source in enumerate(record["paths"]):
                target = clip_target / f"{clip}_{index:02d}.png"
                shutil.copy2(source, target)
                digest = _sha256(target)
                if digest != _sha256(source):
                    raise RuntimeError(f"release frame copy verification failed: {source}")
                relative = target.relative_to(stage).as_posix()
                item = {"file": relative, "sha256": digest}
                packaged_frames.append(item)
                files.append(item)
            packaged_clips[clip] = {
                "fps": record["fps"],
                "loop": record["loop"],
                "frameCount": len(packaged_frames),
                "frames": packaged_frames,
                "validation": record["validation"],
            }

        manifest: dict[str, Any] = {
            "$schema": "urn:assetforge:release-manifest:schema:1",
            "schemaVersion": RELEASE_SCHEMA_VERSION,
            "kind": "sprite-release",
            "assetforgeVersion": __version__,
            "character": character,
            "direction": direction,
            "profile": {
                "id": profile.id,
                "file": "profile.json",
                "sha256": profile_sha256,
                "fingerprint": profile.fingerprint,
            },
            "tier": tier,
            "source": {"layout": "clip-directories-or-flat", "clipCount": len(selected_clips)},
            "clips": packaged_clips,
            "files": files,
        }
        manifest["releaseFingerprint"] = _canonical_fingerprint(manifest)
        _validate_release_manifest_schema(manifest)
        _write_json(stage / "release.json", manifest)
        _stage_swap(stage, destination, overwrite)
        stage = Path()
    except BaseException:
        if stage != Path():
            shutil.rmtree(stage, ignore_errors=True)
        raise

    return {
        "ok": True,
        "output": str(destination),
        "manifest": str(destination / "release.json"),
        "character": character,
        "direction": direction,
        "clips": selected_clips,
        "fileCount": len(files),
        "releaseFingerprint": manifest["releaseFingerprint"],
        "transactional": True,
    }


def verify_release(manifest_path: str | Path) -> dict[str, Any]:
    """Verify a packaged release without trusting its recorded file hashes."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    if manifest_file.is_symlink() or not manifest_file.is_file():
        raise FileNotFoundError(f"release manifest not found: {manifest_file}")
    root = manifest_file.parent
    data = strict_json_loads(manifest_file.read_text(encoding="utf-8"))
    _validate_release_manifest_schema(data)
    if data.get("schemaVersion") != RELEASE_SCHEMA_VERSION or data.get("kind") != "sprite-release":
        raise ValueError("unsupported AssetForge release manifest")

    recorded_fingerprint = data.get("releaseFingerprint")
    unsigned = dict(data)
    unsigned.pop("releaseFingerprint", None)
    actual_fingerprint = _canonical_fingerprint(unsigned)
    errors: list[str] = []
    if recorded_fingerprint != actual_fingerprint:
        errors.append(
            f"release fingerprint mismatch: recorded={recorded_fingerprint!r}, "
            f"actual={actual_fingerprint!r}"
        )

    profile_record = data.get("profile")
    if not isinstance(profile_record, dict):
        errors.append("release profile record is missing")
        profile = None
    else:
        try:
            profile_file = _safe_relative(root, str(profile_record["file"]), "profile file")
            if not profile_file.is_file() or profile_file.is_symlink():
                raise ValueError("profile file is missing or a symbolic link")
            if _sha256(profile_file) != profile_record.get("sha256"):
                raise ValueError("profile file hash mismatch")
            profile_data = strict_json_loads(profile_file.read_text(encoding="utf-8"))
            validate_profile_data(profile_data, str(profile_file))
            profile = Profile(path=profile_file, data=profile_data)
            if profile.id != profile_record.get("id"):
                raise ValueError("profile id mismatch")
            if profile.fingerprint != profile_record.get("fingerprint"):
                raise ValueError("profile fingerprint mismatch")
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"profile verification failed: {exc}")
            profile = None

    verified_files = 0
    for item in data.get("files", []):
        try:
            file_path = _safe_relative(root, str(item["file"]), "release file")
            if file_path.is_symlink() or not file_path.is_file():
                raise ValueError("file is missing or a symbolic link")
            if _sha256(file_path) != item.get("sha256"):
                raise ValueError("file hash mismatch")
            verified_files += 1
        except (KeyError, OSError, ValueError) as exc:
            errors.append(f"{item.get('file', '<unknown>')}: {exc}")

    if profile is not None:
        for clip, record in data.get("clips", {}).items():
            try:
                clip_dir = _safe_relative(root, f"frames/{clip}", "clip directory")
                validation = validate_frames(
                    profile,
                    clip_dir,
                    str(data["tier"]),
                    clip,
                    placement_mode="shared-motion",
                )
                if not validation["ok"]:
                    errors.extend(f"{clip}: {message}" for message in validation["errors"])
                if validation["frameCount"] != record["frameCount"]:
                    errors.append(f"{clip}: manifest frame count does not match files")
            except (KeyError, OSError, ValueError) as exc:
                errors.append(f"{clip}: release validation failed: {exc}")

    return {
        "ok": not errors,
        "manifest": str(manifest_file),
        "character": data.get("character"),
        "direction": data.get("direction"),
        "releaseFingerprint": actual_fingerprint,
        "verifiedFiles": verified_files,
        "errors": errors,
    }


__all__ = ["package_release", "verify_release"]
