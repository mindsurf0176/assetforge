from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .frames import frame_paths
from .profile import Profile


def _animation_frame_paths(input_dir: str | Path, animation: str) -> list[Path]:
    paths = [
        path
        for path in frame_paths(input_dir)
        if path.stem == animation or path.stem.startswith(f"{animation}_")
    ]
    if not paths:
        raise ValueError(
            f"no PNG frames found for animation {animation!r} in "
            f"{Path(input_dir).expanduser().resolve()}"
        )
    return paths


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_child(root: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay inside its asset root: {value!r}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes its asset root: {value!r}")
    return resolved


def _reference_directory(prefix: str, web_root: Path, godot_root: Path) -> Path:
    if prefix.startswith("res://"):
        return _safe_child(godot_root, prefix.removeprefix("res://"), "resource prefix")
    if "://" in prefix or prefix.startswith("/"):
        raise ValueError(
            f"resource prefix {prefix!r} cannot be mapped to local deployed files; "
            "use a relative web prefix or res:// Godot prefix"
        )
    return _safe_child(web_root, prefix.removeprefix("./"), "resource prefix")


def _deploy_animation_frames(
    profile: Profile,
    paths: list[Path],
    output: str | Path,
    resource_prefix: str,
    deploy_dir: str | Path | None,
) -> dict[str, Any]:
    """Copy normalized frames to the exact location referenced by the export.

    Without an explicit deploy directory, an isolated artifact tree is rooted at
    the export file's parent. This keeps builds self-contained and avoids writing
    into either game's source tree. Passing ``deploy_dir`` is the explicit opt-in
    for a runtime/project deployment and must agree with ``resource_prefix``.
    """

    target = Path(output).expanduser().resolve()
    output_root = target.parent
    project_root = profile.project_root
    if deploy_dir is None:
        destination = _reference_directory(resource_prefix, output_root, output_root)
        if destination.is_relative_to(project_root):
            raise ValueError(
                f"implicit export would modify project assets at {destination}; "
                "pass --deploy-dir with that exact path to authorize deployment, "
                "or export outside the project for an isolated artifact bundle"
            )
        deployment_mode = "artifact"
        artifact_root = output_root
    else:
        destination = Path(deploy_dir).expanduser().resolve()
        referenced_destination = _reference_directory(
            resource_prefix,
            project_root,
            project_root,
        )
        if destination != referenced_destination:
            raise ValueError(
                f"deploy directory {destination} does not match resource prefix "
                f"{resource_prefix!r}, which resolves to {referenced_destination}"
            )
        deployment_mode = "explicit"
        artifact_root = project_root

    destination.mkdir(parents=True, exist_ok=True)
    deployed: list[Path] = []
    for source in paths:
        deployed_path = destination / source.name
        if source.resolve() != deployed_path.resolve():
            shutil.copy2(source, deployed_path)
        if not deployed_path.is_file() or _file_digest(deployed_path) != _file_digest(source):
            raise RuntimeError(
                f"deployed frame verification failed: {source} -> {deployed_path}"
            )
        deployed.append(deployed_path)

    return {
        "deploymentMode": deployment_mode,
        "deployDir": str(destination),
        "artifactRoot": str(artifact_root),
        "localReferencePaths": [str(path) for path in deployed],
        "verifiedReferences": len(deployed),
    }


def export_web_registry(
    profile: Profile,
    input_dir: str | Path,
    output: str | Path,
    character: str,
    tier: str,
    animation: str,
    direction: str,
    resource_prefix: str | None = None,
    deploy_dir: str | Path | None = None,
) -> dict[str, Any]:
    paths = _animation_frame_paths(input_dir, animation)
    prefix = (resource_prefix or profile.data["export"].get("resourcePrefix", "./assets/generated")).rstrip("/")
    deployment = _deploy_animation_frames(profile, paths, output, prefix, deploy_dir)
    registry = {
        "schemaVersion": 1,
        "profile": profile.id,
        "profileFingerprint": profile.fingerprint,
        "characters": {
            character: {
                "tiers": {
                    tier: {
                        "animations": {
                            animation: {
                                "directions": {
                                    direction: [f"{prefix}/{path.name}" for path in paths]
                                }
                            }
                        }
                    }
                }
            }
        },
    }
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "format": "web-registry",
        "output": str(target),
        "frames": len(paths),
        "resourcePrefix": prefix,
        **deployment,
    }


def export_godot_spriteframes(
    profile: Profile,
    input_dir: str | Path,
    output: str | Path,
    animation: str,
    resource_prefix: str | None = None,
    deploy_dir: str | Path | None = None,
) -> dict[str, Any]:
    paths = _animation_frame_paths(input_dir, animation)
    config = profile.animation(animation)
    prefix = (resource_prefix or profile.data["export"].get("resourcePrefix", "res://assets/generated")).rstrip("/")
    deployment = _deploy_animation_frames(profile, paths, output, prefix, deploy_dir)
    ext = []
    frames = []
    for index, path in enumerate(paths, start=1):
        ext.append(f'[ext_resource type="Texture2D" path="{prefix}/{path.name}" id="{index}"]')
        frames.append(f'{{"duration": 1.0, "texture": ExtResource("{index}")}}')
    loop = "true" if config.get("loop", False) else "false"
    speed = float(config.get("fps", 8))
    resource = (
        f'[gd_resource type="SpriteFrames" load_steps={len(paths) + 1} format=3]\n\n'
        + "\n".join(ext)
        + "\n\n[resource]\n"
        + "animations = [{\n"
        + f'"frames": [{", ".join(frames)}],\n'
        + f'"loop": {loop},\n"name": &"{animation}",\n"speed": {speed}\n'
        + "}]\n"
    )
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(resource, encoding="utf-8")
    return {
        "ok": True,
        "format": "godot-spriteframes",
        "output": str(target),
        "frames": len(paths),
        "resourcePrefix": prefix,
        **deployment,
    }


def export_assets(
    profile: Profile,
    input_dir: str | Path,
    output: str | Path,
    character: str,
    tier: str,
    animation: str,
    direction: str,
    resource_prefix: str | None = None,
    deploy_dir: str | Path | None = None,
) -> dict[str, Any]:
    engine = profile.data["export"].get("engine")
    if engine == "web":
        return export_web_registry(
            profile,
            input_dir,
            output,
            character,
            tier,
            animation,
            direction,
            resource_prefix,
            deploy_dir,
        )
    if engine == "godot":
        return export_godot_spriteframes(
            profile,
            input_dir,
            output,
            animation,
            resource_prefix,
            deploy_dir,
        )
    raise ValueError(f"profile {profile.id!r} has unsupported export engine {engine!r}")
