from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None
    import msvcrt

from .frames import frame_paths
from .path_safety import safe_output_child
from .profile import Profile


_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@contextmanager
def _deployment_lock(destination: Path):
    """Serialize one live destination across threads and local processes."""

    key = str(destination.resolve())
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(key, threading.Lock())
    if not thread_lock.acquire(blocking=False):
        raise RuntimeError(f"deployment already in progress: {destination}")
    handle = None
    process_lock_acquired = False
    try:
        lock_root = Path(tempfile.gettempdir()) / "assetforge-deployment-locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_name = hashlib.sha256(key.encode("utf-8")).hexdigest() + ".lock"
        handle = (lock_root / lock_name).open("a+")
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                process_lock_acquired = True
            except BlockingIOError as exc:
                raise RuntimeError(f"deployment already in progress: {destination}") from exc
        else:  # pragma: no cover - exercised on Windows
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                process_lock_acquired = True
            except OSError as exc:
                raise RuntimeError(f"deployment already in progress: {destination}") from exc
        yield
    finally:
        try:
            if handle is not None:
                try:
                    if process_lock_acquired and fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    elif process_lock_acquired:  # pragma: no cover - exercised on Windows
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                finally:
                    handle.close()
        finally:
            thread_lock.release()


def normalize_resource_prefix(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("resource prefix must be a non-empty string")
    if "\\" in value:
        raise ValueError(f"resource prefix must use forward slashes: {value!r}")
    if value.startswith("res://"):
        remainder = value.removeprefix("res://").strip("/")
        return f"res://{remainder}"
    normalized = value.rstrip("/")
    if not normalized:
        raise ValueError(f"resource prefix is not a deployable relative path: {value!r}")
    return normalized


def join_resource_prefix(value: str, *parts: str) -> str:
    base = normalize_resource_prefix(value)
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    if not suffix:
        return base
    separator = "" if base == "res://" else "/"
    return f"{base}{separator}{suffix}"


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


def _reject_managed_direction_child(destination: Path, project_root: Path) -> None:
    """Keep direct per-clip export out of whole-direction managed trees."""

    current = destination.resolve()
    root = project_root.resolve()
    while current.is_relative_to(root):
        marker = current / ".assetforge-deployment.json"
        if marker.exists() or marker.is_symlink():
            raise ValueError(
                f"direct export target is managed by a direction transaction at {current}; "
                "use 'assetforge animate --deploy-dir' to synchronize that direction"
            )
        if current == root:
            break
        current = current.parent


def _deploy_animation_frames(
    profile: Profile,
    paths: list[Path],
    output: str | Path,
    animation: str,
    resource_prefix: str,
    deploy_dir: str | Path | None,
    descriptor: str,
) -> dict[str, Any]:
    """Atomically publish frames and their descriptor as one rollback-safe unit.

    Without an explicit deploy directory, an isolated artifact tree is rooted at
    the export file's parent. This keeps builds self-contained and avoids writing
    into either game's source tree. Passing ``deploy_dir`` is the explicit opt-in
    for a runtime/project deployment and must agree with ``resource_prefix``.
    """

    unresolved_target = Path(output).expanduser()
    if unresolved_target.is_symlink():
        raise ValueError(f"export descriptor is a symbolic link: {unresolved_target}")
    target = unresolved_target.resolve()
    if target.exists() and not target.is_file():
        raise ValueError(f"export descriptor is not a file path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    output_root = target.parent
    project_root = profile.project_root
    if deploy_dir is None:
        destination = _reference_directory(resource_prefix, output_root, output_root)
        project_destination = _reference_directory(
            resource_prefix,
            project_root,
            project_root,
        )
        if destination == project_destination:
            raise ValueError(
                f"implicit export would modify project assets at {destination}; "
                "pass --deploy-dir with that exact path to authorize deployment, "
                "or export outside the project for an isolated artifact bundle"
            )
        deployment_mode = "artifact"
        artifact_root = output_root
    else:
        unresolved_destination = Path(deploy_dir).expanduser()
        if unresolved_destination.is_symlink():
            raise ValueError(f"deploy directory is a symbolic link: {unresolved_destination}")
        destination = unresolved_destination.resolve()
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
    lock_destination = project_root if deploy_dir is not None else destination
    with _deployment_lock(lock_destination):
        return _deploy_clip_transaction(
            paths,
            target,
            destination,
            animation,
            descriptor,
            deployment_mode,
            artifact_root,
            project_root if deploy_dir is not None else None,
        )


def _deploy_clip_transaction(
    paths: list[Path],
    target: Path,
    destination: Path,
    animation: str,
    descriptor: str,
    deployment_mode: str,
    artifact_root: Path,
    managed_project_root: Path | None,
) -> dict[str, Any]:
    if managed_project_root is not None:
        _reject_managed_direction_child(destination, managed_project_root)
    destination_created = not destination.exists()
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir():
        raise ValueError(f"deploy destination is not a directory: {destination}")
    incoming_names = {path.name for path in paths}
    if len(incoming_names) != len(paths):
        raise ValueError("deployment source frame names must be unique")
    existing_animation_frames: dict[str, Path] = {}
    for child in destination.iterdir():
        matches_animation = child.suffix.lower() == ".png" and (
            child.stem == animation or child.stem.startswith(f"{animation}_")
        )
        if not matches_animation:
            continue
        if child.is_symlink():
            raise ValueError(f"deployed frame crosses a symbolic link: {child}")
        if not child.is_file():
            raise ValueError(f"deployed frame destination is not a file: {child}")
        existing_animation_frames[child.name] = child

    deployed = [
        safe_output_child(destination, source.name, label="deployed frame")
        for source in paths
    ]
    frame_stage: Path | None = None
    descriptor_stage: Path | None = None
    frame_backup: Path | None = None
    frame_next: Path | None = None
    descriptor_backup: Path | None = None
    commit_started = False
    preserve_recovery = False
    try:
        frame_stage = Path(
            tempfile.mkdtemp(prefix=".assetforge-deploy-", dir=destination.parent)
        )
        descriptor_stage = Path(
            tempfile.mkdtemp(prefix=".assetforge-descriptor-", dir=target.parent)
        )
        staged_descriptor = descriptor_stage / target.name
        descriptor_backup = descriptor_stage / f"{target.name}.previous"
        frame_backup = frame_stage / "previous"
        frame_next = frame_stage / "next"
        frame_backup.mkdir()
        frame_next.mkdir()
        source_digests = {source.name: _file_digest(source) for source in paths}
        for source in paths:
            staged = frame_next / source.name
            shutil.copy2(source, staged)
            if _file_digest(staged) != source_digests[source.name]:
                raise RuntimeError(f"frame staging verification failed: {source} -> {staged}")
        for name, existing in existing_animation_frames.items():
            backup = frame_backup / name
            shutil.copy2(existing, backup)
            if _file_digest(backup) != _file_digest(existing):
                raise RuntimeError(f"frame backup verification failed: {existing} -> {backup}")
        if target.exists():
            shutil.copy2(target, descriptor_backup)
            if _file_digest(descriptor_backup) != _file_digest(target):
                raise RuntimeError(
                    f"descriptor backup verification failed: {target} -> {descriptor_backup}"
                )
        staged_descriptor.write_text(descriptor, encoding="utf-8")

        commit_started = True
        for deployed_path in deployed:
            os.replace(frame_next / deployed_path.name, deployed_path)
        for name, stale in existing_animation_frames.items():
            if name not in incoming_names:
                stale.unlink()
        os.replace(staged_descriptor, target)
        for source, deployed_path in zip(paths, deployed):
            if (
                not deployed_path.is_file()
                or _file_digest(deployed_path) != source_digests[source.name]
            ):
                raise RuntimeError(
                    f"deployed frame verification failed: {source} -> {deployed_path}"
                )
    except BaseException as original_error:
        rollback_error: BaseException | None = None
        if commit_started:
            try:
                assert frame_backup is not None
                for deployed_path in deployed:
                    backup = frame_backup / deployed_path.name
                    if backup.exists():
                        os.replace(backup, deployed_path)
                    elif deployed_path.name not in existing_animation_frames and (
                        deployed_path.exists() or deployed_path.is_symlink()
                    ):
                        deployed_path.unlink()
                for name, stale in existing_animation_frames.items():
                    if name in incoming_names:
                        continue
                    backup = frame_backup / name
                    if backup.exists():
                        os.replace(backup, stale)
                if descriptor_backup is not None and descriptor_backup.exists():
                    os.replace(descriptor_backup, target)
                elif target.exists() or target.is_symlink():
                    target.unlink()
            except BaseException as exc:
                rollback_error = exc
        if rollback_error is not None:
            preserve_recovery = True
            raise RuntimeError(
                "export failed and rollback also failed; recovery copies are preserved at "
                f"frames={frame_stage}, descriptor={descriptor_stage}: {rollback_error}"
            ) from original_error
        raise
    finally:
        if not preserve_recovery:
            if frame_stage is not None:
                shutil.rmtree(frame_stage, ignore_errors=True)
            if descriptor_stage is not None:
                shutil.rmtree(descriptor_stage, ignore_errors=True)
        if destination_created:
            try:
                destination.rmdir()
            except OSError:
                pass

    return {
        "deploymentMode": deployment_mode,
        "deployDir": str(destination),
        "artifactRoot": str(artifact_root),
        "localReferencePaths": [str(path) for path in deployed],
        "verifiedReferences": len(deployed),
        "transactional": True,
    }


def deploy_animation_direction(
    profile: Profile,
    clip_sources: dict[str, list[Path]],
    character: str,
    direction: str,
    resource_prefix: str,
    deploy_dir: str | Path,
    finalize: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Synchronize one character direction as a single rollback-safe tree."""

    if not clip_sources:
        raise ValueError("direction deployment requires at least one clip")
    prefix = join_resource_prefix(resource_prefix, character, direction)
    raw_root = Path(deploy_dir).expanduser()
    if raw_root.is_symlink():
        raise ValueError(f"deploy root is a symbolic link: {raw_root}")
    root = raw_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = safe_output_child(
        root,
        character,
        direction,
        label="direction deployment",
    )
    referenced_destination = _reference_directory(
        prefix,
        profile.project_root,
        profile.project_root,
    )
    if destination != referenced_destination:
        raise ValueError(
            f"direction deploy directory {destination} does not match resource prefix "
            f"{prefix!r}, which resolves to {referenced_destination}"
        )

    with _deployment_lock(profile.project_root):
        return _deploy_direction_transaction(
            clip_sources,
            character,
            direction,
            prefix,
            destination,
            finalize,
        )


def _deploy_direction_transaction(
    clip_sources: dict[str, list[Path]],
    character: str,
    direction: str,
    prefix: str,
    destination: Path,
    finalize: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:

    marker_name = ".assetforge-deployment.json"
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"direction deployment is not a directory: {destination}")
        entries = list(destination.iterdir())
        if entries:
            marker = destination / marker_name
            if marker.is_symlink() or not marker.is_file():
                raise ValueError(
                    "direction deployment is non-empty and is not marked as "
                    f"AssetForge-owned: {destination}"
                )
            try:
                marker_data = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid direction deployment marker: {marker}") from exc
            expected_identity = {
                "owner": "assetforge",
                "schemaVersion": 1,
                "kind": "animation-direction",
                "character": character,
                "direction": direction,
            }
            if any(marker_data.get(key) != value for key, value in expected_identity.items()):
                raise ValueError(f"invalid direction deployment marker: {marker}")
            for descendant in destination.rglob("*"):
                if descendant.is_symlink():
                    raise ValueError(
                        f"direction deployment contains a symbolic link: {descendant}"
                    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage: Path | None = None
    backup: Path | None = None
    failed_tree: Path | None = None
    destination_existed = destination.exists()
    manifest_clips: dict[str, list[dict[str, str]]] = {}
    commit_started = False
    preserve_recovery = False
    try:
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{direction}.assetforge-stage-",
                dir=destination.parent,
            )
        )
        backup = Path(
            tempfile.mkdtemp(
                prefix=f".{direction}.assetforge-backup-",
                dir=destination.parent,
            )
        )
        backup.rmdir()
        failed_tree = Path(
            tempfile.mkdtemp(
                prefix=f".{direction}.assetforge-failed-",
                dir=destination.parent,
            )
        )
        failed_tree.rmdir()
        for clip_name, sources in sorted(clip_sources.items()):
            if not sources:
                raise ValueError(f"clip {clip_name!r} has no frames to deploy")
            clip_dir = safe_output_child(stage, clip_name, label="staged clip deployment")
            clip_dir.mkdir()
            manifest_clips[clip_name] = []
            for source in sources:
                source_path = Path(source).expanduser().resolve()
                if not source_path.is_file():
                    raise FileNotFoundError(f"deployment source frame not found: {source_path}")
                target = safe_output_child(clip_dir, source_path.name, label="staged frame")
                shutil.copy2(source_path, target)
                digest = _file_digest(source_path)
                if _file_digest(target) != digest:
                    raise RuntimeError(
                        f"direction staging verification failed: {source_path} -> {target}"
                    )
                manifest_clips[clip_name].append(
                    {"file": f"{clip_name}/{source_path.name}", "sha256": digest}
                )
        marker_data = {
            "owner": "assetforge",
            "schemaVersion": 1,
            "kind": "animation-direction",
            "character": character,
            "direction": direction,
            "clips": manifest_clips,
        }
        (stage / marker_name).write_text(
            json.dumps(marker_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        clips_result: dict[str, Any] = {}
        for clip_name, entries in manifest_clips.items():
            paths = [destination / entry["file"] for entry in entries]
            clips_result[clip_name] = {
                "deployDir": str(destination / clip_name),
                "localReferencePaths": [str(path) for path in paths],
                "verifiedReferences": len(paths),
            }
        deployment_result = {
            "ok": True,
            "deploymentMode": "explicit",
            "deployRoot": str(destination),
            "resourcePrefix": prefix,
            "manifest": str(destination / marker_name),
            "clips": clips_result,
            "transactional": True,
        }

        commit_started = True
        if destination_existed:
            os.replace(destination, backup)
        os.replace(stage, destination)
        for entries in manifest_clips.values():
            for entry in entries:
                deployed = safe_output_child(
                    destination,
                    *Path(entry["file"]).parts,
                    label="deployed direction frame",
                )
                if not deployed.is_file() or _file_digest(deployed) != entry["sha256"]:
                    raise RuntimeError(f"direction deployment verification failed: {deployed}")
        if finalize is not None:
            finalize(deployment_result)
    except BaseException as original_error:
        rollback_error: BaseException | None = None
        if commit_started:
            try:
                assert backup is not None and failed_tree is not None
                if backup.exists():
                    if destination.exists():
                        os.replace(destination, failed_tree)
                    os.replace(backup, destination)
                elif not destination_existed and destination.exists():
                    os.replace(destination, failed_tree)
            except BaseException as exc:
                rollback_error = exc
        if rollback_error is not None:
            preserve_recovery = True
            raise RuntimeError(
                "direction deployment failed and rollback also failed; recovery trees are "
                f"preserved at live={destination}, previous={backup}, "
                f"failed={failed_tree}, staged={stage}: {rollback_error}"
            ) from original_error
        raise
    finally:
        if not preserve_recovery:
            if stage is not None and stage.exists():
                shutil.rmtree(stage, ignore_errors=True)
            if backup is not None and backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            if failed_tree is not None and failed_tree.exists():
                shutil.rmtree(failed_tree, ignore_errors=True)

    return deployment_result


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
    prefix = normalize_resource_prefix(
        resource_prefix or profile.data["export"].get("resourcePrefix", "./assets/generated")
    )
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
                                    direction: [join_resource_prefix(prefix, path.name) for path in paths]
                                }
                            }
                        }
                    }
                }
            }
        },
    }
    descriptor = json.dumps(registry, ensure_ascii=False, indent=2) + "\n"
    deployment = _deploy_animation_frames(
        profile,
        paths,
        output,
        animation,
        prefix,
        deploy_dir,
        descriptor,
    )
    target = Path(output).expanduser().resolve()
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
    prefix = normalize_resource_prefix(
        resource_prefix or profile.data["export"].get("resourcePrefix", "res://assets/generated")
    )
    ext = []
    frames = []
    for index, path in enumerate(paths, start=1):
        ext.append(
            f'[ext_resource type="Texture2D" path="{join_resource_prefix(prefix, path.name)}" id="{index}"]'
        )
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
    deployment = _deploy_animation_frames(
        profile,
        paths,
        output,
        animation,
        prefix,
        deploy_dir,
        resource,
    )
    target = Path(output).expanduser().resolve()
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
