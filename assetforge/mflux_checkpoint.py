from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .safetensors_utils import has_lora_tensor_pairs, read_safetensors_header


_CHECKPOINT_MANIFEST = "checkpoint.json"
_MAX_CHECKPOINT_JSON_BYTES = 4 * 1024 * 1024
_MAX_LORA_BYTES = 4 * 1024 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_EXPECTED_MFLUX_VERSION = "0.18.0"
_EXPECTED_MODEL = "flux2-klein-base-4b"


def _clean_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if "\0" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must not contain NUL or line breaks")
    return value


def _relative_archive_path(value: Any, *, label: str) -> PurePosixPath:
    raw = _clean_text(value, label=label)
    if "\\" in raw:
        raise ValueError(f"{label} must use forward slashes")
    path = PurePosixPath(raw.rstrip("/"))
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
    ):
        raise ValueError(f"{label} must be a normalized relative path")
    return path


def _strict_json_loads(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"{label} contains non-finite number {token!r}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _lora_member_from_manifest(payload: bytes) -> PurePosixPath:
    checkpoint = _strict_json_loads(payload, label=_CHECKPOINT_MANIFEST)
    files = checkpoint.get("files")
    if not isinstance(files, dict):
        raise ValueError("checkpoint.json must contain a files object")
    if "lora_adapter" not in files:
        raise ValueError("checkpoint.json files must contain lora_adapter")
    member = _relative_archive_path(files["lora_adapter"], label="files.lora_adapter")
    if member.suffix != ".safetensors":
        raise ValueError("files.lora_adapter must name a .safetensors file")
    return member


def _member_collision_key(path: PurePosixPath) -> str:
    return unicodedata.normalize("NFC", path.as_posix()).casefold()


def _validate_zip_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    collision_keys: set[str] = set()
    checkpoint_members: list[str] = []
    for info in archive.infolist():
        path = _relative_archive_path(info.filename, label="ZIP member")
        normalized = path.as_posix()
        collision_key = _member_collision_key(path)
        if normalized in members or collision_key in collision_keys:
            raise ValueError(f"ZIP archive contains a duplicate member: {normalized}")
        members[normalized] = info
        collision_keys.add(collision_key)
        if path.name == _CHECKPOINT_MANIFEST:
            checkpoint_members.append(normalized)
        if info.flag_bits & 0x1:
            raise ValueError(f"encrypted ZIP members are not supported: {normalized}")

        unix_mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(unix_mode)
        if file_type == stat.S_IFLNK:
            raise ValueError(f"ZIP archive contains a symbolic link: {normalized}")
        if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ValueError(f"ZIP archive contains a special file: {normalized}")
    if checkpoint_members != [_CHECKPOINT_MANIFEST]:
        raise ValueError("ZIP archive must contain exactly one root checkpoint.json")
    return members


def _read_zip_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    limit: int,
    label: str,
) -> bytes:
    if info.is_dir():
        raise ValueError(f"{label} must be a regular file")
    if info.file_size > limit:
        raise ValueError(f"{label} exceeds the {limit}-byte safety limit")
    try:
        with archive.open(info, "r") as reader:
            payload = reader.read(limit + 1)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{label} could not be read from the ZIP archive") from exc
    if len(payload) > limit or len(payload) != info.file_size:
        raise ValueError(f"{label} has an invalid or oversized payload")
    return payload


def _inspect_zip_checkpoint(path: Path) -> tuple[str, bytes, Any]:
    before = path.stat()
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = _validate_zip_members(archive)
            checkpoint_payload = _read_zip_member(
                archive,
                members[_CHECKPOINT_MANIFEST],
                limit=_MAX_CHECKPOINT_JSON_BYTES,
                label=_CHECKPOINT_MANIFEST,
            )
            lora_member = _lora_member_from_manifest(checkpoint_payload).as_posix()
            matches = [info for name, info in members.items() if name == lora_member]
            if len(matches) != 1:
                raise ValueError(
                    "checkpoint.json files.lora_adapter must reference exactly one ZIP member"
                )
            info = matches[0]
            if info.is_dir() or info.file_size <= 0:
                raise ValueError("checkpoint LoRA adapter must be a non-empty regular file")
            if info.file_size > _MAX_LORA_BYTES:
                raise ValueError("checkpoint LoRA adapter exceeds the extraction safety limit")

            expected_size = info.file_size

            def copy_to(writer) -> None:
                written = 0
                try:
                    with zipfile.ZipFile(path, "r") as reopened:
                        reopened_members = _validate_zip_members(reopened)
                        reopened_info = reopened_members.get(lora_member)
                        if reopened_info is None or reopened_info.file_size != expected_size:
                            raise ValueError("MFLUX checkpoint changed before LoRA extraction")
                        with reopened.open(reopened_info, "r") as reader:
                            while True:
                                chunk = reader.read(_COPY_CHUNK_BYTES)
                                if not chunk:
                                    break
                                written += len(chunk)
                                if written > _MAX_LORA_BYTES:
                                    raise ValueError(
                                        "checkpoint LoRA adapter exceeds the extraction safety limit"
                                    )
                                writer.write(chunk)
                except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                    raise ValueError("checkpoint LoRA adapter could not be read") from exc
                after = path.stat()
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise ValueError("MFLUX checkpoint changed during LoRA extraction")
                if written != expected_size:
                    raise ValueError("checkpoint LoRA adapter size does not match its ZIP metadata")

            return lora_member, checkpoint_payload, copy_to
    except zipfile.BadZipFile as exc:
        raise ValueError(f"MFLUX checkpoint is not a valid ZIP archive: {path}") from exc


def _directory_entries(root: Path) -> None:
    collision_keys: set[str] = set()
    checkpoint_paths: list[Path] = []
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root)
        path = _relative_archive_path(relative.as_posix(), label="checkpoint directory entry")
        collision_key = _member_collision_key(path)
        if collision_key in collision_keys:
            raise ValueError(f"checkpoint directory contains a duplicate entry: {path}")
        collision_keys.add(collision_key)
        if candidate.is_symlink():
            raise ValueError(f"checkpoint directory contains a symbolic link: {candidate}")
        if not candidate.is_file() and not candidate.is_dir():
            raise ValueError(f"checkpoint directory contains a special file: {candidate}")
        if path.name == _CHECKPOINT_MANIFEST:
            checkpoint_paths.append(candidate)
    if checkpoint_paths != [root / _CHECKPOINT_MANIFEST]:
        raise ValueError("checkpoint directory must contain exactly one root checkpoint.json")


def _read_regular_file(path: Path, *, limit: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    size = path.stat().st_size
    if size > limit:
        raise ValueError(f"{label} exceeds the {limit}-byte safety limit")
    with path.open("rb") as reader:
        payload = reader.read(limit + 1)
    if len(payload) > limit or len(payload) != size:
        raise ValueError(f"{label} changed or exceeded its safety limit while being read")
    return payload


def _inspect_directory_checkpoint(path: Path) -> tuple[str, bytes, Any]:
    _directory_entries(path)
    checkpoint_payload = _read_regular_file(
        path / _CHECKPOINT_MANIFEST,
        limit=_MAX_CHECKPOINT_JSON_BYTES,
        label=_CHECKPOINT_MANIFEST,
    )
    lora_relative = _lora_member_from_manifest(checkpoint_payload)
    lora_path = path.joinpath(*lora_relative.parts)
    resolved_lora = lora_path.resolve()
    try:
        resolved_lora.relative_to(path)
    except ValueError as exc:
        raise ValueError("checkpoint LoRA adapter escapes the checkpoint directory") from exc
    if resolved_lora.is_symlink() or not resolved_lora.is_file():
        raise ValueError("checkpoint.json files.lora_adapter must reference one regular file")
    if resolved_lora.stat().st_size <= 0 or resolved_lora.stat().st_size > _MAX_LORA_BYTES:
        raise ValueError("checkpoint LoRA adapter has an invalid size")

    def copy_to(writer) -> None:
        before = resolved_lora.stat()
        with resolved_lora.open("rb") as reader:
            shutil.copyfileobj(reader, writer, length=_COPY_CHUNK_BYTES)
        after = resolved_lora.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("checkpoint LoRA adapter changed while it was being copied")

    return lora_relative.as_posix(), checkpoint_payload, copy_to


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_mflux_lora_adapter(
    checkpoint: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Extract and validate exactly the LoRA named by an MFLUX checkpoint manifest."""

    checkpoint_value = Path(checkpoint).expanduser()
    if checkpoint_value.is_symlink():
        raise ValueError(f"MFLUX checkpoint must not be a symbolic link: {checkpoint_value}")
    checkpoint_path = checkpoint_value.resolve()
    if checkpoint_path.is_dir():
        source_type = "directory"
        member, _, copy_to = _inspect_directory_checkpoint(checkpoint_path)
    elif checkpoint_path.is_file():
        source_type = "zip"
        member, _, copy_to = _inspect_zip_checkpoint(checkpoint_path)
    else:
        raise FileNotFoundError(f"MFLUX checkpoint not found: {checkpoint_path}")

    output_value = Path(output).expanduser()
    if output_value.suffix != ".safetensors":
        raise ValueError("LoRA output must use the .safetensors suffix")
    if output_value.is_symlink() or output_value.exists():
        raise FileExistsError(f"LoRA output already exists; overwrite is disabled: {output_value}")
    output_parent = output_value.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    output_path = output_parent / output_value.name
    if output_path.is_symlink() or output_path.exists():
        raise FileExistsError(f"LoRA output already exists; overwrite is disabled: {output_path}")

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as writer:
            copy_to(writer)
            writer.flush()
            os.fsync(writer.fileno())
        header = read_safetensors_header(temporary_path)
        if not has_lora_tensor_pairs(temporary_path):
            raise ValueError("checkpoint safetensors contains no matching LoRA tensor pair")
        metadata = header.get("__metadata__")
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint LoRA is missing MFLUX safetensors metadata")
        if metadata.get("mflux_version") != _EXPECTED_MFLUX_VERSION:
            raise ValueError(
                "checkpoint LoRA MFLUX version does not match the tested "
                f"{_EXPECTED_MFLUX_VERSION} runtime"
            )
        if metadata.get("model") != _EXPECTED_MODEL:
            raise ValueError(
                "checkpoint LoRA model does not identify FLUX.2 Klein base 4B"
            )
        output_bytes = temporary_path.stat().st_size
        output_sha256 = _sha256(temporary_path)
        tensor_count = len(header) - (1 if "__metadata__" in header else 0)
        try:
            os.link(temporary_path, output_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"LoRA output already exists; overwrite is disabled: {output_path}"
            ) from exc
        return {
            "schemaVersion": 1,
            "sourceType": source_type,
            "checkpoint": str(checkpoint_path),
            "checkpointManifest": _CHECKPOINT_MANIFEST,
            "loraAdapter": member,
            "output": str(output_path),
            "mfluxVersion": _EXPECTED_MFLUX_VERSION,
            "model": _EXPECTED_MODEL,
            "bytes": output_bytes,
            "sha256": output_sha256,
            "tensorCount": tensor_count,
            "loraTensorPairsValid": True,
            "overwroteExisting": False,
        }
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


extract_mflux_lora = extract_mflux_lora_adapter


__all__ = ["extract_mflux_lora", "extract_mflux_lora_adapter"]
