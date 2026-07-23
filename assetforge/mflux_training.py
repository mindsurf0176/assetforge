from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, UnidentifiedImageError

from .json_utils import strict_json_loads
from .safetensors_utils import read_safetensors_header


MFLUX_TRAINING_VERSION = "0.18.0"
MFLUX_TRAIN_EXECUTABLE = "mflux-train"
MFLUX_TRAINING_MODEL = "flux2-klein-base-4b"
MIN_RECOMMENDED_SAMPLES = 50
MIN_LOCAL_TRAINING_RAM_GIB = 24.0
MIN_LOCAL_TRAINING_FREE_DISK_GIB = 20.0

_ENTRY_KEYS = {"index", "sample", "input", "target", "prompt", "promptIndex"}
_MODEL_WEIGHT_SUFFIXES = {".safetensors"}
_LORA_MODULES = (
    ("transformer_blocks.{block}.attn.to_q", 0, 5),
    ("transformer_blocks.{block}.attn.to_k", 0, 5),
    ("transformer_blocks.{block}.attn.to_v", 0, 5),
    ("transformer_blocks.{block}.attn.to_out", 0, 5),
    ("transformer_blocks.{block}.attn.add_q_proj", 0, 5),
    ("transformer_blocks.{block}.attn.add_k_proj", 0, 5),
    ("transformer_blocks.{block}.attn.add_v_proj", 0, 5),
    ("transformer_blocks.{block}.attn.to_add_out", 0, 5),
    ("transformer_blocks.{block}.ff.linear_in", 0, 5),
    ("transformer_blocks.{block}.ff.linear_out", 0, 5),
    ("transformer_blocks.{block}.ff_context.linear_in", 0, 5),
    ("transformer_blocks.{block}.ff_context.linear_out", 0, 5),
    ("single_transformer_blocks.{block}.attn.to_qkv_mlp_proj", 0, 20),
    ("single_transformer_blocks.{block}.attn.to_out", 0, 20),
)


def _clean_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if "\0" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must not contain NUL or line breaks")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _positive_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def _strict_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise ValueError(f"{label} has invalid keys: {'; '.join(details)}")


def _relative_path(value: Any, *, label: str) -> PurePosixPath:
    raw = _clean_string(value, label=label)
    if "\\" in raw:
        raise ValueError(f"{label} must use forward slashes")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a normalized relative path")
    return path


def _resolve_child(root: Path, relative: PurePosixPath, *, label: str) -> Path:
    candidate = root.joinpath(*relative.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the dataset root") from exc
    return candidate


def _read_png(path: Path, *, label: str) -> tuple[tuple[int, int], str]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    try:
        with Image.open(path) as opened:
            if opened.format != "PNG":
                raise ValueError(f"{label} must contain PNG data: {path}")
            size = opened.size
            opened.verify()
        with Image.open(path) as opened:
            rendered = opened.convert("RGB")
            rendered.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"{label} is not a readable PNG: {path}") from exc
    if size[0] < 16 or size[1] < 16:
        raise ValueError(f"{label} must be at least 16x16: {path}")
    return size, hashlib.sha256(rendered.tobytes()).hexdigest()


def _read_prompt(path: Path, *, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{label} is not readable UTF-8: {path}") from exc
    if not value or "\0" in value:
        raise ValueError(f"{label} must contain a non-empty prompt: {path}")
    return value


def _load_manifest(manifest: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"redraw dataset manifest not found: {manifest_path}")
    try:
        value = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"redraw dataset manifest is not readable JSON: {manifest_path}") from exc
    if not isinstance(value, dict):
        raise ValueError("redraw dataset manifest root must be an object")
    return manifest_path, value


def _validate_split(
    root: Path,
    value: Any,
    *,
    split: str,
    prompt_count: int,
    canonical_samples: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"mflux.{split} must be an object")
    _strict_keys(value, {"path", "sampleCount", "entries"}, label=f"mflux.{split}")
    split_relative = _relative_path(value["path"], label=f"mflux.{split}.path")
    split_path = _resolve_child(root, split_relative, label=f"mflux.{split}.path")
    if split_path.is_symlink() or not split_path.is_dir():
        raise FileNotFoundError(f"mflux.{split}.path is not a directory: {split_path}")
    sample_count = _integer(value["sampleCount"], label=f"mflux.{split}.sampleCount", minimum=1)
    entries_value = value["entries"]
    if not isinstance(entries_value, list) or len(entries_value) != sample_count:
        raise ValueError(f"mflux.{split}.entries length must equal sampleCount")

    entries: list[dict[str, Any]] = []
    samples: set[str] = set()
    files: set[Path] = set()
    all_sizes: set[tuple[int, int]] = set()
    for position, raw_entry in enumerate(entries_value, start=1):
        label = f"mflux.{split}.entries[{position - 1}]"
        if not isinstance(raw_entry, dict):
            raise ValueError(f"{label} must be an object")
        _strict_keys(raw_entry, _ENTRY_KEYS, label=label)
        index = _integer(raw_entry["index"], label=f"{label}.index", minimum=1)
        if index != position:
            raise ValueError(f"{label}.index must be contiguous and equal {position}")
        sample = _clean_string(raw_entry["sample"], label=f"{label}.sample")
        if sample in samples:
            raise ValueError(f"mflux.{split} contains duplicate sample {sample!r}")
        samples.add(sample)
        canonical = canonical_samples.get(sample)
        expected_split = "train" if split == "train" else "validation"
        if canonical is None or canonical.get("split") != expected_split:
            raise ValueError(f"{label}.sample is not a canonical {expected_split} sample")
        prompt_index = _integer(raw_entry["promptIndex"], label=f"{label}.promptIndex")
        if prompt_index >= prompt_count:
            raise ValueError(f"{label}.promptIndex must be less than mflux.promptCount")

        logical: dict[str, PurePosixPath] = {}
        absolute: dict[str, Path] = {}
        for field in ("input", "target", "prompt"):
            relative = _relative_path(raw_entry[field], label=f"{label}.{field}")
            if relative.parent != split_relative:
                raise ValueError(f"{label}.{field} must be a flat child of mflux.{split}.path")
            path = _resolve_child(root, relative, label=f"{label}.{field}")
            if path in files:
                raise ValueError(f"mflux.{split} reuses a triplet path: {path}")
            files.add(path)
            logical[field] = relative
            absolute[field] = path

        input_name = logical["input"].name
        target_name = logical["target"].name
        prompt_name = logical["prompt"].name
        if not input_name.endswith("_in.png") or not target_name.endswith("_out.png"):
            raise ValueError(f"{label} must use *_in.png and *_out.png names")
        if not prompt_name.endswith("_in.txt"):
            raise ValueError(f"{label}.prompt must use a *_in.txt name")
        input_base = input_name[: -len("_in.png")]
        target_base = target_name[: -len("_out.png")]
        prompt_base = prompt_name[: -len("_in.txt")]
        if input_base != target_base or input_base != prompt_base:
            raise ValueError(f"{label} triplet basenames do not match")
        if not input_base.isdigit() or int(input_base) != index:
            raise ValueError(f"{label} triplet basename must encode its index")

        input_size, input_digest = _read_png(absolute["input"], label=f"{label}.input")
        target_size, target_digest = _read_png(absolute["target"], label=f"{label}.target")
        if input_size != target_size:
            raise ValueError(f"{label} input and target PNG sizes differ")
        if input_digest != canonical["inputPixelSha256"]:
            raise ValueError(f"{label}.input pixels differ from canonical sample {sample!r}")
        if target_digest != canonical["targetPixelSha256"]:
            raise ValueError(f"{label}.target pixels differ from canonical sample {sample!r}")
        all_sizes.add(input_size)
        _read_prompt(absolute["prompt"], label=f"{label}.prompt")
        entries.append(
            {
                "index": index,
                "sample": sample,
                "promptIndex": prompt_index,
                "input": str(absolute["input"]),
                "target": str(absolute["target"]),
                "prompt": str(absolute["prompt"]),
            }
        )

    if len(all_sizes) != 1:
        rendered = ", ".join(f"{width}x{height}" for width, height in sorted(all_sizes))
        raise ValueError(f"mflux.{split} triplets must all have one image size; found {rendered}")
    width, height = next(iter(all_sizes))
    return {
        "path": str(split_path),
        "sampleCount": sample_count,
        "imageSize": [width, height],
        "entries": entries,
        "samples": sorted(samples),
        "files": sorted(str(path) for path in files),
    }


def validate_redraw_training_dataset(manifest: str | Path) -> dict[str, Any]:
    """Strictly validate AssetForge's paired MFLUX train and holdout exports."""

    manifest_path, value = _load_manifest(manifest)
    mflux = value.get("mflux")
    if not isinstance(mflux, dict):
        raise ValueError("redraw dataset manifest must contain an mflux object")
    _strict_keys(mflux, {"format", "promptCount", "train", "holdout"}, label="mflux")
    if mflux["format"] != "paired-edit-lora-flat-v1":
        raise ValueError("unsupported mflux.format; expected paired-edit-lora-flat-v1")
    prompt_count = _integer(mflux["promptCount"], label="mflux.promptCount", minimum=1)
    root = manifest_path.parent.resolve()
    samples = value.get("samples")
    if not isinstance(samples, list):
        raise ValueError("redraw dataset manifest must contain samples")
    declared: dict[str, str] = {}
    canonical_samples: dict[str, dict[str, Any]] = {}
    digest_pattern = re.compile(r"^[0-9a-f]{64}$")
    for position, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"samples[{position}] must be an object")
        sample_id = _clean_string(sample.get("id"), label=f"samples[{position}].id")
        split = _clean_string(sample.get("split"), label=f"samples[{position}].split")
        if split not in {"train", "validation"}:
            raise ValueError(f"samples[{position}].split must be train or validation")
        if sample_id in declared:
            raise ValueError(f"manifest samples contains duplicate id {sample_id!r}")
        declared[sample_id] = split
        canonical: dict[str, Any] = {"split": split}
        for field, digest_field in (
            ("input", "inputPixelSha256"),
            ("target", "targetPixelSha256"),
        ):
            relative = _relative_path(sample.get(field), label=f"samples[{position}].{field}")
            path = _resolve_child(root, relative, label=f"samples[{position}].{field}")
            _, actual_digest = _read_png(path, label=f"samples[{position}].{field}")
            declared_digest = _clean_string(
                sample.get(digest_field), label=f"samples[{position}].{digest_field}"
            )
            if not digest_pattern.fullmatch(declared_digest):
                raise ValueError(f"samples[{position}].{digest_field} must be a lowercase SHA-256")
            if actual_digest != declared_digest:
                raise ValueError(f"samples[{position}].{field} pixels do not match {digest_field}")
            canonical[digest_field] = declared_digest
        canonical_samples[sample_id] = canonical

    train_target_digests = {
        sample["targetPixelSha256"]: sample_id
        for sample_id, sample in canonical_samples.items()
        if sample["split"] == "train"
    }
    holdout_target_digests = {
        sample["targetPixelSha256"]: sample_id
        for sample_id, sample in canonical_samples.items()
        if sample["split"] == "validation"
    }
    duplicate_target_digests = sorted(set(train_target_digests) & set(holdout_target_digests))
    if duplicate_target_digests:
        digest = duplicate_target_digests[0]
        raise ValueError(
            "train and holdout contain identical canonical target pixels: "
            f"{train_target_digests[digest]!r} and {holdout_target_digests[digest]!r}"
        )

    train = _validate_split(
        root,
        mflux["train"],
        split="train",
        prompt_count=prompt_count,
        canonical_samples=canonical_samples,
    )
    holdout = _validate_split(
        root,
        mflux["holdout"],
        split="holdout",
        prompt_count=prompt_count,
        canonical_samples=canonical_samples,
    )

    if Path(train["path"]).resolve() == Path(holdout["path"]).resolve():
        raise ValueError("mflux.train.path and mflux.holdout.path must be different")
    overlapping_samples = sorted(set(train["samples"]) & set(holdout["samples"]))
    if overlapping_samples:
        raise ValueError(f"holdout samples are mixed into train entries: {', '.join(overlapping_samples)}")
    overlapping_files = sorted(set(train["files"]) & set(holdout["files"]))
    if overlapping_files:
        raise ValueError("holdout files are mixed into train entries")
    if train["imageSize"] != holdout["imageSize"]:
        raise ValueError("train and holdout image sizes must match")

    splits = value.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("redraw dataset manifest must contain splits")
    if splits.get("train") != train["sampleCount"] or splits.get("validation") != holdout["sampleCount"]:
        raise ValueError("manifest splits do not match mflux train/holdout sampleCount")
    if value.get("sampleCount") != train["sampleCount"] + holdout["sampleCount"]:
        raise ValueError("manifest sampleCount does not match mflux train plus holdout")

    declared_train = {sample_id for sample_id, split in declared.items() if split == "train"}
    declared_holdout = {sample_id for sample_id, split in declared.items() if split == "validation"}
    if declared_train != set(train["samples"]):
        raise ValueError("mflux.train entries do not match manifest train samples")
    if declared_holdout != set(holdout["samples"]):
        raise ValueError("mflux.holdout entries do not match manifest validation samples")

    warnings: list[str] = []
    if train["sampleCount"] < MIN_RECOMMENDED_SAMPLES:
        warnings.append(
            f"training has {train['sampleCount']} samples; at least {MIN_RECOMMENDED_SAMPLES} are recommended for a quality run"
        )
    return {
        "ok": True,
        "datasetId": _clean_string(value.get("id"), label="redraw dataset id"),
        "manifest": str(manifest_path),
        "manifestSha256": _sha256(manifest_path),
        "root": str(root),
        "format": mflux["format"],
        "promptCount": prompt_count,
        "train": train,
        "holdout": holdout,
        "warnings": warnings,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_exclusive(source: Path, destination: Path) -> None:
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)


def _selected_entries(report: Mapping[str, Any], sample_limit: int | None) -> list[dict[str, Any]]:
    entries = list(report["train"]["entries"])
    if sample_limit is None:
        return entries
    limit = _integer(sample_limit, label="sample_limit", minimum=1)
    if limit > len(entries):
        raise ValueError(f"sample_limit cannot exceed the {len(entries)} train samples")
    return entries[:limit]


def prepare_training_data(
    manifest: str | Path,
    output: str | Path,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Prepare a deterministic train-only smoke subset without changing sources."""

    report = validate_redraw_training_dataset(manifest)
    selected = _selected_entries(report, sample_limit)
    train_count = report["train"]["sampleCount"]
    if len(selected) == train_count:
        return {
            "mode": "source",
            "prepared": True,
            "dataPath": report["train"]["path"],
            "selectedCount": train_count,
            "sampleLimit": sample_limit,
            "samples": [entry["sample"] for entry in selected],
            "artifactManifest": None,
        }

    output_value = Path(output).expanduser()
    if output_value.is_symlink():
        raise ValueError(f"training data output must not be a symbolic link: {output_value}")
    output_root = output_value.resolve()
    source_root = Path(report["root"])
    try:
        output_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("training data output must be outside the source dataset")
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"training data output is not a directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    data_path = output_root / "data"
    if data_path.exists() or data_path.is_symlink():
        raise FileExistsError(f"training data output already exists; overwrite is disabled: {data_path}")
    data_path.mkdir()
    metadata_entries: list[dict[str, Any]] = []
    try:
        for entry in selected:
            files: dict[str, str] = {}
            hashes: dict[str, str] = {}
            for field in ("input", "target", "prompt"):
                source = Path(entry[field])
                destination = data_path / source.name
                _copy_exclusive(source, destination)
                files[field] = destination.name
                hashes[field] = _sha256(destination)
            metadata_entries.append(
                {"index": entry["index"], "sample": entry["sample"], "files": files, "sha256": hashes}
            )
        artifact = data_path / "assetforge-training-subset.json"
        artifact.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "sourceDatasetId": report["datasetId"],
                    "sourceManifestSha256": report["manifestSha256"],
                    "sourceTrainCount": train_count,
                    "sampleLimit": len(selected),
                    "entries": metadata_entries,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(data_path)
        raise
    return {
        "mode": "subset",
        "prepared": True,
        "dataPath": str(data_path.resolve()),
        "selectedCount": len(selected),
        "sampleLimit": len(selected),
        "samples": [entry["sample"] for entry in selected],
        "artifactManifest": str((data_path / "assetforge-training-subset.json").resolve()),
    }


def _ready_executable(path: Path) -> bool:
    try:
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:
        return False


def discover_mflux_train_executable(
    explicit: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Find mflux-train without importing MFLUX or starting a process."""

    env = os.environ if environ is None else environ
    configured = explicit if explicit is not None else env.get("ASSETFORGE_MFLUX_TRAIN_EXECUTABLE")
    if configured is not None:
        raw = _clean_string(str(configured), label="MFLUX train executable")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute() and candidate.parent == Path("."):
            found = shutil.which(raw, path=env.get("PATH"))
            candidate = Path(found) if found else candidate
        candidate = candidate.resolve()
        return candidate if _ready_executable(candidate) else None

    found = shutil.which(MFLUX_TRAIN_EXECUTABLE, path=env.get("PATH"))
    if found and _ready_executable(Path(found)):
        return Path(found).resolve()
    candidates = (
        Path(sys.executable).resolve().parent / MFLUX_TRAIN_EXECUTABLE,
        Path.home() / ".local" / "share" / "assetforge" / "mflux-venv" / "bin" / MFLUX_TRAIN_EXECUTABLE,
        Path.home() / ".local" / "bin" / MFLUX_TRAIN_EXECUTABLE,
    )
    for candidate in candidates:
        if _ready_executable(candidate):
            return candidate.resolve()
    return None


def _physical_memory_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    total = page_size * page_count
    return total if total > 0 else None


def _detect_adjacent_mflux_version(executable: Path | None) -> dict[str, Any]:
    """Read MFLUX package metadata adjacent to an executable without running it."""

    report: dict[str, Any] = {
        "required": MFLUX_TRAINING_VERSION,
        "detected": None,
        "compatible": None,
        "metadataPaths": [],
        "detectedVersions": [],
    }
    if executable is None:
        return report

    environment_root = executable.parent.parent
    site_packages: list[Path] = []
    lib_root = environment_root / "lib"
    if lib_root.is_dir():
        site_packages.extend(sorted(lib_root.glob("python*/site-packages")))
    windows_site_packages = environment_root / "Lib" / "site-packages"
    if windows_site_packages.is_dir():
        site_packages.append(windows_site_packages)

    metadata_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for site_packages_path in site_packages:
        for candidate in sorted(site_packages_path.glob("mflux-*.dist-info/METADATA")):
            if candidate.is_symlink() or candidate.parent.is_symlink() or not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved not in seen_paths:
                metadata_paths.append(resolved)
                seen_paths.add(resolved)

    versions: list[str] = []
    accepted_metadata: list[str] = []
    for metadata_path in metadata_paths:
        if not metadata_path.is_file():
            continue
        try:
            metadata = metadata_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        name_match = re.search(r"^Name:\s*([^\r\n]+)$", metadata, flags=re.MULTILINE | re.IGNORECASE)
        version_match = re.search(r"^Version:\s*([^\r\n]+)$", metadata, flags=re.MULTILINE | re.IGNORECASE)
        if name_match is None or version_match is None:
            continue
        normalized_name = re.sub(r"[-_.]+", "-", name_match.group(1).strip().lower())
        version = version_match.group(1).strip()
        if normalized_name != "mflux" or not version or len(version) > 128:
            continue
        versions.append(version)
        accepted_metadata.append(str(metadata_path))

    unique_versions = sorted(set(versions))
    report["metadataPaths"] = accepted_metadata
    report["detectedVersions"] = unique_versions
    if len(unique_versions) == 1:
        report["detected"] = unique_versions[0]
        report["compatible"] = unique_versions[0] == MFLUX_TRAINING_VERSION
    elif len(unique_versions) > 1:
        report["compatible"] = False
    return report


def _nearest_existing_directory(path: Path) -> Path | None:
    candidate = path if path.exists() else path.parent
    while True:
        if candidate.is_dir():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            return None
        candidate = parent


def _absolute_without_resolving(value: str | Path, *, label: str) -> Path:
    """Return an absolute lexical path while preserving symlink components."""

    raw = _clean_string(str(value), label=f"{label}_path")
    return Path(os.path.abspath(os.path.expanduser(raw)))


def _managed_cache_path_blockers(
    data_path: str | Path,
    cache_path: str | Path,
) -> list[str]:
    """Reject cache layouts that could make MFLUX delete outside the dataset."""

    data = _absolute_without_resolving(data_path, label="data")
    cache = _absolute_without_resolving(cache_path, label="cache")
    expected = data / ".mflux_cache" / "training"
    blockers: list[str] = []
    if cache != expected:
        blockers.append(
            f"managed cache path must be exactly inside the data path: expected {expected}, got {cache}"
        )
        return blockers

    for candidate in (data / ".mflux_cache", expected):
        if candidate.is_symlink():
            blockers.append(f"managed cache path contains a symbolic link: {candidate}")

    try:
        resolved_data = data.resolve()
        resolved_cache = cache.resolve()
        resolved_cache.relative_to(resolved_data)
    except (OSError, RuntimeError, ValueError):
        blockers.append(f"managed cache path escapes the resolved data path: {cache}")
    return blockers


def _inspect_training_path(
    value: str | Path | None,
    *,
    label: str,
    must_exist: bool,
    minimum_free_bytes: int,
) -> dict[str, Any]:
    if value is None:
        return {
            "checked": False,
            "path": None,
            "mustExist": must_exist,
            "exists": None,
            "writable": None,
            "freeBytes": None,
            "freeGiB": None,
            "enoughFreeDisk": None,
            "ready": None,
            "blockers": [],
        }

    raw = _clean_string(str(value), label=f"{label}_path")
    path = Path(raw).expanduser().resolve()
    exists = path.exists()
    is_directory = path.is_dir()
    blockers: list[str] = []
    if must_exist and not exists:
        blockers.append(f"{label} path does not exist: {path}")
    if exists and not is_directory:
        blockers.append(f"{label} path must be a directory: {path}")

    anchor = _nearest_existing_directory(path)
    writable = False
    free_bytes: int | None = None
    device: int | None = None
    if anchor is None:
        blockers.append(f"{label} path has no existing directory ancestor: {path}")
    else:
        writable = os.access(anchor, os.W_OK | os.X_OK)
        if not writable:
            blockers.append(f"{label} path is not writable through existing directory: {anchor}")
        try:
            free_bytes = int(shutil.disk_usage(anchor).free)
            device = int(anchor.stat().st_dev)
        except (OSError, TypeError, ValueError):
            blockers.append(f"free disk could not be determined for {label} path: {anchor}")

    enough_free = None if free_bytes is None else free_bytes >= minimum_free_bytes
    return {
        "checked": True,
        "path": str(path),
        "mustExist": must_exist,
        "exists": exists,
        "directory": is_directory if exists else None,
        "existingAnchor": str(anchor) if anchor else None,
        "writable": writable,
        "device": device,
        "freeBytes": free_bytes,
        "freeGiB": None if free_bytes is None else round(free_bytes / (1024**3), 2),
        "enoughFreeDisk": enough_free,
        "ready": not blockers and enough_free is not False,
        "blockers": blockers,
    }


def mflux_training_doctor(
    *,
    model_path: str | Path,
    executable: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    physical_memory_bytes: int | None = None,
    data_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    minimum_free_disk_gib: float = MIN_LOCAL_TRAINING_FREE_DISK_GIB,
) -> dict[str, Any]:
    """Inspect local training prerequisites without importing or executing MFLUX."""

    model = validate_local_mflux_model(model_path)
    executable_path = discover_mflux_train_executable(executable, environ=environ)
    if physical_memory_bytes is None:
        memory_bytes = _physical_memory_bytes()
    else:
        memory_bytes = _integer(
            physical_memory_bytes,
            label="physical_memory_bytes",
            minimum=1,
        )
    memory_gib = None if memory_bytes is None else memory_bytes / (1024**3)
    minimum_free_gib = _positive_number(
        minimum_free_disk_gib,
        label="minimum_free_disk_gib",
    )
    minimum_free_bytes = int(minimum_free_gib * 1024**3)
    if cache_path is None and data_path is not None:
        cache_path = Path(data_path).expanduser() / ".mflux_cache" / "training"
    paths = {
        "data": _inspect_training_path(
            data_path,
            label="data",
            must_exist=True,
            minimum_free_bytes=minimum_free_bytes,
        ),
        "checkpoint": _inspect_training_path(
            checkpoint_path,
            label="checkpoint",
            must_exist=False,
            minimum_free_bytes=minimum_free_bytes,
        ),
        "cache": _inspect_training_path(
            cache_path,
            label="cache",
            must_exist=False,
            minimum_free_bytes=minimum_free_bytes,
        ),
    }
    if data_path is not None and cache_path is not None:
        cache_blockers = _managed_cache_path_blockers(data_path, cache_path)
        paths["cache"]["blockers"].extend(cache_blockers)
        if cache_blockers:
            paths["cache"]["ready"] = False
    version = _detect_adjacent_mflux_version(executable_path)
    blockers: list[str] = []
    warnings: list[str] = []
    if executable_path is None:
        blockers.append("mflux-train executable was not found")
    elif len(version["detectedVersions"]) > 1:
        blockers.append(
            "multiple adjacent MFLUX versions were detected for mflux-train: "
            + ", ".join(version["detectedVersions"])
        )
    elif version["detected"] is None:
        warnings.append(
            "MFLUX version could not be detected from dist-info adjacent to mflux-train; "
            f"required version is {MFLUX_TRAINING_VERSION}"
        )
    elif not version["compatible"]:
        blockers.append(
            f"mflux-train version mismatch: detected {version['detected']}, "
            f"required {MFLUX_TRAINING_VERSION}"
        )
    if model.get("family") != MFLUX_TRAINING_MODEL:
        blockers.append("training model metadata does not identify FLUX.2 Klein base 4B")
    if memory_gib is None:
        blockers.append("physical memory could not be determined; local training is disabled")
    elif memory_gib < MIN_LOCAL_TRAINING_RAM_GIB:
        blockers.append(
            f"local FLUX.2 edit-LoRA training requires at least {MIN_LOCAL_TRAINING_RAM_GIB:g} GiB physical memory; detected {memory_gib:.1f} GiB"
        )

    for path_report in paths.values():
        blockers.extend(path_report["blockers"])
    low_disk_by_device: dict[int | str, list[tuple[str, dict[str, Any]]]] = {}
    for label, path_report in paths.items():
        if path_report["checked"] and path_report["enoughFreeDisk"] is False:
            key = path_report["device"]
            if key is None:
                key = path_report["existingAnchor"] or label
            low_disk_by_device.setdefault(key, []).append((label, path_report))
    for entries in low_disk_by_device.values():
        labels = ", ".join(label for label, _ in entries)
        free_gib = min(float(report["freeGiB"]) for _, report in entries)
        blockers.append(
            f"insufficient free disk for training paths ({labels}): "
            f"{free_gib:.2f} GiB available, {minimum_free_gib:.2f} GiB required"
        )
    return {
        "localTrainingReady": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "model": model,
        "executable": str(executable_path) if executable_path else None,
        "mfluxVersion": version,
        "physicalMemoryBytes": memory_bytes,
        "physicalMemoryGiB": None if memory_gib is None else round(memory_gib, 2),
        "minimumPhysicalMemoryGiB": MIN_LOCAL_TRAINING_RAM_GIB,
        "minimumFreeDiskBytes": minimum_free_bytes,
        "minimumFreeDiskGiB": minimum_free_gib,
        "paths": paths,
        "shell": False,
        "trainingExecutionProvided": True,
        "trainingExecutionRequiresExplicitExecute": True,
    }


def validate_local_mflux_model(model_path: str | Path) -> dict[str, Any]:
    """Require a complete local MFLUX model without assuming Diffusers layout."""

    raw = _clean_string(str(model_path), label="model_path")
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"local MFLUX model directory not found: {path}")
    configs: list[Path] = []
    weights: list[Path] = []
    indexes: list[Path] = []
    for candidate in path.rglob("*"):
        try:
            relative = candidate.relative_to(path)
        except ValueError:
            continue
        if ".cache" in relative.parts:
            continue
        if candidate.name.endswith(".incomplete"):
            raise ValueError(f"local MFLUX model contains an incomplete file: {candidate}")
        if not candidate.is_file():
            continue
        if candidate.name in {"config.json", "model_index.json"}:
            configs.append(candidate)
        if candidate.suffix.lower() in _MODEL_WEIGHT_SUFFIXES:
            if candidate.stat().st_size <= 0:
                raise ValueError(f"local MFLUX model contains an empty weight file: {candidate}")
            read_safetensors_header(candidate)
            weights.append(candidate)
        if candidate.name.endswith(".safetensors.index.json"):
            indexes.append(candidate)
    if not weights:
        raise ValueError(f"local MFLUX model has no non-empty weight files: {path}")

    indexed_shards: set[Path] = set()
    for index in indexes:
        try:
            value = strict_json_loads(index.read_text(encoding="utf-8"))
            weight_map = value["weight_map"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid MFLUX weight index: {index}") from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"MFLUX weight index has an empty weight_map: {index}")
        for shard_value in weight_map.values():
            shard_relative = _relative_path(shard_value, label=f"weight shard in {index.name}")
            shard = (index.parent / Path(*shard_relative.parts)).resolve()
            try:
                shard.relative_to(path)
            except ValueError as exc:
                raise ValueError(f"MFLUX weight index shard escapes model_path: {shard_relative}") from exc
            if not shard.is_file() or shard.stat().st_size <= 0:
                raise ValueError(f"MFLUX weight index references a missing shard: {shard}")
            indexed_shards.add(shard)

    converted_components = ("text_encoder", "transformer", "vae")
    converted_layout = all(
        (path / component).is_dir()
        and any(weight.parent == path / component for weight in weights)
        and any(index.parent == path / component for index in indexes)
        for component in converted_components
    ) and (path / "tokenizer" / "tokenizer.json").is_file()
    diffusers_layout = (path / "model_index.json").is_file() and all(
        (path / component / "config.json").is_file()
        and any(weight.parent == path / component for weight in weights)
        for component in converted_components
    )
    if not converted_layout and not diffusers_layout:
        raise ValueError(
            "local MFLUX model is neither a component-sharded conversion nor a configured Diffusers layout: "
            f"{path}"
        )
    identity_text = path.name
    for metadata_path in (path / "config.json", path / "model_index.json", path / "README.md"):
        if metadata_path.is_file() and not metadata_path.is_symlink():
            try:
                identity_text += "\n" + metadata_path.read_text(encoding="utf-8")[:65536]
            except (OSError, UnicodeError):
                pass
    normalized_identity = re.sub(r"[^a-z0-9]", "", identity_text.lower())
    family = (
        MFLUX_TRAINING_MODEL
        if "flux2kleinbase4b" in normalized_identity
        else None
    )
    return {
        "ready": True,
        "path": str(path),
        "layout": "mflux-component-sharded" if converted_layout else "diffusers",
        "family": family,
        "configCount": len(configs),
        "weightFileCount": len(weights),
        "indexCount": len(indexes),
        "indexedShardCount": len(indexed_shards),
    }


def _validate_prepared_subset(path: Path, selected: list[dict[str, Any]]) -> None:
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(f"prepared training data not found: {path}")
    expected_names = {"assetforge-training-subset.json"}
    for entry in selected:
        for field in ("input", "target", "prompt"):
            source = Path(entry[field])
            expected_names.add(source.name)
            destination = path / source.name
            if destination.is_symlink() or not destination.is_file():
                raise FileNotFoundError(f"prepared training triplet is missing: {destination}")
            if _sha256(destination) != _sha256(source):
                raise ValueError(f"prepared training triplet differs from its train source: {destination}")
    actual_names = {candidate.name for candidate in path.iterdir() if candidate.is_file()}
    if actual_names != expected_names:
        raise ValueError("prepared training data contains missing or unexpected files")


def _lora_targets(rank: int) -> list[dict[str, Any]]:
    return [
        {
            "module_path": module,
            "blocks": {"start": start, "end": end},
            "rank": rank,
        }
        for module, start, end in _LORA_MODULES
    ]


def build_mflux_training_plan(
    manifest: str | Path,
    *,
    model_path: str | Path,
    config_output: str | Path,
    prepared_data_path: str | Path | None = None,
    sample_limit: int | None = None,
    checkpoint_output: str | Path | None = None,
    executable: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    low_ram: bool = True,
    max_resolution: int = 576,
    epochs: int = 100,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    checkpoint_frequency: int = 25,
    plot_frequency: int = 1,
    generate_image_frequency: int = 20,
    lora_rank: int = 16,
    seed: int = 42,
    physical_memory_bytes: int | None = None,
    minimum_free_disk_gib: float = MIN_LOCAL_TRAINING_FREE_DISK_GIB,
    allow_existing_config: bool = False,
) -> dict[str, Any]:
    """Create a deterministic MFLUX 0.18.0 FLUX.2 edit-LoRA config plan."""

    report = validate_redraw_training_dataset(manifest)
    selected = _selected_entries(report, sample_limit)
    if not isinstance(low_ram, bool):
        raise ValueError("low_ram must be a boolean")
    if not isinstance(allow_existing_config, bool):
        raise ValueError("allow_existing_config must be a boolean")
    max_resolution = _integer(max_resolution, label="max_resolution", minimum=16)
    epochs = _integer(epochs, label="epochs", minimum=1)
    batch_size = _integer(batch_size, label="batch_size", minimum=1)
    checkpoint_frequency = _integer(checkpoint_frequency, label="checkpoint_frequency", minimum=1)
    plot_frequency = _integer(plot_frequency, label="plot_frequency", minimum=1)
    generate_image_frequency = _integer(
        generate_image_frequency, label="generate_image_frequency", minimum=1
    )
    lora_rank = _integer(lora_rank, label="lora_rank", minimum=1)
    seed = _integer(seed, label="seed")
    learning_rate_value = _positive_number(learning_rate, label="learning_rate")
    total_updates = epochs * math.ceil(len(selected) / batch_size)
    if checkpoint_frequency > total_updates:
        raise ValueError(
            f"checkpoint_frequency={checkpoint_frequency} exceeds the planned "
            f"{total_updates} optimizer updates; training would produce no checkpoint"
        )

    if len(selected) < report["train"]["sampleCount"]:
        if prepared_data_path is None:
            raise ValueError(
                "prepared_data_path is required for a smoke subset; call prepare_training_data first"
            )
        data_path = Path(prepared_data_path).expanduser().resolve()
        _validate_prepared_subset(data_path, selected)
    else:
        if prepared_data_path is not None:
            raise ValueError("prepared_data_path is only accepted when sample_limit selects a subset")
        data_path = Path(report["train"]["path"])

    config_path = Path(config_output).expanduser().resolve()
    checkpoint_path = (
        Path(checkpoint_output).expanduser().resolve()
        if checkpoint_output is not None
        else config_path.parent / "checkpoints"
    )
    source_root = Path(report["root"])
    for candidate, label in ((config_path, "config_output"), (checkpoint_path, "checkpoint_output")):
        try:
            candidate.relative_to(source_root)
        except ValueError:
            pass
        else:
            raise ValueError(f"{label} must be outside the source dataset")

    cache_path = data_path / ".mflux_cache" / "training"
    doctor = mflux_training_doctor(
        model_path=model_path,
        executable=executable,
        environ=environ,
        physical_memory_bytes=physical_memory_bytes,
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        cache_path=cache_path,
        minimum_free_disk_gib=minimum_free_disk_gib,
    )
    model = doctor["model"]

    width, height = report["train"]["imageSize"]
    area = width * height
    max_area = max_resolution * max_resolution
    if area > max_area:
        raise ValueError(
            f"max_resolution={max_resolution} would resize {width}x{height} training boards; "
            "pixel-art edit training requires native board pixels"
        )
    preview_width = 16 * (width // 16)
    preview_height = 16 * (height // 16)
    config = {
        "model": MFLUX_TRAINING_MODEL,
        "model_path": model["path"],
        "data": str(data_path),
        "seed": seed,
        "steps": 40,
        "guidance": 1.0,
        "quantize": 4,
        "max_resolution": max_resolution,
        "low_ram": low_ram,
        "training_loop": {
            "num_epochs": epochs,
            "batch_size": batch_size,
            "timestep_low": 25,
            "timestep_high": 40,
        },
        "optimizer": {"name": "AdamW", "learning_rate": learning_rate_value},
        "checkpoint": {
            "output_path": str(checkpoint_path),
            "save_frequency": checkpoint_frequency,
        },
        "monitoring": {
            "preview_width": preview_width,
            "preview_height": preview_height,
            "plot_frequency": plot_frequency,
            "generate_image_frequency": generate_image_frequency,
        },
        "lora_layers": {"targets": _lora_targets(lora_rank)},
    }
    executable_path = doctor["executable"]
    blockers: list[str] = list(doctor["blockers"])
    config_reused = False
    if config_path.exists() or config_path.is_symlink():
        if allow_existing_config and config_path.is_file() and not config_path.is_symlink():
            try:
                existing_config = strict_json_loads(config_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"existing training config is unreadable: {config_path}"
                ) from exc
            if existing_config != config:
                blockers.append("existing config differs from the newly validated plan")
            else:
                config_reused = True
        else:
            blockers.append("config output already exists; overwrite is disabled")
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        blockers.append("checkpoint output already exists; choose an empty path")
    warnings = list(report["warnings"]) + list(doctor["warnings"])
    if len(selected) < report["train"]["sampleCount"]:
        warnings.append(
            f"smoke subset selects {len(selected)} of {report['train']['sampleCount']} train samples; do not use it as a quality run"
        )
    return {
        "schemaVersion": 1,
        "provider": "mflux",
        "mfluxVersion": MFLUX_TRAINING_VERSION,
        "ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "manifest": report["manifest"],
        "dataset": {
            "sourceTrainCount": report["train"]["sampleCount"],
            "selectedCount": len(selected),
            "sampleLimit": sample_limit,
            "dataPath": str(data_path),
            "holdoutIncluded": False,
            "samples": [entry["sample"] for entry in selected],
        },
        "model": model,
        "doctor": doctor,
        "config": config,
        "configOutput": str(config_path),
        "configReused": config_reused,
        "executable": executable_path,
        "execution": {
            "shell": False,
            "supportedMode": "dry-run-or-explicit-gated-training",
            "actualTrainingRequiresCompatible24GiBHost": True,
            "actualTrainingCommandCompilationProvided": True,
            "actualTrainingExecutionProvided": True,
            "actualTrainingSubprocessExecutionProvided": True,
            "configContainsHostSpecificAbsolutePaths": True,
        },
    }


def write_mflux_training_config(plan: Mapping[str, Any]) -> Path:
    """Write one planned config with exclusive creation; existing files are never replaced."""

    config = plan.get("config")
    output = plan.get("configOutput")
    if not isinstance(config, dict) or not isinstance(output, str):
        raise ValueError("invalid MFLUX training plan")
    path = Path(output)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"config output already exists; overwrite is disabled: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as writer:
        json.dump(config, writer, indent=2, ensure_ascii=False)
        writer.write("\n")
    return path.resolve()


def compile_mflux_train_command(
    plan: Mapping[str, Any],
    *,
    execute: bool = False,
) -> list[str]:
    """Compile dry-run argv, or training argv after a fresh host-safety audit."""

    if not isinstance(execute, bool):
        raise ValueError("execute must be a boolean")
    executable = plan.get("executable")
    config_output = plan.get("configOutput")
    config = plan.get("config")
    if not isinstance(executable, str) or not _ready_executable(Path(executable)):
        raise RuntimeError("refusing to compile MFLUX training command without an executable")
    if not isinstance(config_output, str) or not isinstance(config, dict):
        raise ValueError("invalid MFLUX training plan")
    config_path = Path(config_output)
    if not config_path.is_file():
        raise RuntimeError("write and verify the MFLUX training config before command compilation")
    try:
        written = strict_json_loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("written MFLUX training config is unreadable") from exc
    if written != config:
        raise RuntimeError("written MFLUX training config differs from the approved plan")
    command = [
        str(Path(executable).resolve()),
        "--config",
        str(config_path.resolve()),
    ]
    if not execute:
        return command + ["--dry-run"]

    if plan.get("ready") is not True:
        raise RuntimeError("refusing to compile a training command from a plan that is not ready")
    doctor = plan.get("doctor")
    if not isinstance(doctor, Mapping):
        raise RuntimeError("refusing to compile a training command without a doctor report")
    checkpoint_conf = config.get("checkpoint")
    if not isinstance(checkpoint_conf, dict) or not isinstance(checkpoint_conf.get("output_path"), str):
        raise ValueError("invalid MFLUX checkpoint config")
    if config.get("model") != MFLUX_TRAINING_MODEL:
        raise RuntimeError(f"refusing to train a model other than {MFLUX_TRAINING_MODEL}")
    model_path = config.get("model_path")
    data_path = config.get("data")
    if not isinstance(model_path, str) or not isinstance(data_path, str):
        raise ValueError("invalid MFLUX model or data path in config")
    checkpoint_path = Path(checkpoint_conf["output_path"]).expanduser()
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        raise RuntimeError(
            f"refusing to compile a training command because checkpoint output already exists: {checkpoint_path}"
        )
    cache_path = Path(data_path).expanduser() / ".mflux_cache" / "training"
    minimum_free_value = doctor.get("minimumFreeDiskGiB", MIN_LOCAL_TRAINING_FREE_DISK_GIB)
    try:
        minimum_free_gib = max(float(minimum_free_value), MIN_LOCAL_TRAINING_FREE_DISK_GIB)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("doctor report has an invalid minimum free disk gate") from exc

    fresh_doctor = mflux_training_doctor(
        model_path=model_path,
        executable=executable,
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        cache_path=cache_path,
        minimum_free_disk_gib=minimum_free_gib,
    )
    version = fresh_doctor["mfluxVersion"]
    if version["detected"] != MFLUX_TRAINING_VERSION or version["compatible"] is not True:
        raise RuntimeError(
            f"refusing to compile a training command without verified MFLUX {MFLUX_TRAINING_VERSION}"
        )
    if not fresh_doctor["localTrainingReady"]:
        details = "; ".join(fresh_doctor["blockers"])
        raise RuntimeError(f"refusing to compile a training command on an unsafe host: {details}")
    return command


def run_mflux_training_plan(
    plan: Mapping[str, Any],
    *,
    execute: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run a verified dry-run followed by training only after explicit consent."""

    if execute is not True:
        raise RuntimeError("actual MFLUX training requires execute=True")
    # Perform the full execution preflight before invoking even MFLUX's
    # dry-run mode. A dry-run is still an external executable and may touch
    # its dataset cache, so an unverified binary or unsafe cache must never be
    # allowed to run first.
    verified_command = compile_mflux_train_command(plan, execute=True)
    dry_run_command = verified_command + ["--dry-run"]
    environment = dict(os.environ if environ is None else environ)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    dry_run = subprocess.run(
        dry_run_command,
        shell=False,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if dry_run.returncode != 0:
        details = (dry_run.stderr or dry_run.stdout or "no output").strip()[-2000:]
        raise RuntimeError(
            f"MFLUX training dry-run failed with exit code {dry_run.returncode}: {details}"
        )

    # Re-audit the live host after the parser check, immediately before the
    # expensive process starts. This also rejects a newly-created checkpoint
    # destination instead of resuming or overwriting it implicitly.
    command = compile_mflux_train_command(plan, execute=True)
    result = subprocess.run(
        command,
        shell=False,
        env=environment,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"MFLUX training failed with exit code {result.returncode}")

    config = plan.get("config")
    checkpoint_config = config.get("checkpoint") if isinstance(config, Mapping) else None
    checkpoint_output = (
        checkpoint_config.get("output_path")
        if isinstance(checkpoint_config, Mapping)
        else None
    )
    if not isinstance(checkpoint_output, str):
        raise RuntimeError("completed MFLUX training plan has no checkpoint output")
    checkpoint_directory = Path(checkpoint_output).expanduser().resolve() / "checkpoints"
    if checkpoint_directory.is_symlink() or not checkpoint_directory.is_dir():
        raise RuntimeError(
            f"MFLUX training completed without a checkpoint directory: {checkpoint_directory}"
        )
    checkpoints = sorted(
        path.resolve()
        for path in checkpoint_directory.glob("*_checkpoint.zip")
        if path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    )
    if not checkpoints:
        raise RuntimeError("MFLUX training completed without a non-empty checkpoint ZIP")
    return {
        "ok": True,
        "provider": "mflux",
        "mfluxVersion": MFLUX_TRAINING_VERSION,
        "argv": command,
        "shell": False,
        "offlineModelMode": True,
        "checkpointDirectory": str(checkpoint_directory),
        "checkpointCount": len(checkpoints),
        "latestCheckpoint": str(checkpoints[-1]),
        "checkpoints": [str(path) for path in checkpoints],
    }


__all__ = [
    "MFLUX_TRAINING_MODEL",
    "MFLUX_TRAINING_VERSION",
    "MIN_LOCAL_TRAINING_FREE_DISK_GIB",
    "MIN_LOCAL_TRAINING_RAM_GIB",
    "build_mflux_training_plan",
    "compile_mflux_train_command",
    "discover_mflux_train_executable",
    "mflux_training_doctor",
    "prepare_training_data",
    "run_mflux_training_plan",
    "validate_local_mflux_model",
    "validate_redraw_training_dataset",
    "write_mflux_training_config",
]
