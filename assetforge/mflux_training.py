from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
MIN_CUDA_TRAINING_VRAM_GIB = 23.0
MIN_CUDA_COMPUTE_CAPABILITY = 7.5
MIN_CUDA13_DRIVER_MAJOR = 580
DEFAULT_TARGET_UPDATES = 1500
PORTABLE_BUNDLE_FILENAME = "assetforge-mflux-bundle.json"
PORTABLE_BUNDLE_KIND = "assetforge-mflux-training-bundle"
PORTABLE_BUNDLE_SCHEMA_VERSION = 2
PORTABLE_BUNDLE_FORMAT = "paired-edit-lora-portable-v2"

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
    candidate = root.joinpath(*relative.parts)
    current = candidate
    while current != root:
        if current.is_symlink():
            raise ValueError(f"{label} contains a symbolic link: {current}")
        current = current.parent
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the dataset root") from exc
    return candidate


def _paths_overlap(left: str | Path, right: str | Path) -> bool:
    left_path = Path(left).expanduser().resolve()
    right_path = Path(right).expanduser().resolve()
    try:
        left_path.relative_to(right_path)
        return True
    except ValueError:
        pass
    try:
        right_path.relative_to(left_path)
        return True
    except ValueError:
        return False


def _reject_symlink_components(value: str | Path, *, label: str) -> None:
    candidate = Path(os.path.abspath(os.path.expanduser(str(value))))
    while True:
        # Root-level compatibility links such as macOS /var -> /private/var
        # require administrator control. Reject symlinks below that trusted
        # system boundary, where a dataset owner can redirect paths.
        if candidate.parent != Path(candidate.anchor) and candidate.is_symlink():
            raise ValueError(f"{label} contains a symbolic link: {candidate}")
        parent = candidate.parent
        if parent == candidate:
            return
        candidate = parent


def _inspect_png(path: Path, *, label: str) -> tuple[tuple[int, int], str, str]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    try:
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.format != "PNG":
                raise ValueError(f"{label} must contain PNG data: {path}")
            size = opened.size
            opened.verify()
        with Image.open(io.BytesIO(raw)) as opened:
            rendered = opened.convert("RGB")
            rendered.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"{label} is not a readable PNG: {path}") from exc
    if size[0] < 16 or size[1] < 16:
        raise ValueError(f"{label} must be at least 16x16: {path}")
    return (
        size,
        hashlib.sha256(rendered.tobytes()).hexdigest(),
        hashlib.sha256(raw).hexdigest(),
    )


def _read_png(path: Path, *, label: str) -> tuple[tuple[int, int], str]:
    size, pixel_digest, _ = _inspect_png(path, label=label)
    return size, pixel_digest


def _inspect_prompt(path: Path, *, label: str) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    try:
        raw = path.read_bytes()
        value = raw.decode("utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{label} is not readable UTF-8: {path}") from exc
    if not value or "\0" in value:
        raise ValueError(f"{label} must contain a non-empty prompt: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _read_prompt(path: Path, *, label: str) -> str:
    value, _ = _inspect_prompt(path, label=label)
    return value


def _load_manifest(manifest: str | Path) -> tuple[Path, dict[str, Any], str]:
    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"redraw dataset manifest not found: {manifest_path}")
    try:
        raw = manifest_path.read_bytes()
        value = strict_json_loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"redraw dataset manifest is not readable JSON: {manifest_path}") from exc
    if not isinstance(value, dict):
        raise ValueError("redraw dataset manifest root must be an object")
    return manifest_path, value, hashlib.sha256(raw).hexdigest()


def _validate_split(
    root: Path,
    value: Any,
    *,
    split: str,
    prompt_count: int,
    canonical_samples: Mapping[str, Mapping[str, Any]],
    allow_managed_cache: bool = False,
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
    characters: set[str] = set()
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
        character = canonical["character"]
        characters.add(character)
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

        input_size, input_digest, input_byte_digest = _inspect_png(
            absolute["input"], label=f"{label}.input"
        )
        target_size, target_digest, target_byte_digest = _inspect_png(
            absolute["target"], label=f"{label}.target"
        )
        if input_size != target_size:
            raise ValueError(f"{label} input and target PNG sizes differ")
        if input_digest != canonical["inputPixelSha256"]:
            raise ValueError(f"{label}.input pixels differ from canonical sample {sample!r}")
        if target_digest != canonical["targetPixelSha256"]:
            raise ValueError(f"{label}.target pixels differ from canonical sample {sample!r}")
        all_sizes.add(input_size)
        prompt_text, prompt_byte_digest = _inspect_prompt(
            absolute["prompt"], label=f"{label}.prompt"
        )
        entries.append(
            {
                "index": index,
                "sample": sample,
                "character": character,
                "promptIndex": prompt_index,
                "input": str(absolute["input"]),
                "target": str(absolute["target"]),
                "prompt": str(absolute["prompt"]),
                "inputPixelSha256": input_digest,
                "targetPixelSha256": target_digest,
                "promptText": prompt_text,
                "sourceSha256": {
                    "input": input_byte_digest,
                    "target": target_byte_digest,
                    "prompt": prompt_byte_digest,
                },
            }
        )

    actual_files: set[Path] = set()
    unexpected_children = []
    for candidate in split_path.iterdir():
        if candidate.is_symlink():
            unexpected_children.append(candidate.name)
            continue
        if candidate.is_file():
            actual_files.add(candidate.resolve())
            continue
        if (
            allow_managed_cache
            and split == "train"
            and candidate.name == ".mflux_cache"
            and candidate.is_dir()
        ):
            continue
        unexpected_children.append(candidate.name)
    if actual_files != files or unexpected_children:
        raise ValueError(f"mflux.{split} contains missing or unexpected files")

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
        "characters": sorted(characters),
        "files": sorted(str(path) for path in files),
    }


def validate_redraw_training_dataset(
    manifest: str | Path,
    *,
    allow_managed_cache: bool = False,
) -> dict[str, Any]:
    """Strictly validate AssetForge's paired MFLUX train and holdout exports."""

    manifest_path, value, manifest_sha256 = _load_manifest(manifest)
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
        character = _clean_string(sample.get("character"), label=f"samples[{position}].character")
        canonical: dict[str, Any] = {"split": split, "character": character}
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

    for split_name in ("train", "validation"):
        for digest_field in ("inputPixelSha256", "targetPixelSha256"):
            seen_digests: dict[str, str] = {}
            for sample_id, sample in canonical_samples.items():
                if sample["split"] != split_name:
                    continue
                digest = sample[digest_field]
                previous = seen_digests.get(digest)
                if previous is not None:
                    raise ValueError(
                        f"{split_name} samples must have independent {digest_field} pixels: "
                        f"{previous!r} and {sample_id!r} are identical"
                    )
                seen_digests[digest] = sample_id

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

    train_image_digests: dict[str, str] = {}
    holdout_image_digests: dict[str, str] = {}
    for sample_id, sample in canonical_samples.items():
        destination = train_image_digests if sample["split"] == "train" else holdout_image_digests
        for digest_field in ("inputPixelSha256", "targetPixelSha256"):
            destination.setdefault(sample[digest_field], f"{sample_id}.{digest_field}")
    duplicate_image_digests = sorted(set(train_image_digests) & set(holdout_image_digests))
    if duplicate_image_digests:
        digest = duplicate_image_digests[0]
        raise ValueError(
            "train and holdout contain identical canonical image pixels: "
            f"{train_image_digests[digest]!r} and {holdout_image_digests[digest]!r}"
        )

    train = _validate_split(
        root,
        mflux["train"],
        split="train",
        prompt_count=prompt_count,
        canonical_samples=canonical_samples,
        allow_managed_cache=allow_managed_cache,
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
    if any(dimension % 16 for dimension in train["imageSize"]):
        raise ValueError("MFLUX training board width and height must both be divisible by 16")
    overlapping_characters = sorted(set(train["characters"]) & set(holdout["characters"]))
    if overlapping_characters:
        raise ValueError(
            "validation characters must be held out from every train sample: "
            + ", ".join(overlapping_characters)
        )

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
        "sourceType": "redraw-dataset",
        "datasetId": _clean_string(value.get("id"), label="redraw dataset id"),
        "manifest": str(manifest_path),
        "manifestSha256": manifest_sha256,
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
                if _sha256(destination) != entry["sourceSha256"][field]:
                    raise RuntimeError(
                        f"source training file changed during subset preparation: {source}"
                    )
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


def _portable_bundle_manifest_path(bundle: str | Path) -> Path:
    raw = Path(bundle).expanduser()
    _reject_symlink_components(raw, label="portable training bundle path")
    lexical = raw / PORTABLE_BUNDLE_FILENAME if raw.is_dir() else raw
    _reject_symlink_components(lexical, label="portable training bundle manifest")
    if not lexical.is_file():
        raise FileNotFoundError(f"portable training bundle manifest not found: {lexical}")
    return lexical.resolve()


def _normalize_external_model_lock(
    value: Any,
    *,
    label: str,
    included: bool | None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    expected_keys = {
        "family",
        "layout",
        "fileCount",
        "totalBytes",
        "fingerprintSha256",
        "files",
    }
    if included is not None:
        expected_keys.add("included")
    _strict_keys(value, expected_keys, label=label)
    if value["family"] != MFLUX_TRAINING_MODEL:
        raise ValueError(f"{label}.family must be {MFLUX_TRAINING_MODEL}")
    if value["layout"] not in {"mflux-component-sharded", "diffusers"}:
        raise ValueError(f"{label}.layout is unsupported")
    if included is not None and value["included"] is not included:
        raise ValueError(f"{label}.included must be {str(included).lower()}")
    file_count = _integer(value["fileCount"], label=f"{label}.fileCount", minimum=1)
    total_bytes = _integer(value["totalBytes"], label=f"{label}.totalBytes", minimum=1)
    fingerprint = _clean_string(
        value["fingerprintSha256"],
        label=f"{label}.fingerprintSha256",
    )
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError(f"{label}.fingerprintSha256 must be a lowercase SHA-256")
    raw_files = value["files"]
    if not isinstance(raw_files, list) or len(raw_files) != file_count:
        raise ValueError(f"{label}.files length must match fileCount")

    files: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    summed_bytes = 0
    for position, raw_file in enumerate(raw_files):
        file_label = f"{label}.files[{position}]"
        if not isinstance(raw_file, dict):
            raise ValueError(f"{file_label} must be an object")
        _strict_keys(raw_file, {"path", "bytes", "sha256"}, label=file_label)
        relative = _relative_path(raw_file["path"], label=f"{file_label}.path")
        logical = relative.as_posix()
        if logical in seen_paths:
            raise ValueError(f"{label}.files contains duplicate path {logical!r}")
        seen_paths.add(logical)
        size = _integer(raw_file["bytes"], label=f"{file_label}.bytes", minimum=1)
        digest = _clean_string(raw_file["sha256"], label=f"{file_label}.sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{file_label}.sha256 must be a lowercase SHA-256")
        summed_bytes += size
        files.append({"path": logical, "bytes": size, "sha256": digest})
    if files != sorted(files, key=lambda item: item["path"]):
        raise ValueError(f"{label}.files must be sorted by path")
    if summed_bytes != total_bytes:
        raise ValueError(f"{label}.totalBytes does not match files")
    calculated_fingerprint = hashlib.sha256(
        json.dumps(
            files,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if calculated_fingerprint != fingerprint:
        raise ValueError(f"{label}.fingerprintSha256 does not match files")
    normalized = {
        "family": MFLUX_TRAINING_MODEL,
        "layout": value["layout"],
        "fileCount": file_count,
        "totalBytes": total_bytes,
        "fingerprintSha256": fingerprint,
        "files": files,
    }
    if included is not None:
        normalized["included"] = included
    return normalized


def validate_portable_training_bundle(
    bundle: str | Path,
    *,
    allow_managed_cache: bool = False,
) -> dict[str, Any]:
    """Validate a train-only, byte-pinned MFLUX bundle after transfer to another host."""

    manifest_path = _portable_bundle_manifest_path(bundle)
    try:
        manifest_bytes = manifest_path.read_bytes()
        value = strict_json_loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"portable training bundle manifest is not readable JSON: {manifest_path}") from exc
    if not isinstance(value, dict):
        raise ValueError("portable training bundle manifest root must be an object")
    _strict_keys(
        value,
        {"schemaVersion", "kind", "format", "source", "holdout", "model", "data"},
        label="portable training bundle",
    )
    if value["schemaVersion"] != PORTABLE_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported portable training bundle schemaVersion; "
            f"expected {PORTABLE_BUNDLE_SCHEMA_VERSION}, rebuild it with the current AssetForge"
        )
    if value["kind"] != PORTABLE_BUNDLE_KIND:
        raise ValueError(f"portable training bundle kind must be {PORTABLE_BUNDLE_KIND}")
    if value["format"] != PORTABLE_BUNDLE_FORMAT:
        raise ValueError(f"portable training bundle format must be {PORTABLE_BUNDLE_FORMAT}")

    source = value["source"]
    if not isinstance(source, dict):
        raise ValueError("portable training bundle source must be an object")
    _strict_keys(
        source,
        {
            "datasetId",
            "manifestSha256",
            "trainSampleCount",
            "holdoutSampleCount",
            "promptCount",
            "trainCharacters",
            "holdoutCharacters",
        },
        label="portable training bundle source",
    )
    dataset_id = _clean_string(source["datasetId"], label="portable source.datasetId")
    source_manifest_sha256 = _clean_string(
        source["manifestSha256"], label="portable source.manifestSha256"
    )
    if not re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256):
        raise ValueError("portable source.manifestSha256 must be a lowercase SHA-256")
    train_count = _integer(
        source["trainSampleCount"], label="portable source.trainSampleCount", minimum=1
    )
    holdout_count = _integer(
        source["holdoutSampleCount"], label="portable source.holdoutSampleCount", minimum=1
    )
    prompt_count = _integer(source["promptCount"], label="portable source.promptCount", minimum=1)
    train_characters = source["trainCharacters"]
    holdout_characters = source["holdoutCharacters"]
    for characters, label in (
        (train_characters, "portable source.trainCharacters"),
        (holdout_characters, "portable source.holdoutCharacters"),
    ):
        if not isinstance(characters, list) or not characters:
            raise ValueError(f"{label} must be a non-empty list")
        cleaned = [_clean_string(character, label=label) for character in characters]
        if cleaned != sorted(set(cleaned)):
            raise ValueError(f"{label} must be sorted and unique")
    if set(train_characters) & set(holdout_characters):
        raise ValueError("portable train and holdout character sets must be disjoint")

    model_lock = _normalize_external_model_lock(
        value["model"],
        label="portable training bundle model",
        included=False,
    )

    holdout = value["holdout"]
    if not isinstance(holdout, dict):
        raise ValueError("portable training bundle holdout must be an object")
    _strict_keys(
        holdout,
        {"included", "sampleCount", "samples"},
        label="portable training bundle holdout",
    )
    if holdout["included"] is not False:
        raise ValueError("portable training bundle must exclude holdout files")
    if _integer(holdout["sampleCount"], label="portable holdout.sampleCount", minimum=1) != holdout_count:
        raise ValueError("portable holdout.sampleCount does not match source.holdoutSampleCount")
    holdout_samples = holdout["samples"]
    if not isinstance(holdout_samples, list) or len(holdout_samples) != holdout_count:
        raise ValueError("portable holdout.samples length must match source.holdoutSampleCount")
    cleaned_holdout_samples = [
        _clean_string(sample, label="portable holdout.samples") for sample in holdout_samples
    ]
    if cleaned_holdout_samples != sorted(set(cleaned_holdout_samples)):
        raise ValueError("portable holdout.samples must be sorted and unique")

    data = value["data"]
    if not isinstance(data, dict):
        raise ValueError("portable training bundle data must be an object")
    _strict_keys(data, {"path", "sampleCount", "imageSize", "entries"}, label="portable data")
    data_relative = _relative_path(data["path"], label="portable data.path")
    if data_relative != PurePosixPath("data"):
        raise ValueError("portable data.path must be exactly data")
    root = manifest_path.parent.resolve()
    data_path = _resolve_child(root, data_relative, label="portable data.path")
    if data_path.is_symlink() or not data_path.is_dir():
        raise FileNotFoundError(f"portable data.path is not a directory: {data_path}")
    if _integer(data["sampleCount"], label="portable data.sampleCount", minimum=1) != train_count:
        raise ValueError("portable data.sampleCount does not match source.trainSampleCount")
    image_size = data["imageSize"]
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 16 for item in image_size)
    ):
        raise ValueError("portable data.imageSize must contain two integers >= 16")
    if any(item % 16 for item in image_size):
        raise ValueError("portable data.imageSize dimensions must both be divisible by 16")
    entries_value = data["entries"]
    if not isinstance(entries_value, list) or len(entries_value) != train_count:
        raise ValueError("portable data.entries length must equal source.trainSampleCount")

    entries: list[dict[str, Any]] = []
    samples: set[str] = set()
    expected_files: set[str] = set()
    digest_pattern = re.compile(r"[0-9a-f]{64}")
    for position, raw_entry in enumerate(entries_value, start=1):
        label = f"portable data.entries[{position - 1}]"
        if not isinstance(raw_entry, dict):
            raise ValueError(f"{label} must be an object")
        _strict_keys(
            raw_entry,
            {"index", "sample", "character", "promptIndex", "files", "sha256"},
            label=label,
        )
        index = _integer(raw_entry["index"], label=f"{label}.index", minimum=1)
        if index != position:
            raise ValueError(f"{label}.index must be contiguous and equal {position}")
        sample = _clean_string(raw_entry["sample"], label=f"{label}.sample")
        if sample in samples:
            raise ValueError(f"portable data contains duplicate sample {sample!r}")
        samples.add(sample)
        character = _clean_string(raw_entry["character"], label=f"{label}.character")
        if character not in train_characters:
            raise ValueError(f"{label}.character is not declared in source.trainCharacters")
        prompt_index = _integer(raw_entry["promptIndex"], label=f"{label}.promptIndex")
        if prompt_index >= prompt_count:
            raise ValueError(f"{label}.promptIndex must be less than source.promptCount")
        files = raw_entry["files"]
        hashes = raw_entry["sha256"]
        if not isinstance(files, dict) or not isinstance(hashes, dict):
            raise ValueError(f"{label}.files and {label}.sha256 must be objects")
        _strict_keys(files, {"input", "target", "prompt"}, label=f"{label}.files")
        _strict_keys(hashes, {"input", "target", "prompt"}, label=f"{label}.sha256")
        absolute: dict[str, Path] = {}
        names: dict[str, str] = {}
        image_sizes: dict[str, tuple[int, int]] = {}
        for field in ("input", "target", "prompt"):
            relative = _relative_path(files[field], label=f"{label}.files.{field}")
            if relative.parent != PurePosixPath("."):
                raise ValueError(f"{label}.files.{field} must be a flat child of data")
            name = relative.name
            if name in expected_files:
                raise ValueError(f"portable data reuses a triplet filename: {name}")
            expected_files.add(name)
            names[field] = name
            lexical_path = data_path.joinpath(*relative.parts)
            if lexical_path.is_symlink():
                raise ValueError(f"portable training file must not be a symbolic link: {lexical_path}")
            path = _resolve_child(data_path, relative, label=f"{label}.files.{field}")
            if not path.is_file():
                raise FileNotFoundError(f"portable training file is missing: {path}")
            declared_digest = _clean_string(hashes[field], label=f"{label}.sha256.{field}")
            if not digest_pattern.fullmatch(declared_digest):
                raise ValueError(f"{label}.sha256.{field} must be a lowercase SHA-256")
            if field == "prompt":
                _, actual_digest = _inspect_prompt(path, label=f"{label}.{field}")
            else:
                size, _, actual_digest = _inspect_png(path, label=f"{label}.{field}")
                image_sizes[field] = size
            if actual_digest != declared_digest:
                raise ValueError(f"portable training file differs from its SHA-256: {path}")
            absolute[field] = path

        if not names["input"].endswith("_in.png") or not names["target"].endswith("_out.png"):
            raise ValueError(f"{label} must use *_in.png and *_out.png names")
        if not names["prompt"].endswith("_in.txt"):
            raise ValueError(f"{label}.prompt must use a *_in.txt name")
        input_base = names["input"][: -len("_in.png")]
        if (
            input_base != names["target"][: -len("_out.png")]
            or input_base != names["prompt"][: -len("_in.txt")]
            or not input_base.isdigit()
            or int(input_base) != index
        ):
            raise ValueError(f"{label} triplet basenames must encode the same index")
        input_size = image_sizes["input"]
        target_size = image_sizes["target"]
        if input_size != target_size or list(input_size) != image_size:
            raise ValueError(f"{label} PNG size does not match portable data.imageSize")
        entries.append(
            {
                "index": index,
                "sample": sample,
                "character": character,
                "promptIndex": prompt_index,
                "input": str(absolute["input"]),
                "target": str(absolute["target"]),
                "prompt": str(absolute["prompt"]),
                "sourceSha256": {
                    field: _clean_string(hashes[field], label=f"{label}.sha256.{field}")
                    for field in ("input", "target", "prompt")
                },
            }
        )

    actual_files = {candidate.name for candidate in data_path.iterdir() if candidate.is_file()}
    unexpected_children = []
    for candidate in data_path.iterdir():
        if candidate.is_file():
            continue
        if (
            allow_managed_cache
            and candidate.name == ".mflux_cache"
            and not candidate.is_symlink()
            and candidate.is_dir()
        ):
            continue
        unexpected_children.append(candidate.name)
    if actual_files != expected_files or unexpected_children:
        raise ValueError("portable data contains missing or unexpected files")
    root_children = {candidate.name for candidate in root.iterdir()}
    if root_children != {PORTABLE_BUNDLE_FILENAME, "data"}:
        raise ValueError("portable training bundle contains unexpected top-level files")

    warnings: list[str] = []
    if train_count < MIN_RECOMMENDED_SAMPLES:
        warnings.append(
            f"training has {train_count} samples; at least {MIN_RECOMMENDED_SAMPLES} are recommended for a quality run"
        )
    return {
        "ok": True,
        "sourceType": "portable-bundle",
        "datasetId": dataset_id,
        "manifest": str(manifest_path),
        "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "sourceManifestSha256": source_manifest_sha256,
        "root": str(root),
        "format": PORTABLE_BUNDLE_FORMAT,
        "promptCount": prompt_count,
        "modelLock": model_lock,
        "train": {
            "path": str(data_path),
            "sampleCount": train_count,
            "imageSize": image_size,
            "entries": entries,
            "samples": sorted(samples),
            "characters": train_characters,
            "files": sorted(str(data_path / name) for name in expected_files),
        },
        "holdout": {
            "included": False,
            "sampleCount": holdout_count,
            "samples": cleaned_holdout_samples,
            "characters": holdout_characters,
        },
        "warnings": warnings,
    }


def _load_external_model_lock_file(path: str | Path) -> dict[str, Any]:
    lexical = Path(path).expanduser()
    _reject_symlink_components(lexical, label="external model lock")
    resolved = lexical.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"external model lock not found: {resolved}")
    try:
        raw = resolved.read_bytes()
        value = strict_json_loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"external model lock is not readable JSON: {resolved}") from exc
    return _normalize_external_model_lock(
        value,
        label="external model lock",
        included=None,
    )


def create_portable_training_bundle(
    manifest: str | Path,
    output: str | Path,
    *,
    model_path: str | Path | None = None,
    model_lock_path: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically export only validated train triplets for an MFLUX host transfer."""

    if (model_path is None) == (model_lock_path is None):
        raise ValueError("provide exactly one of model_path or model_lock_path")
    report = validate_redraw_training_dataset(manifest)
    if model_path is not None:
        model_lock = fingerprint_local_mflux_model(
            model_path,
            require_unquantized=True,
        )
        model_root: Path | None = Path(model_path).expanduser().resolve()
    else:
        model_lock = _load_external_model_lock_file(model_lock_path)
        model_root = None
    output_value = Path(output).expanduser()
    if output_value.is_symlink():
        raise ValueError(f"portable training bundle output must not be a symbolic link: {output_value}")
    destination = output_value.resolve()
    source_root = Path(report["root"])
    try:
        destination.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("portable training bundle output must be outside the source dataset")
    if model_root is not None and _paths_overlap(source_root, model_root):
        raise ValueError("source dataset and local MFLUX model directory must be disjoint")
    if model_root is not None and _paths_overlap(destination, model_root):
        raise ValueError("portable training bundle output and local MFLUX model directory must be disjoint")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"portable training bundle output already exists; overwrite is disabled: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.assetforge-staging-",
            dir=destination.parent,
        )
    )
    try:
        data_path = staging / "data"
        data_path.mkdir()
        entries: list[dict[str, Any]] = []
        for entry in report["train"]["entries"]:
            files: dict[str, str] = {}
            hashes: dict[str, str] = {}
            for field in ("input", "target", "prompt"):
                source = Path(entry[field])
                copied = data_path / source.name
                _copy_exclusive(source, copied)
                if field == "prompt":
                    copied_prompt, copied_sha256 = _inspect_prompt(
                        copied, label=f"copied {field}"
                    )
                    copied_pixel_digest = None
                else:
                    _, copied_pixel_digest, copied_sha256 = _inspect_png(
                        copied, label=f"copied {field}"
                    )
                    copied_prompt = None
                if copied_sha256 != entry["sourceSha256"][field]:
                    raise RuntimeError(
                        f"source training file changed during portable bundle export: {source}"
                    )
                if field == "prompt":
                    if copied_prompt != entry["promptText"]:
                        raise RuntimeError(
                            f"source training prompt changed during portable bundle export: {source}"
                        )
                else:
                    expected_pixel_digest = entry[f"{field}PixelSha256"]
                    if copied_pixel_digest != expected_pixel_digest:
                        raise RuntimeError(
                            f"source training image changed during portable bundle export: {source}"
                        )
                files[field] = copied.name
                hashes[field] = copied_sha256
            entries.append(
                {
                    "index": entry["index"],
                    "sample": entry["sample"],
                    "promptIndex": entry["promptIndex"],
                    "character": entry["character"],
                    "files": files,
                    "sha256": hashes,
                }
            )
        bundle_manifest = staging / PORTABLE_BUNDLE_FILENAME
        bundle_manifest.write_text(
            json.dumps(
                {
                    "schemaVersion": PORTABLE_BUNDLE_SCHEMA_VERSION,
                    "kind": PORTABLE_BUNDLE_KIND,
                    "format": PORTABLE_BUNDLE_FORMAT,
                    "source": {
                        "datasetId": report["datasetId"],
                        "manifestSha256": report["manifestSha256"],
                        "trainSampleCount": report["train"]["sampleCount"],
                        "holdoutSampleCount": report["holdout"]["sampleCount"],
                        "promptCount": report["promptCount"],
                        "trainCharacters": report["train"]["characters"],
                        "holdoutCharacters": report["holdout"]["characters"],
                    },
                    "holdout": {
                        "included": False,
                        "sampleCount": report["holdout"]["sampleCount"],
                        "samples": report["holdout"]["samples"],
                    },
                    "model": {
                        **model_lock,
                        "included": False,
                    },
                    "data": {
                        "path": "data",
                        "sampleCount": report["train"]["sampleCount"],
                        "imageSize": report["train"]["imageSize"],
                        "entries": entries,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        validate_portable_training_bundle(bundle_manifest)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"portable training bundle output already exists; overwrite is disabled: {destination}"
            )
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    result = validate_portable_training_bundle(destination)
    return {
        "created": True,
        "bundlePath": str(destination),
        "bundleManifest": result["manifest"],
        "bundleManifestSha256": result["manifestSha256"],
        "sourceManifestSha256": result["sourceManifestSha256"],
        "modelFingerprintSha256": result["modelLock"]["fingerprintSha256"],
        "trainSampleCount": result["train"]["sampleCount"],
        "holdoutSampleCount": result["holdout"]["sampleCount"],
        "holdoutIncluded": False,
        "dataPath": result["train"]["path"],
        "warnings": result["warnings"],
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


def _numeric_version(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[+.-].*)?", value.strip())
    return tuple(int(part) for part in match.groups()) if match else None


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
    lexical_path = Path(raw).expanduser()
    blockers: list[str] = []
    try:
        _reject_symlink_components(lexical_path, label=f"{label} path")
    except ValueError as exc:
        blockers.append(str(exc))
    path = lexical_path.resolve()
    exists = path.exists()
    is_directory = path.is_dir()
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


def mflux_cuda_training_probe(
    executable: str | Path | None,
    *,
    version_report: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> dict[str, Any]:
    """On Linux, require a CUDA 13 MLX install and execute a tiny real GPU kernel."""

    platform_value = sys.platform if platform_name is None else platform_name
    base: dict[str, Any] = {
        "required": platform_value.startswith("linux"),
        "platform": platform_value,
        "ready": True,
        "blockers": [],
        "minimumDriverMajor": MIN_CUDA13_DRIVER_MAJOR,
        "minimumVramGiB": MIN_CUDA_TRAINING_VRAM_GIB,
        "minimumComputeCapability": MIN_CUDA_COMPUTE_CAPABILITY,
        "gpus": [],
        "mlx": None,
    }
    if not base["required"]:
        return base

    blockers: list[str] = base["blockers"]
    if (
        version_report.get("detected") != MFLUX_TRAINING_VERSION
        or version_report.get("compatible") is not True
    ):
        blockers.append(f"CUDA probe requires verified MFLUX {MFLUX_TRAINING_VERSION}")
        base["ready"] = False
        return base
    if executable is None:
        blockers.append("CUDA probe requires the verified mflux-train executable")
        base["ready"] = False
        return base
    executable_path = Path(executable).expanduser().resolve()
    python_path = executable_path.parent / "python"
    if not _ready_executable(python_path):
        blockers.append(f"CUDA probe could not find the MFLUX environment Python: {python_path}")
        base["ready"] = False
        return base

    environment = dict(os.environ if environ is None else environ)
    nvidia_smi = shutil.which("nvidia-smi", path=environment.get("PATH"))
    if nvidia_smi is None:
        blockers.append("nvidia-smi was not found on the Linux training host")
        base["ready"] = False
        return base
    try:
        query = subprocess.run(
            [
                str(Path(nvidia_smi).resolve()),
                "--query-gpu=index,uuid,name,driver_version,memory.total,memory.free,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            shell=False,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        blockers.append(f"nvidia-smi GPU query failed: {exc}")
        base["ready"] = False
        return base
    if query.returncode != 0:
        details = (query.stderr or query.stdout or "no output").strip()[-1000:]
        blockers.append(f"nvidia-smi GPU query failed: {details}")
        base["ready"] = False
        return base
    try:
        rows = list(csv.reader(query.stdout.splitlines(), skipinitialspace=True))
        if not rows:
            raise ValueError("no GPU rows")
        for row in rows:
            if len(row) != 7:
                raise ValueError(f"expected 7 columns, found {len(row)}")
            driver = row[3].strip()
            total_memory_mib = float(row[4].strip())
            free_memory_mib = float(row[5].strip())
            compute_capability = float(row[6].strip())
            driver_major = int(driver.split(".", 1)[0])
            base["gpus"].append(
                {
                    "index": int(row[0].strip()),
                    "uuid": row[1].strip(),
                    "name": row[2].strip(),
                    "driverVersion": driver,
                    "driverMajor": driver_major,
                    "memoryMiB": total_memory_mib,
                    "memoryGiB": round(total_memory_mib / 1024, 2),
                    "freeMemoryMiB": free_memory_mib,
                    "freeMemoryGiB": round(free_memory_mib / 1024, 2),
                    "computeCapability": compute_capability,
                    "eligible": (
                        driver_major >= MIN_CUDA13_DRIVER_MAJOR
                        and total_memory_mib >= MIN_CUDA_TRAINING_VRAM_GIB * 1024
                        and free_memory_mib >= MIN_CUDA_TRAINING_VRAM_GIB * 1024
                        and compute_capability >= MIN_CUDA_COMPUTE_CAPABILITY
                    ),
                }
            )
    except (TypeError, ValueError) as exc:
        blockers.append(f"nvidia-smi returned an unreadable GPU inventory: {exc}")
        base["ready"] = False
        return base
    if len(base["gpus"]) != 1:
        blockers.append(
            "CUDA probe requires exactly one GPU in the training container; "
            "isolate the intended device before running AssetForge"
        )
        base["ready"] = False
        return base
    if base["gpus"][0]["eligible"] is not True:
        blockers.append(
            "the isolated NVIDIA GPU does not satisfy driver >=580, compute capability >=7.5, "
            "and at least 23 GiB total and free VRAM"
        )
        base["ready"] = False
        return base

    probe_script = """
import importlib.metadata as metadata
import json
import mlx.core as mx

payload = {
    "mlxVersion": metadata.version("mlx"),
    "cudaPackageVersion": metadata.version("mlx-cuda-13"),
    "cudaAvailable": bool(mx.cuda.is_available()),
    "gpuAvailable": bool(mx.is_available(mx.gpu)),
}
if not payload["cudaAvailable"] or not payload["gpuAvailable"]:
    raise RuntimeError("MLX CUDA GPU is unavailable")
mx.set_default_device(mx.gpu)
left = mx.arange(16, dtype=mx.float32).reshape((4, 4))
right = mx.ones((4, 4), dtype=mx.float32)
result = left @ right
mx.eval(result)
payload["kernelSum"] = float(mx.sum(result).item())
payload["defaultDevice"] = str(mx.default_device())
print(json.dumps(payload, sort_keys=True))
""".strip()
    environment["PYTHONNOUSERSITE"] = "1"
    try:
        probe = subprocess.run(
            [str(python_path), "-I", "-c", probe_script],
            shell=False,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        blockers.append(f"MLX CUDA kernel probe failed: {exc}")
        base["ready"] = False
        return base
    if probe.returncode != 0:
        details = (probe.stderr or probe.stdout or "no output").strip()[-1000:]
        blockers.append(f"MLX CUDA kernel probe failed: {details}")
        base["ready"] = False
        return base
    try:
        lines = [line for line in probe.stdout.splitlines() if line.strip()]
        mlx_report = strict_json_loads(lines[-1])
        if not isinstance(mlx_report, dict):
            raise ValueError("probe output is not an object")
        _strict_keys(
            mlx_report,
            {
                "mlxVersion",
                "cudaPackageVersion",
                "cudaAvailable",
                "gpuAvailable",
                "kernelSum",
                "defaultDevice",
            },
            label="MLX CUDA probe",
        )
        mlx_version = _numeric_version(mlx_report["mlxVersion"])
        cuda_version = _numeric_version(mlx_report["cudaPackageVersion"])
        if mlx_version is None or not ((0, 30, 3) <= mlx_version < (0, 32, 0)):
            raise ValueError(
                f"mlx version is {mlx_report['mlxVersion']}, expected >=0.30.3 and <0.32.0"
            )
        if cuda_version is None or cuda_version != mlx_version:
            raise ValueError("mlx-cuda-13 must be installed at the exact mlx version")
        if mlx_report["cudaAvailable"] is not True:
            raise ValueError("mx.cuda.is_available() is false")
        if mlx_report["gpuAvailable"] is not True:
            raise ValueError("MLX reports no available GPU device")
        if not math.isclose(float(mlx_report["kernelSum"]), 480.0, rel_tol=0, abs_tol=1e-4):
            raise ValueError("GPU matrix kernel returned the wrong result")
        if "gpu" not in str(mlx_report["defaultDevice"]).lower():
            raise ValueError("MLX default device is not GPU")
    except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        blockers.append(f"MLX CUDA probe returned an invalid attestation: {exc}")
        base["ready"] = False
        return base
    base["mlx"] = mlx_report
    base["ready"] = True
    return base


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

    model = validate_local_mflux_model(
        model_path,
        require_unquantized=True,
    )
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
    accelerator = mflux_cuda_training_probe(
        executable_path,
        version_report=version,
        environ=environ,
    )
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
    blockers.extend(accelerator["blockers"])
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
        "accelerator": accelerator,
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


def validate_local_mflux_model(
    model_path: str | Path,
    *,
    require_unquantized: bool = False,
) -> dict[str, Any]:
    """Require a complete local MFLUX model without assuming Diffusers layout."""

    if not isinstance(require_unquantized, bool):
        raise ValueError("require_unquantized must be a boolean")
    raw = _clean_string(str(model_path), label="model_path")
    lexical_path = Path(raw).expanduser()
    _reject_symlink_components(lexical_path, label="local MFLUX model directory")
    path = lexical_path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"local MFLUX model directory not found: {path}")
    configs: list[Path] = []
    weights: list[Path] = []
    indexes: list[Path] = []
    quantization_levels: set[int] = set()
    for candidate in path.rglob("*"):
        try:
            relative = candidate.relative_to(path)
        except ValueError:
            continue
        if ".cache" in relative.parts:
            continue
        if candidate.is_symlink():
            raise ValueError(f"local MFLUX model must not contain symbolic links: {candidate}")
        if candidate.name.endswith(".incomplete"):
            raise ValueError(f"local MFLUX model contains an incomplete file: {candidate}")
        if not candidate.is_file():
            continue
        if candidate.name in {"config.json", "model_index.json"}:
            configs.append(candidate)
        if candidate.suffix.lower() in _MODEL_WEIGHT_SUFFIXES:
            if candidate.stat().st_size <= 0:
                raise ValueError(f"local MFLUX model contains an empty weight file: {candidate}")
            header = read_safetensors_header(candidate)
            metadata = header.get("__metadata__")
            if isinstance(metadata, Mapping):
                raw_level = metadata.get("quantization_level")
                if raw_level not in (None, "None", "null", ""):
                    try:
                        level = int(raw_level)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"local MFLUX model has invalid quantization metadata: {candidate}"
                        ) from exc
                    if level <= 0:
                        raise ValueError(
                            f"local MFLUX model has invalid quantization metadata: {candidate}"
                        )
                    quantization_levels.add(level)
            weights.append(candidate)
        if candidate.name.endswith(".safetensors.index.json"):
            indexes.append(candidate)
    if not weights:
        raise ValueError(f"local MFLUX model has no non-empty weight files: {path}")
    if len(quantization_levels) > 1:
        raise ValueError(
            "local MFLUX model has inconsistent quantization levels: "
            + ", ".join(str(level) for level in sorted(quantization_levels))
        )
    quantization_level = next(iter(quantization_levels), None)
    if require_unquantized and quantization_level is not None:
        raise ValueError(
            "this validation call requires unquantized base weights; "
            f"the supplied model is stored at {quantization_level}-bit"
        )

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

    tokenizer_root = path / "tokenizer"
    required_tokenizer_files = (
        tokenizer_root / "tokenizer.json",
        tokenizer_root / "tokenizer_config.json",
    )
    for tokenizer_file in required_tokenizer_files:
        if tokenizer_file.is_symlink() or not tokenizer_file.is_file():
            raise ValueError(
                f"local MFLUX model is missing a required tokenizer file: {tokenizer_file}"
            )
        try:
            tokenizer_value = strict_json_loads(tokenizer_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"local MFLUX model has an unreadable tokenizer file: {tokenizer_file}"
            ) from exc
        if not isinstance(tokenizer_value, dict):
            raise ValueError(
                f"local MFLUX model tokenizer file must contain a JSON object: {tokenizer_file}"
            )

    converted_components = ("text_encoder", "transformer", "vae")
    converted_layout = all(
        (path / component).is_dir()
        and any(weight.parent == path / component for weight in weights)
        and any(index.parent == path / component for index in indexes)
        for component in converted_components
    )
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
        "quantizationLevel": quantization_level,
    }


def fingerprint_local_mflux_model(
    model_path: str | Path,
    *,
    require_unquantized: bool = False,
) -> dict[str, Any]:
    """Hash every non-cache regular model file for relocation-safe identity locking."""

    validation = validate_local_mflux_model(
        model_path,
        require_unquantized=require_unquantized,
    )
    root = Path(validation["path"])
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root)
        if ".cache" in relative.parts:
            continue
        if candidate.is_symlink():
            raise ValueError(f"local MFLUX model must not contain symbolic links: {candidate}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(f"local MFLUX model contains a non-regular file: {candidate}")
        size = candidate.stat().st_size
        if size <= 0:
            raise ValueError(f"local MFLUX model contains an empty file: {candidate}")
        total_bytes += size
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": size,
                "sha256": _sha256(candidate),
            }
        )
    if not files:
        raise ValueError(f"local MFLUX model has no fingerprintable files: {root}")
    fingerprint = hashlib.sha256(
        json.dumps(
            files,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "family": validation["family"],
        "layout": validation["layout"],
        "fileCount": len(files),
        "totalBytes": total_bytes,
        "fingerprintSha256": fingerprint,
        "files": files,
    }


def _validate_prepared_subset(
    path: Path,
    selected: list[dict[str, Any]],
    *,
    allow_managed_cache: bool = False,
) -> None:
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
            source_hashes = entry.get("sourceSha256")
            if not isinstance(source_hashes, Mapping) or not isinstance(
                source_hashes.get(field), str
            ):
                raise ValueError("selected training entry has no approved source SHA-256")
            if _sha256(destination) != source_hashes[field]:
                raise ValueError(f"prepared training triplet differs from its train source: {destination}")
    actual_names = {candidate.name for candidate in path.iterdir() if candidate.is_file()}
    unexpected_children = []
    for candidate in path.iterdir():
        if candidate.is_file():
            continue
        if (
            allow_managed_cache
            and candidate.name == ".mflux_cache"
            and not candidate.is_symlink()
            and candidate.is_dir()
        ):
            continue
        unexpected_children.append(candidate.name)
    if actual_names != expected_names or unexpected_children:
        raise ValueError("prepared training data contains missing or unexpected files")


def _selected_files_fingerprint(entries: list[dict[str, Any]]) -> str:
    locked: list[dict[str, Any]] = []
    for entry in entries:
        hashes = entry.get("sourceSha256")
        if not isinstance(hashes, Mapping):
            raise ValueError("selected training entry has no approved source SHA-256")
        normalized: dict[str, str] = {}
        for field in ("input", "target", "prompt"):
            digest = hashes.get(field)
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("selected training entry has an invalid source SHA-256")
            normalized[field] = digest
        locked.append(
            {
                "index": entry["index"],
                "sample": entry["sample"],
                "sha256": normalized,
            }
        )
    return hashlib.sha256(
        json.dumps(locked, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _revalidate_plan_dataset(plan: Mapping[str, Any]) -> None:
    """Fail closed if auto-discovered training data changed after plan approval."""

    manifest = plan.get("manifest")
    dataset = plan.get("dataset")
    config = plan.get("config")
    if not isinstance(manifest, str) or not isinstance(dataset, Mapping) or not isinstance(config, Mapping):
        raise RuntimeError("training plan is missing its dataset integrity contract")
    source_type = dataset.get("sourceType")
    try:
        if source_type == "redraw-dataset":
            fresh = validate_redraw_training_dataset(manifest, allow_managed_cache=True)
        elif source_type == "portable-bundle":
            fresh = validate_portable_training_bundle(manifest, allow_managed_cache=True)
        else:
            raise ValueError(f"unsupported training dataset sourceType: {source_type!r}")
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(f"training dataset changed after plan approval: {exc}") from exc

    comparisons = {
        "manifestSha256": fresh["manifestSha256"],
        "sourceManifestSha256": fresh.get("sourceManifestSha256"),
        "sourceTrainCount": fresh["train"]["sampleCount"],
        "sourceHoldoutCount": fresh["holdout"]["sampleCount"],
    }
    for field, actual in comparisons.items():
        if dataset.get(field) != actual:
            raise RuntimeError(f"training dataset changed after plan approval: {field} differs")

    sample_limit = dataset.get("sampleLimit")
    selected = _selected_entries(fresh, sample_limit)
    expected_samples = [entry["sample"] for entry in selected]
    if dataset.get("samples") != expected_samples or dataset.get("selectedCount") != len(selected):
        raise RuntimeError("training dataset changed after plan approval: selected samples differ")
    if dataset.get("selectedFilesFingerprintSha256") != _selected_files_fingerprint(selected):
        raise RuntimeError("training dataset changed after plan approval: selected files differ")
    data_value = config.get("data")
    if not isinstance(data_value, str):
        raise RuntimeError("training plan has no validated data path")
    data_path = Path(data_value).expanduser().resolve()
    if sample_limit is None:
        if data_path != Path(fresh["train"]["path"]).resolve():
            raise RuntimeError("training dataset changed after plan approval: data path differs")
    else:
        try:
            _validate_prepared_subset(data_path, selected, allow_managed_cache=True)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise RuntimeError(f"training dataset changed after plan approval: {exc}") from exc


def _external_model_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "included"}


def _revalidate_plan_model(plan: Mapping[str, Any]) -> None:
    expected = plan.get("modelLock")
    config = plan.get("config")
    if expected is None:
        return
    if not isinstance(expected, Mapping) or not isinstance(config, Mapping):
        raise RuntimeError("training plan has an invalid model fingerprint contract")
    model_path = config.get("model_path")
    if not isinstance(model_path, str):
        raise RuntimeError("training plan has no model path for fingerprint verification")
    try:
        actual = fingerprint_local_mflux_model(
            model_path,
            require_unquantized=True,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(f"training model changed after plan approval: {exc}") from exc
    if actual != _external_model_lock(expected):
        raise RuntimeError("training model changed after plan approval: fingerprint differs")


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
    manifest: str | Path | None = None,
    *,
    portable_bundle: str | Path | None = None,
    expected_bundle_sha256: str | None = None,
    model_path: str | Path,
    config_output: str | Path,
    prepared_data_path: str | Path | None = None,
    sample_limit: int | None = None,
    checkpoint_output: str | Path | None = None,
    executable: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    low_ram: bool = False,
    max_resolution: int = 576,
    epochs: int | None = None,
    target_updates: int | None = None,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    checkpoint_frequency: int = 250,
    plot_frequency: int = 25,
    generate_image_frequency: int = 250,
    lora_rank: int = 16,
    seed: int = 42,
    physical_memory_bytes: int | None = None,
    minimum_free_disk_gib: float = MIN_LOCAL_TRAINING_FREE_DISK_GIB,
    allow_existing_config: bool = False,
) -> dict[str, Any]:
    """Create a deterministic MFLUX 0.18.0 FLUX.2 edit-LoRA config plan."""

    if (manifest is None) == (portable_bundle is None):
        raise ValueError("provide exactly one of manifest or portable_bundle")
    report = (
        validate_redraw_training_dataset(manifest)
        if manifest is not None
        else validate_portable_training_bundle(portable_bundle)
    )
    if portable_bundle is not None:
        expected_bundle_sha256 = _clean_string(
            expected_bundle_sha256,
            label="expected_bundle_sha256",
        )
        if not re.fullmatch(r"[0-9a-f]{64}", expected_bundle_sha256):
            raise ValueError("expected_bundle_sha256 must be a lowercase SHA-256")
        if report["manifestSha256"] != expected_bundle_sha256:
            raise ValueError("portable bundle manifest does not match expected_bundle_sha256")
    elif expected_bundle_sha256 is not None:
        raise ValueError("expected_bundle_sha256 is only accepted with portable_bundle")
    selected = _selected_entries(report, sample_limit)
    if not isinstance(low_ram, bool):
        raise ValueError("low_ram must be a boolean")
    if not isinstance(allow_existing_config, bool):
        raise ValueError("allow_existing_config must be a boolean")
    max_resolution = _integer(max_resolution, label="max_resolution", minimum=16)
    batch_size = _integer(batch_size, label="batch_size", minimum=1)
    if epochs is not None and target_updates is not None:
        raise ValueError("epochs and target_updates are mutually exclusive")
    updates_per_epoch = math.ceil(len(selected) / batch_size)
    if epochs is None:
        requested_target_updates = _integer(
            DEFAULT_TARGET_UPDATES if target_updates is None else target_updates,
            label="target_updates",
            minimum=1,
        )
        epochs = math.ceil(requested_target_updates / updates_per_epoch)
        schedule_mode = "target-updates"
    else:
        epochs = _integer(epochs, label="epochs", minimum=1)
        requested_target_updates = None
        schedule_mode = "epochs"
    checkpoint_frequency = _integer(checkpoint_frequency, label="checkpoint_frequency", minimum=1)
    plot_frequency = _integer(plot_frequency, label="plot_frequency", minimum=1)
    generate_image_frequency = _integer(
        generate_image_frequency, label="generate_image_frequency", minimum=1
    )
    lora_rank = _integer(lora_rank, label="lora_rank", minimum=1)
    seed = _integer(seed, label="seed")
    learning_rate_value = _positive_number(learning_rate, label="learning_rate")
    total_updates = epochs * updates_per_epoch
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

    config_lexical = Path(config_output).expanduser()
    _reject_symlink_components(config_lexical, label="config_output")
    config_path = config_lexical.resolve()
    if checkpoint_output is not None:
        checkpoint_lexical = Path(checkpoint_output).expanduser()
        _reject_symlink_components(checkpoint_lexical, label="checkpoint_output")
        checkpoint_path = checkpoint_lexical.resolve()
    else:
        checkpoint_path = config_path.parent / "checkpoints"
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
    model_root = Path(model["path"])
    disjoint_pairs = (
        (source_root, "source dataset", model_root, "local MFLUX model directory"),
        (data_path, "training data", model_root, "local MFLUX model directory"),
        (config_path, "config_output", model_root, "local MFLUX model directory"),
        (checkpoint_path, "checkpoint_output", model_root, "local MFLUX model directory"),
        (config_path, "config_output", data_path, "training data"),
        (checkpoint_path, "checkpoint_output", data_path, "training data"),
        (config_path, "config_output", checkpoint_path, "checkpoint_output"),
    )
    for left, left_label, right, right_label in disjoint_pairs:
        if _paths_overlap(left, right):
            raise ValueError(f"{left_label} and {right_label} must be disjoint")
    portable_model_lock = report.get("modelLock")
    actual_model_lock = fingerprint_local_mflux_model(
        model["path"],
        require_unquantized=True,
    )
    if isinstance(portable_model_lock, Mapping):
        expected_model_lock: Mapping[str, Any] = portable_model_lock
        model_lock_verified = actual_model_lock == _external_model_lock(portable_model_lock)
    else:
        expected_model_lock = actual_model_lock
        model_lock_verified = True

    width, height = report["train"]["imageSize"]
    if width % 16 or height % 16:
        raise ValueError(
            f"training board size {width}x{height} must use dimensions divisible by 16"
        )
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
        "quantize": None,
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
    if low_ram:
        blockers.append(
            "MFLUX 0.18.0 low_ram mode is disabled for managed execution because its "
            "training cache performs recursive deletion; use low_ram=false"
        )
    if model_lock_verified is False:
        blockers.append("training model fingerprint differs from the portable bundle lock")
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
            "sourceType": report["sourceType"],
            "manifestSha256": report["manifestSha256"],
            "expectedBundleSha256": expected_bundle_sha256,
            "sourceManifestSha256": report.get("sourceManifestSha256"),
            "sourceTrainCount": report["train"]["sampleCount"],
            "sourceHoldoutCount": report["holdout"]["sampleCount"],
            "sourceHoldoutCharacters": report["holdout"].get("characters"),
            "modelFingerprintSha256": (
                expected_model_lock.get("fingerprintSha256")
                if isinstance(expected_model_lock, Mapping)
                else None
            ),
            "selectedCount": len(selected),
            "sampleLimit": sample_limit,
            "dataPath": str(data_path),
            "holdoutIncluded": False,
            "samples": [entry["sample"] for entry in selected],
            "selectedFilesFingerprintSha256": _selected_files_fingerprint(selected),
        },
        "schedule": {
            "mode": schedule_mode,
            "requestedTargetUpdates": requested_target_updates,
            "updatesPerEpoch": updates_per_epoch,
            "numEpochs": epochs,
            "plannedUpdates": total_updates,
            "targetOvershoot": (
                total_updates - requested_target_updates
                if requested_target_updates is not None
                else None
            ),
        },
        "model": model,
        "modelLock": expected_model_lock,
        "modelFingerprintVerified": model_lock_verified,
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
    _reject_symlink_components(path, label="training config output")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"config output already exists; overwrite is disabled: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as writer:
        json.dump(config, writer, indent=2, ensure_ascii=False)
        writer.write("\n")
    return path.resolve()


def _revalidate_execution_paths(
    plan: Mapping[str, Any],
    config_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Path]:
    """Re-resolve every approved path and repeat overlap checks immediately before execution."""

    dataset = plan.get("dataset")
    model = plan.get("model")
    manifest = plan.get("manifest")
    checkpoint = config.get("checkpoint")
    if (
        not isinstance(dataset, Mapping)
        or not isinstance(model, Mapping)
        or not isinstance(manifest, str)
        or not isinstance(checkpoint, Mapping)
    ):
        raise RuntimeError("refusing to execute a training plan with an invalid path contract")

    expected_values = {
        "config": plan.get("configOutput"),
        "model": model.get("path"),
        "data": dataset.get("dataPath"),
        "checkpoint": checkpoint.get("output_path"),
        "source dataset": str(Path(manifest).expanduser().parent),
    }
    paths: dict[str, Path] = {}
    for label, raw in expected_values.items():
        if not isinstance(raw, str):
            raise RuntimeError(f"refusing to execute without an approved {label} path")
        lexical = Path(raw).expanduser()
        try:
            _reject_symlink_components(lexical, label=f"approved {label} path")
        except ValueError as exc:
            raise RuntimeError(f"refusing to execute with an unsafe path: {exc}") from exc
        paths[label] = lexical.resolve()

    if paths["config"] != config_path.resolve():
        raise RuntimeError("written training config path differs from the approved plan")
    model_value = config.get("model_path")
    data_value = config.get("data")
    if not isinstance(model_value, str) or Path(model_value).expanduser().resolve() != paths["model"]:
        raise RuntimeError("training model path differs from the approved plan")
    if not isinstance(data_value, str) or Path(data_value).expanduser().resolve() != paths["data"]:
        raise RuntimeError("training data path differs from the approved plan")

    disjoint_pairs = (
        ("source dataset", "model"),
        ("source dataset", "config"),
        ("source dataset", "checkpoint"),
        ("data", "model"),
        ("config", "model"),
        ("checkpoint", "model"),
        ("config", "data"),
        ("checkpoint", "data"),
        ("config", "checkpoint"),
    )
    for left, right in disjoint_pairs:
        if _paths_overlap(paths[left], paths[right]):
            raise RuntimeError(
                f"refusing to execute because approved {left} and {right} paths overlap"
            )
    return paths


def compile_mflux_train_command(
    plan: Mapping[str, Any],
    *,
    execute: bool = False,
    environ: Mapping[str, str] | None = None,
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
    try:
        _reject_symlink_components(config_path, label="training config")
    except ValueError as exc:
        raise RuntimeError(f"refusing to compile with an unsafe config path: {exc}") from exc
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
    live_paths = _revalidate_execution_paths(plan, config_path, config)
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
    checkpoint_path = live_paths["checkpoint"]
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        raise RuntimeError(
            f"refusing to compile a training command because checkpoint output already exists: {checkpoint_path}"
        )
    model_path = str(live_paths["model"])
    data_path = str(live_paths["data"])
    cache_path = live_paths["data"] / ".mflux_cache" / "training"
    minimum_free_value = doctor.get("minimumFreeDiskGiB", MIN_LOCAL_TRAINING_FREE_DISK_GIB)
    try:
        minimum_free_gib = max(float(minimum_free_value), MIN_LOCAL_TRAINING_FREE_DISK_GIB)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("doctor report has an invalid minimum free disk gate") from exc

    fresh_doctor = mflux_training_doctor(
        model_path=model_path,
        executable=executable,
        environ=environ,
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
    _revalidate_plan_model(plan)
    _revalidate_plan_dataset(plan)
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
    environment = dict(os.environ if environ is None else environ)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    # Perform the full execution preflight before invoking even MFLUX's
    # dry-run mode. A dry-run is still an external executable and may touch
    # its dataset cache, so an unverified binary or unsafe cache must never be
    # allowed to run first.
    verified_command = compile_mflux_train_command(plan, execute=True, environ=environment)
    dry_run_command = verified_command + ["--dry-run"]
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
    command = compile_mflux_train_command(plan, execute=True, environ=environment)
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
    "create_portable_training_bundle",
    "discover_mflux_train_executable",
    "fingerprint_local_mflux_model",
    "mflux_cuda_training_probe",
    "mflux_training_doctor",
    "prepare_training_data",
    "run_mflux_training_plan",
    "validate_local_mflux_model",
    "validate_portable_training_bundle",
    "validate_redraw_training_dataset",
    "write_mflux_training_config",
]
