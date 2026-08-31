from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_MAX_HEADER_BYTES = 16 * 1024 * 1024


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    """Parse and bounds-check a safetensors header without loading tensor data."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"safetensors path must not be a symbolic link: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"safetensors path must be a regular file: {resolved}")
    size = resolved.stat().st_size
    if size < 10:
        raise ValueError(f"safetensors file is too small: {resolved}")
    with resolved.open("rb") as reader:
        raw_length = reader.read(8)
        header_length = int.from_bytes(raw_length, "little", signed=False)
        if header_length < 2 or header_length > _MAX_HEADER_BYTES or 8 + header_length > size:
            raise ValueError(f"safetensors header length is invalid: {resolved}")
        raw_header = reader.read(header_length)
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"safetensors header is not valid UTF-8 JSON: {resolved}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header must be an object: {resolved}")
    data_size = size - 8 - header_length
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors:
        raise ValueError(f"safetensors file contains no tensors: {resolved}")
    for name, value in tensors.items():
        if not isinstance(name, str) or not name or not isinstance(value, dict):
            raise ValueError(f"safetensors tensor entry is invalid: {resolved}")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"safetensors tensor {name!r} has no dtype: {resolved}")
        if not isinstance(shape, list) or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
            for dimension in shape
        ):
            raise ValueError(f"safetensors tensor {name!r} has an invalid shape: {resolved}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > data_size
        ):
            raise ValueError(f"safetensors tensor {name!r} has invalid data offsets: {resolved}")
    metadata = header.get("__metadata__")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"safetensors metadata must be an object: {resolved}")
    return header


def safetensors_tensor_names(path: str | Path) -> set[str]:
    return set(read_safetensors_header(path)) - {"__metadata__"}


def has_lora_tensor_pairs(path: str | Path) -> bool:
    names = safetensors_tensor_names(path)
    pairs = (
        (".lora_A.weight", ".lora_B.weight"),
        (".lora_down.weight", ".lora_up.weight"),
        (".lora_a", ".lora_b"),
    )
    for left_suffix, right_suffix in pairs:
        left_stems = {name[: -len(left_suffix)] for name in names if name.endswith(left_suffix)}
        right_stems = {name[: -len(right_suffix)] for name in names if name.endswith(right_suffix)}
        if left_stems & right_stems:
            return True
    return False


__all__ = ["has_lora_tensor_pairs", "read_safetensors_header", "safetensors_tensor_names"]
