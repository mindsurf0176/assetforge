from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .safetensors_utils import has_lora_tensor_pairs, read_safetensors_header


DEFAULT_EXECUTABLE = "mflux-generate-flux2-edit"
DEFAULT_MODEL = "Runpod/FLUX.2-klein-4B-mflux-4bit"
DEFAULT_BASE_MODEL = "flux2-klein-4b"
DEFAULT_PROMPT = (
    "Use cell 0 as the exact character identity and pixel-art style reference. "
    "Replace every pose-guide cell with a finished frame of that same character. "
    "Preserve the board layout, canvas size, transparent sprite background, palette, proportions, "
    "outline weight, and cell alignment. Do not add labels, crop, resize, merge, "
    "remove, or reorder cells. Do not paint a white, gray, black, checkerboard, or gradient "
    "backdrop. Output genuinely transparent RGBA pixels around every sprite and avoid "
    "semi-transparent white matte or light halo pixels on the silhouette edge."
)

_MODEL_REPOSITORIES = {
    "flux2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
    "flux2-klein-9b": "black-forest-labs/FLUX.2-klein-9B",
    "flux2-klein-9b-kv": "black-forest-labs/FLUX.2-klein-9b-kv",
    "flux2-klein-base-4b": "black-forest-labs/FLUX.2-klein-base-4B",
    "flux2-klein-base-9b": "black-forest-labs/FLUX.2-klein-base-9B",
}
_MFLUX_MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "flux2-klein-4b": ("flux2-klein-4b", "flux2-klein", "klein-4b"),
    "flux2-klein-9b": ("flux2-klein-9b", "klein-9b"),
    "flux2-klein-9b-kv": ("flux2-klein-9b-kv", "klein-9b-kv"),
    "flux2-klein-base-4b": ("flux2-klein-base-4b", "flux2-base-4b", "klein-base-4b"),
    "flux2-klein-base-9b": ("flux2-klein-base-9b", "flux2-base-9b", "klein-base-9b"),
}
_QUANTIZE_CHOICES = {3, 4, 5, 6, 8}
_IMAGE_SUFFIXES = {".png"}
_SAFE_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$", re.IGNORECASE)
_WEIGHT_SUFFIXES = {".safetensors"}


def _clean_text(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{label} must not be empty")
    if "\x00" in cleaned or "\r" in cleaned or "\n" in cleaned:
        raise ValueError(f"{label} must not contain NUL or line breaks")
    return cleaned


def _resolve_path(value: str | Path, *, label: str) -> Path:
    raw = str(value)
    if "\x00" in raw or "\r" in raw or "\n" in raw:
        raise ValueError(f"{label} must not contain NUL or line breaks")
    return Path(value).expanduser().resolve()


def _ready_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _find_executable(value: str | Path, *, search_path: str | None = None) -> Path | None:
    raw = str(value)
    if "\x00" in raw or "\r" in raw or "\n" in raw:
        raise ValueError("MFLUX executable must not contain NUL or line breaks")
    expanded = Path(value).expanduser()
    has_path_component = expanded.is_absolute() or expanded.parent != Path(".")
    if has_path_component:
        candidate = expanded.resolve()
    else:
        discovered = shutil.which(raw, path=search_path)
        if not discovered:
            return None
        candidate = Path(discovered).resolve()
    return candidate if _ready_executable(candidate) else None


def discover_mflux_executable(
    explicit: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Find the FLUX.2 edit CLI without importing or starting MFLUX.

    An explicit or environment-configured executable is authoritative: if it is
    invalid, discovery fails instead of silently selecting a different binary.
    """

    env = os.environ if environ is None else environ
    configured = explicit if explicit is not None else env.get("ASSETFORGE_MFLUX_EXECUTABLE")
    if configured is not None:
        return _find_executable(configured, search_path=env.get("PATH"))

    discovered = _find_executable(DEFAULT_EXECUTABLE, search_path=env.get("PATH"))
    if discovered:
        return discovered

    home = Path.home()
    candidates = [
        home / ".local" / "share" / "assetforge" / "mflux-venv" / "bin" / DEFAULT_EXECUTABLE,
        home / ".local" / "bin" / DEFAULT_EXECUTABLE,
        home / ".local" / "share" / "uv" / "tools" / "mflux" / "bin" / DEFAULT_EXECUTABLE,
        home / "Library" / "Application Support" / "uv" / "tools" / "mflux" / "bin" / DEFAULT_EXECUTABLE,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if _ready_executable(resolved):
            return resolved
    return None


def _default_mflux_cache() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "mflux"
    return Path.home() / ".cache" / "mflux"


def _cache_roots(
    cache_dir: str | Path | None,
    env: Mapping[str, str],
) -> tuple[Path, list[Path]]:
    configured = cache_dir or env.get("MFLUX_CACHE_DIR")
    mflux_cache = _resolve_path(configured, label="MFLUX cache") if configured else _default_mflux_cache().resolve()

    if env.get("HF_HUB_CACHE"):
        huggingface_cache = _resolve_path(env["HF_HUB_CACHE"], label="Hugging Face cache")
    elif env.get("HF_HOME"):
        huggingface_cache = (_resolve_path(env["HF_HOME"], label="Hugging Face home") / "hub").resolve()
    else:
        huggingface_cache = (Path.home() / ".cache" / "huggingface" / "hub").resolve()

    roots: list[Path] = []
    for root in (mflux_cache, huggingface_cache):
        if root not in roots:
            roots.append(root)
    return mflux_cache, roots


def _directory_has_complete_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_model_index = False
    weight_files: list[Path] = []
    weight_indexes: list[Path] = []
    try:
        for candidate in path.rglob("*"):
            if ".cache" in candidate.relative_to(path).parts:
                continue
            if candidate.name.endswith(".incomplete"):
                return False
            if not candidate.is_file():
                continue
            if candidate.name in {"config.json", "model_index.json"}:
                has_model_index |= candidate.name == "model_index.json" and candidate.parent == path
            if candidate.name.endswith(".safetensors.index.json"):
                weight_indexes.append(candidate)
            if candidate.suffix.lower() in _WEIGHT_SUFFIXES and candidate.stat().st_size > 0:
                weight_files.append(candidate)
                if candidate.suffix.lower() == ".safetensors":
                    read_safetensors_header(candidate)
    except (OSError, ValueError):
        return False
    for index_path in weight_indexes:
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            required = set(data["weight_map"].values())
        except (AttributeError, OSError, KeyError, TypeError, json.JSONDecodeError):
            return False
        if not required:
            return False
        for name in required:
            if not isinstance(name, str):
                return False
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                return False
            shard = index_path.parent / name
            try:
                shard_ready = shard.is_file() and shard.stat().st_size > 0
            except OSError:
                return False
            if not shard_ready:
                return False
    required_components = ("text_encoder", "transformer", "vae")
    component_layout = all(
        (path / component).is_dir()
        and any(weight.parent == path / component for weight in weight_files)
        for component in required_components
    ) and (path / "tokenizer" / "tokenizer.json").is_file()
    diffusers_layout = has_model_index and all(
        (path / component / "config.json").is_file()
        and any(weight.parent == path / component for weight in weight_files)
        for component in required_components
    )
    return bool(weight_files) and (component_layout or diffusers_layout)


def _cached_model(model: str, repository: str, roots: list[Path]) -> Path | None:
    organization, name = repository.split("/", 1)
    hub_slug = f"models--{organization}--{name}"
    direct_names = (model, name, hub_slug)
    for root in roots:
        for direct_name in direct_names:
            candidate = root / direct_name
            snapshots = candidate / "snapshots"
            if snapshots.is_dir():
                try:
                    snapshot_entries = sorted(snapshots.iterdir(), key=lambda entry: entry.name)
                except OSError:
                    snapshot_entries = []
                for snapshot in snapshot_entries:
                    if _directory_has_complete_weights(snapshot):
                        return snapshot.resolve()
            if _directory_has_complete_weights(candidate):
                return candidate.resolve()
    return None


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path if path.is_dir() else path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _model_state(
    model: str,
    base_model: str | None,
    model_path: str | Path | None,
    cache_roots: list[Path],
) -> dict[str, Any]:
    if base_model is None and model == DEFAULT_MODEL:
        base_model = DEFAULT_BASE_MODEL
    if base_model is not None and base_model not in _MODEL_REPOSITORIES:
        raise ValueError(f"unsupported MFLUX base model: {base_model!r}")
    configured_path = model_path
    if configured_path is None:
        possible_path = Path(model).expanduser()
        explicit_relative_path = model.startswith(("./", "../", "~/"))
        if possible_path.is_absolute() or explicit_relative_path or possible_path.exists():
            configured_path = possible_path

    if configured_path is not None:
        effective_base = base_model or (model if model in _MODEL_REPOSITORIES else DEFAULT_BASE_MODEL)
        local_path = _resolve_path(configured_path, label="MFLUX model path")
        ready = _directory_has_complete_weights(local_path)
        return {
            "id": str(local_path),
            "baseModel": effective_base,
            "commandModel": str(local_path),
            "includeBaseModel": True,
            "repository": None,
            "source": "local",
            "path": str(local_path),
            "ready": ready,
            "implicitDownloadAllowed": False,
        }

    if model in _MODEL_REPOSITORIES:
        repository = _MODEL_REPOSITORIES[model]
        effective_base = base_model or model
        include_base = base_model is not None and base_model != model
    else:
        if not _SAFE_MODEL_ID.fullmatch(model) or model.count("/") != 1:
            raise ValueError(f"unsupported MFLUX model: {model!r}")
        repository = model
        effective_base = base_model
        if effective_base is None:
            raise ValueError("base_model is required for a third-party MFLUX model")
        include_base = True
    cached = _cached_model(model, repository, cache_roots)
    return {
        "id": str(cached) if cached else model,
        "baseModel": effective_base,
        "commandModel": str(cached) if cached else model,
        "includeBaseModel": True if cached else include_base,
        "repository": repository,
        "source": "cache",
        "path": str(cached) if cached else None,
        "ready": cached is not None,
        "implicitDownloadAllowed": False,
    }


def _lora_values(lora: str | Path | Sequence[str | Path] | None) -> list[str | Path]:
    if lora is None:
        return []
    if isinstance(lora, (str, Path)):
        return [lora]
    return list(lora)


def _lora_state(lora: str | Path | Sequence[str | Path] | None) -> dict[str, Any]:
    items = []
    for value in _lora_values(lora):
        path = _resolve_path(value, label="LoRA path")
        error = None
        ready = path.is_file() and path.suffix.lower() == ".safetensors"
        if ready:
            try:
                ready = has_lora_tensor_pairs(path)
                if not ready:
                    error = "LoRA safetensors contains no compatible A/B tensor pair"
            except ValueError as exc:
                ready = False
                error = str(exc)
        elif path.suffix.lower() != ".safetensors":
            error = "LoRA must use a .safetensors filename"
        else:
            error = "LoRA file does not exist"
        items.append({"path": str(path), "ready": ready, "error": error})
    return {
        "configured": bool(items),
        "path": items[0]["path"] if len(items) == 1 else None,
        "paths": [item["path"] for item in items],
        "items": items,
        "ready": all(item["ready"] for item in items),
    }


def mflux_doctor(
    *,
    executable: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    base_model: str | None = None,
    model_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    lora: str | Path | Sequence[str | Path] | None = None,
    disk_path: str | Path | None = None,
    minimum_free_gib: float = 6.0,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect local edit-inference prerequisites without loading model code."""

    model = _clean_text(model, label="model")
    if not math.isfinite(minimum_free_gib) or minimum_free_gib < 0:
        raise ValueError("minimum_free_gib must be a finite non-negative number")
    env = os.environ if environ is None else environ
    executable_path = discover_mflux_executable(executable, environ=env)
    mflux_cache, cache_roots = _cache_roots(cache_dir, env)
    if model_path is None and env.get("ASSETFORGE_MFLUX_MODEL"):
        model_path = env["ASSETFORGE_MFLUX_MODEL"]
    if model_path is None and cache_dir is None and model == DEFAULT_MODEL:
        conventional_model = (
            Path.home() / ".local" / "share" / "assetforge" / "models" / "FLUX.2-klein-4B-mflux-4bit"
        )
        if conventional_model.exists():
            model_path = conventional_model
    model_state = _model_state(model, base_model, model_path, cache_roots)
    lora_state = _lora_state(lora)

    usage_target = _resolve_path(disk_path or Path.cwd(), label="disk path")
    usage_root = _nearest_existing_directory(usage_target)
    usage = shutil.disk_usage(usage_root)
    free_gib = usage.free / (1024**3)
    disk_ready = free_gib >= minimum_free_gib
    cache_ready = mflux_cache.is_dir() and os.access(mflux_cache, os.R_OK | os.W_OK)

    blockers: list[str] = []
    warnings: list[str] = []
    if executable_path is None:
        blockers.append(f"{DEFAULT_EXECUTABLE} executable is not installed or executable")
    if not model_state["ready"]:
        blockers.append("model weights are not complete in a local path or cache; implicit download is disabled")
    if model_state["source"] == "cache" and not cache_ready:
        blockers.append(f"MFLUX cache is not an existing readable and writable directory: {mflux_cache}")
    if not lora_state["ready"]:
        blockers.append("LoRA must be an existing local .safetensors file")
    if not disk_ready:
        blockers.append(
            f"insufficient free disk: {free_gib:.2f} GiB available, {minimum_free_gib:.2f} GiB required"
        )
    if model_state["includeBaseModel"]:
        command_model = str(model_state["commandModel"]).lower()
        matches = [
            (base, alias)
            for base, aliases in _MFLUX_MODEL_ALIASES.items()
            for alias in aliases
            if alias in command_model
        ]
        inferred_base = (
            sorted(matches, key=lambda match: (-len(match[1]), list(_MFLUX_MODEL_ALIASES).index(match[0])))[0][0]
            if matches
            else None
        )
        if inferred_base != model_state["baseModel"]:
            blockers.append(
                "MFLUX 0.18.0 FLUX.2 edit may ignore --base-model; rename or select a model path/repository "
                "containing its recognizable base-model alias"
            )
        else:
            warnings.append(
                "MFLUX 0.18.0 FLUX.2 edit may ignore --base-model; this model name contains the "
                "recognizable base-model alias used for fallback inference"
            )

    ready = not blockers
    return {
        "ok": ready,
        "generationReady": ready,
        "backend": "mflux-flux2-edit",
        "executable": {
            "name": DEFAULT_EXECUTABLE,
            "path": str(executable_path) if executable_path else None,
            "ready": executable_path is not None,
        },
        "model": model_state,
        "cache": {
            "path": str(mflux_cache),
            "rootsChecked": [str(root) for root in cache_roots],
            "exists": mflux_cache.is_dir(),
            "readWrite": cache_ready,
        },
        "lora": lora_state,
        "disk": {
            "path": str(usage_root),
            "freeBytes": usage.free,
            "freeGiB": round(free_gib, 2),
            "minimumFreeGiB": minimum_free_gib,
            "ready": disk_ready,
        },
        "blockers": blockers,
        "warnings": warnings,
    }


def _input_state(path: Path) -> dict[str, Any]:
    state: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "format": None,
        "size": None,
        "ready": False,
        "error": None,
    }
    if not path.is_file():
        state["error"] = "input board is not an existing regular file"
        return state
    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        state["error"] = "paired-board input must be PNG"
        return state
    try:
        with Image.open(path) as opened:
            state["format"] = opened.format
            state["size"] = [opened.width, opened.height]
            opened.verify()
    except (OSError, ValueError) as exc:
        state["error"] = f"input board is not a readable image: {exc}"
        return state
    width, height = state["size"]
    if state["format"] != "PNG":
        state["error"] = "paired-board input must contain PNG image data"
    elif width < 64 or height < 64 or width > 4096 or height > 4096:
        state["error"] = "paired-board dimensions must be between 64 and 4096 pixels"
    elif width % 16 or height % 16:
        state["error"] = "paired-board dimensions must be divisible by 16"
    else:
        state["ready"] = True
    return state


def _output_state(path: Path, *, overwrite: bool) -> dict[str, Any]:
    parent = path.parent
    error = None
    raw_exists = path.exists() or path.is_symlink()
    if path.suffix.lower() != ".png":
        error = "paired-board output must use a .png filename"
    elif path.is_symlink():
        error = "paired-board output must not be a symbolic link"
    elif raw_exists and path.is_dir():
        error = "paired-board output points to a directory"
    elif raw_exists and not overwrite:
        error = "paired-board output already exists and overwrite is disabled"
    elif not parent.is_dir():
        error = "paired-board output parent directory does not exist"
    elif not os.access(parent, os.W_OK):
        error = "paired-board output parent directory is not writable"
    return {
        "path": str(path),
        "parent": str(parent),
        "exists": raw_exists,
        "overwrite": overwrite,
        "ready": error is None,
        "error": error,
    }


def _format_float(value: float) -> str:
    return format(value, ".12g")


def _argv_from_plan(plan: Mapping[str, Any]) -> list[str]:
    executable = plan["doctor"]["executable"]["path"]
    model_state = plan["doctor"]["model"]
    parameters = plan["parameters"]
    argv = [str(executable), "--model", str(model_state["commandModel"])]
    if model_state["includeBaseModel"]:
        argv.extend(["--base-model", str(model_state["baseModel"])])
    argv.extend(
        [
            "--image-paths",
            plan["input"]["path"],
            "--prompt",
            parameters["prompt"],
            "--steps",
            str(parameters["steps"]),
            "--guidance",
            _format_float(parameters["guidance"]),
            "--seed",
            str(parameters["seed"]),
            "--width",
            str(parameters["width"]),
            "--height",
            str(parameters["height"]),
        ]
    )
    if parameters["quantize"] is not None:
        argv.extend(["--quantize", str(parameters["quantize"])])
    if parameters["lowRam"]:
        argv.append("--low-ram")
    if parameters["mlxCacheLimitGiB"] is not None:
        argv.extend(["--mlx-cache-limit-gb", _format_float(parameters["mlxCacheLimitGiB"])])
    lora = plan["doctor"]["lora"]
    if lora["configured"]:
        # PyPI mflux 0.18.0 predates the main-branch atomic --lora flag.
        argv.append("--lora-paths")
        argv.extend(lora["paths"])
        argv.append("--lora-scales")
        argv.extend(_format_float(scale) for scale in parameters["loraScales"])
    if parameters["metadata"]:
        argv.append("--metadata")
    argv.extend(["--output", plan["output"]["path"]])
    return argv


def paired_board_edit_plan(
    input_path: str | Path,
    output_path: str | Path,
    *,
    prompt: str = DEFAULT_PROMPT,
    executable: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    base_model: str | None = None,
    model_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    lora: str | Path | Sequence[str | Path] | None = None,
    lora_scale: float | Sequence[float] = 1.0,
    seed: int = 42,
    steps: int | None = None,
    guidance: float = 1.0,
    quantize: int | None = None,
    low_ram: bool = True,
    mlx_cache_limit_gib: float | None = None,
    metadata: bool = True,
    overwrite: bool = False,
    minimum_free_gib: float = 6.0,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Plan one deterministic, offline paired-board edit inference run."""

    prompt = _clean_text(prompt, label="prompt")
    model = _clean_text(model, label="model")
    if model_path is None and model not in _MODEL_REPOSITORIES and not _SAFE_MODEL_ID.fullmatch(model):
        raise ValueError(f"unsafe model id: {model!r}")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2_147_483_647:
        raise ValueError("seed must be an integer between 0 and 2147483647")
    if steps is not None and (
        not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 100
    ):
        raise ValueError("steps must be an integer between 1 and 100")
    if not math.isfinite(guidance) or guidance <= 0 or guidance > 20:
        raise ValueError("guidance must be finite and greater than 0 and at most 20")
    if quantize is not None and quantize not in _QUANTIZE_CHOICES:
        raise ValueError(f"quantize must be one of {sorted(_QUANTIZE_CHOICES)} or None")
    lora_values = _lora_values(lora)
    if isinstance(lora_scale, Sequence) and not isinstance(lora_scale, (str, bytes)):
        lora_scales = list(lora_scale)
    else:
        lora_scales = [lora_scale] * len(lora_values)
    if len(lora_scales) != len(lora_values):
        raise ValueError("lora_scale sequence must have exactly one scale per LoRA")
    for scale in lora_scales:
        if not isinstance(scale, (int, float)) or isinstance(scale, bool):
            raise TypeError("each LoRA scale must be a number")
        if not math.isfinite(scale) or scale <= 0 or scale > 2:
            raise ValueError("each LoRA scale must be finite and greater than 0 and at most 2")
    if mlx_cache_limit_gib is not None and (
        not math.isfinite(mlx_cache_limit_gib) or mlx_cache_limit_gib <= 0
    ):
        raise ValueError("mlx_cache_limit_gib must be a finite positive number or None")
    for label, value in (("low_ram", low_ram), ("metadata", metadata), ("overwrite", overwrite)):
        if not isinstance(value, bool):
            raise TypeError(f"{label} must be a boolean")

    input_resolved = _resolve_path(input_path, label="input path")
    output_raw = Path(output_path).expanduser()
    if output_raw.is_symlink():
        output_resolved = output_raw.absolute()
    else:
        output_resolved = output_raw.resolve()
    if input_resolved == output_resolved:
        raise ValueError("input and output paths must be different")
    input_state = _input_state(input_resolved)
    output_state = _output_state(output_resolved, overwrite=overwrite)
    doctor = mflux_doctor(
        executable=executable,
        model=model,
        base_model=base_model,
        model_path=model_path,
        cache_dir=cache_dir,
        lora=lora,
        disk_path=output_resolved.parent,
        minimum_free_gib=minimum_free_gib,
        environ=environ,
    )
    if steps is None:
        steps = 50 if "base" in str(doctor["model"]["baseModel"]).lower() else 4

    blockers = list(doctor["blockers"])
    if not input_state["ready"]:
        blockers.append(input_state["error"])
    if not output_state["ready"]:
        blockers.append(output_state["error"])
    ready = not blockers
    size = input_state["size"] or [None, None]
    plan: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "mflux-paired-board-edit",
        "backend": "mflux-flux2-edit",
        "ready": ready,
        "executable": ready,
        "input": input_state,
        "output": output_state,
        "parameters": {
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
            "quantize": quantize,
            "width": size[0],
            "height": size[1],
            "lowRam": bool(low_ram),
            "mlxCacheLimitGiB": mlx_cache_limit_gib,
            "metadata": bool(metadata),
            "loraScale": lora_scales[0] if len(lora_scales) == 1 else None,
            "loraScales": lora_scales,
        },
        "doctor": doctor,
        "execution": {
            "shell": False,
            "cwd": str(output_resolved.parent),
            "environment": {
                "MFLUX_CACHE_DIR": doctor["cache"]["path"],
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        },
        "blockers": blockers,
        "argv": None,
    }
    if ready:
        plan["argv"] = _argv_from_plan(plan)
    return plan


def compile_mflux_command(plan: Mapping[str, Any]) -> list[str]:
    """Compile a checked plan to argv suitable for subprocess.run(shell=False)."""

    if plan.get("schemaVersion") != 1 or plan.get("kind") != "mflux-paired-board-edit":
        raise ValueError("not an AssetForge MFLUX paired-board plan")
    if not plan.get("ready") or plan.get("blockers"):
        blockers = "; ".join(str(item) for item in plan.get("blockers", [])) or "plan is not ready"
        raise RuntimeError(f"refusing to compile blocked MFLUX plan: {blockers}")
    if plan.get("execution", {}).get("shell") is not False:
        raise ValueError("MFLUX plans must execute with shell=False")
    argv = _argv_from_plan(plan)
    recorded = plan.get("argv")
    if recorded is not None and list(recorded) != argv:
        raise ValueError("MFLUX plan argv does not match its checked parameters")
    return argv


def _publish_staged_output_pair(staged_output: Path, output_path: Path) -> None:
    """Publish a PNG and optional metadata sidecar with rollback on failure."""

    staged_metadata = staged_output.with_suffix(".metadata.json")
    destination_metadata = output_path.with_suffix(".metadata.json")
    destination_exists = output_path.exists() or output_path.is_symlink()
    metadata_exists = destination_metadata.exists() or destination_metadata.is_symlink()
    if output_path.is_symlink() or (destination_exists and not output_path.is_file()):
        raise RuntimeError("MFLUX output destination changed to an unsafe path during generation")
    if destination_metadata.is_symlink() or (
        metadata_exists and not destination_metadata.is_file()
    ):
        raise RuntimeError("MFLUX metadata destination is unsafe")
    if staged_metadata.exists() and (
        not staged_metadata.is_file() or staged_metadata.is_symlink()
    ):
        raise RuntimeError("MFLUX produced an unsafe metadata sidecar")

    backup_output = staged_output.parent / ".assetforge-previous-output.png"
    backup_metadata = staged_output.parent / ".assetforge-previous-metadata.json"
    if backup_output.exists() or backup_output.is_symlink():
        raise RuntimeError("MFLUX staging output backup path is not empty")
    if backup_metadata.exists() or backup_metadata.is_symlink():
        raise RuntimeError("MFLUX staging metadata backup path is not empty")

    moved_output = False
    moved_metadata = False
    published_output = False
    published_metadata = False
    try:
        if destination_exists:
            os.replace(output_path, backup_output)
            moved_output = True
        if metadata_exists:
            os.replace(destination_metadata, backup_metadata)
            moved_metadata = True
        os.replace(staged_output, output_path)
        published_output = True
        if staged_metadata.is_file():
            os.replace(staged_metadata, destination_metadata)
            published_metadata = True
    except OSError as exc:
        rollback_errors: list[str] = []

        def rollback_replace(source: Path, destination: Path, *, label: str) -> None:
            try:
                os.replace(source, destination)
            except OSError as rollback_exc:
                rollback_errors.append(f"{label}: {rollback_exc}")

        if published_metadata:
            rollback_replace(
                destination_metadata,
                staged_metadata,
                label="remove newly published metadata",
            )
        if published_output:
            rollback_replace(
                output_path,
                staged_output,
                label="remove newly published output",
            )
        if moved_metadata:
            rollback_replace(
                backup_metadata,
                destination_metadata,
                label="restore previous metadata",
            )
        if moved_output:
            rollback_replace(
                backup_output,
                output_path,
                label="restore previous output",
            )
        if rollback_errors:
            raise RuntimeError(
                "MFLUX output publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        raise RuntimeError("MFLUX output publication failed; previous output restored") from exc


def run_mflux_plan(
    plan: Mapping[str, Any],
    *,
    execute: bool = False,
) -> dict[str, Any]:
    """Execute one checked plan without a shell and verify the produced PNG.

    The explicit execute gate keeps plans inspectable by default. MFLUX output is
    streamed to the current terminal so long model downloads and inference do
    not look stalled.
    """

    if not execute:
        raise RuntimeError("MFLUX execution requires execute=True")
    argv = compile_mflux_command(plan)
    output_path = _resolve_path(plan["output"]["path"], label="output path")
    environment = os.environ.copy()
    environment.update(
        {
            str(key): str(value)
            for key, value in plan.get("execution", {}).get("environment", {}).items()
        }
    )
    # MFLUX auto-suffixes an existing output instead of overwriting it. Always
    # render into an exclusive staging directory, verify that exact new file,
    # then atomically publish it. A failed run therefore preserves an approved
    # previous output even when overwrite was explicitly enabled.
    with tempfile.TemporaryDirectory(prefix=".assetforge-mflux-", dir=output_path.parent) as staging:
        staged_output = Path(staging) / output_path.name
        execution_argv = list(argv)
        if len(execution_argv) < 2 or execution_argv[-2] != "--output":
            raise RuntimeError("checked MFLUX argv does not end in an output path")
        execution_argv[-1] = str(staged_output)
        completed = subprocess.run(
            execution_argv,
            cwd=plan["execution"]["cwd"],
            env=environment,
            shell=False,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"MFLUX exited with status {completed.returncode}")
        output_state = _input_state(staged_output)
        if not output_state["ready"]:
            raise RuntimeError(f"MFLUX did not produce a valid PNG: {output_state['error']}")
        expected_size = plan["input"]["size"]
        if output_state["size"] != expected_size:
            raise RuntimeError(
                "MFLUX output dimensions changed: "
                f"expected {expected_size[0]}x{expected_size[1]}, "
                f"got {output_state['size'][0]}x{output_state['size'][1]}"
            )

        destination_exists = output_path.exists() or output_path.is_symlink()
        if output_path.is_symlink() or (destination_exists and output_path.is_dir()):
            raise RuntimeError("MFLUX output destination changed to an unsafe path during generation")
        if destination_exists and not plan["output"].get("overwrite"):
            raise RuntimeError("MFLUX output destination appeared while overwrite is disabled")
        _publish_staged_output_pair(staged_output, output_path)
    return {
        "ok": True,
        "backend": plan["backend"],
        "input": plan["input"]["path"],
        "output": str(output_path),
        "size": output_state["size"],
        "seed": plan["parameters"]["seed"],
        "steps": plan["parameters"]["steps"],
        "model": plan["doctor"]["model"]["id"],
        "lora": plan["doctor"]["lora"]["paths"],
    }


# Short aliases keep the isolated backend convenient without coupling it to cli.py.
doctor = mflux_doctor
build_plan = paired_board_edit_plan
compile_command = compile_mflux_command
run_plan = run_mflux_plan


__all__ = [
    "DEFAULT_EXECUTABLE",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_MODEL",
    "DEFAULT_PROMPT",
    "build_plan",
    "compile_command",
    "compile_mflux_command",
    "discover_mflux_executable",
    "doctor",
    "mflux_doctor",
    "paired_board_edit_plan",
    "run_mflux_plan",
    "run_plan",
]
