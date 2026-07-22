from __future__ import annotations

import json
import os
import shutil
import uuid
import hashlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from . import __version__
from .json_utils import strict_json_loads
from .profile import Profile


def _comfy_endpoint(profile: Profile) -> str:
    provider = profile.data.get("providers", {}).get("comfyLocal", {})
    return os.environ.get("ASSETFORGE_COMFY_URL", provider.get("endpoint", "http://127.0.0.1:8188"))


def _resolve_workflow_path(profile: Profile, value: str | Path, *, explicit: bool = False) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = Path.cwd() if explicit else profile.path.parent
    return (base / path).resolve()


def comfy_workflow_path(profile: Profile, value: str | Path | None = None) -> Path:
    provider = profile.data.get("providers", {}).get("comfyLocal", {})
    configured = value or provider.get("workflowTemplate")
    if not configured:
        raise ValueError(f"profile {profile.id!r} has no ComfyUI workflowTemplate")
    path = _resolve_workflow_path(profile, configured, explicit=value is not None)
    if not path.is_file():
        raise FileNotFoundError(f"ComfyUI API workflow not found: {path}")
    return path


def load_comfy_workflow(profile: Profile, value: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    path = comfy_workflow_path(profile, value)
    try:
        workflow = strict_json_loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid ComfyUI workflow JSON {path}: {exc}") from exc
    validate_comfy_workflow(workflow)
    return path, workflow


def _ping(url: str) -> tuple[bool, str | None]:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/system_stats", timeout=1.5) as response:
            return response.status == 200, None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)


def doctor(profile: Profile) -> dict[str, Any]:
    endpoint = _comfy_endpoint(profile)
    reachable, error = _ping(endpoint)
    provider = profile.data.get("providers", {}).get("comfyLocal", {})
    workflow = provider.get("workflowTemplate")
    workflow_path = _resolve_workflow_path(profile, workflow) if workflow else None
    installation_path = Path(provider.get("installationPath", "~/ComfyUI-Local")).expanduser().resolve()
    manual_installed = (installation_path / "main.py").is_file()
    checkpoint = provider.get("checkpoint")
    checkpoint_path = installation_path / "models" / "checkpoints" / checkpoint if checkpoint else None
    checkpoint_ready = bool(not checkpoint or (checkpoint_path and checkpoint_path.is_file()))
    lora = provider.get("lora")
    lora_path = installation_path / "models" / "loras" / lora if lora else None
    lora_ready = bool(not lora or (lora_path and lora_path.is_file()))
    free_bytes = shutil.disk_usage(profile.project_root).free
    free_gib = round(free_bytes / (1024 ** 3), 2)
    recommended_gib = 16 if "xl" in str(checkpoint or "").lower() else 12
    comfy_apps = [Path("/Applications/Comfy Desktop.app"), Path("/Applications/ComfyUI.app")]
    tools = []
    for entry in profile.data.get("toolchain", []):
        raw = Path(entry["path"]).expanduser()
        tools.append({"id": entry["id"], "path": str(raw), "exists": raw.is_file()})
    core_ready = profile.project_root.exists() and all(item["exists"] for item in tools)
    return {
        "ok": core_ready,
        "coreReady": core_ready,
        "assetforgeVersion": __version__,
        "profile": profile.id,
        "profilePath": str(profile.path),
        "profileFingerprint": profile.fingerprint,
        "projectRoot": str(profile.project_root),
        "projectRootExists": profile.project_root.exists(),
        "providers": {
            "comfyLocal": {
                "endpoint": endpoint,
                "reachable": reachable,
                "error": error,
                "workflowTemplate": str(workflow_path) if workflow_path else None,
                "workflowTemplateExists": bool(workflow_path and workflow_path.is_file()),
                "checkpoint": checkpoint,
                "checkpointPath": str(checkpoint_path) if checkpoint_path else None,
                "checkpointExists": checkpoint_ready,
                "lora": lora,
                "loraPath": str(lora_path) if lora_path else None,
                "loraExists": lora_ready,
                "installationPath": str(installation_path),
                "manualInstalled": manual_installed,
                "generationReady": bool(
                    reachable and workflow_path and workflow_path.is_file() and checkpoint_ready and lora_ready
                ),
                "desktopInstalled": any(path.exists() for path in comfy_apps),
                "diskFreeGiB": free_gib,
                "recommendedFreeGiBForInstallAndModel": recommended_gib,
                "installReady": free_gib >= recommended_gib,
            },
            "imagegen": {"role": "concept-and-master-reference-only"},
            "pixellab": {"role": "optional-provider-not-a-runtime-dependency"},
        },
        "toolchain": tools,
    }


def generation_plan(
    profile: Profile,
    character: str,
    tier: str,
    animation: str,
    direction: str,
    provider: str = "comfy_local",
    reference: str | None = None,
) -> dict[str, Any]:
    tier_data = profile.tier(tier)
    animation_data = profile.animation(animation)
    generation = profile.data.get("generation", {})
    provider_config = profile.data.get("providers", {}).get("comfyLocal", {})
    directions = profile.data.get("directions", [direction])
    if direction not in directions and direction not in profile.data.get("mirrorDirections", {}):
        raise ValueError(f"direction {direction!r} is not allowed by profile {profile.id!r}")
    provider_state = doctor(profile)["providers"].get("comfyLocal", {})
    character_prompt = generation.get("characters", {}).get(character, character)
    source_canvas = generation.get("canvas", tier_data.get("canvas")) or [512, 512]
    pixel_canvas = generation.get("pixelCanvas") or [
        max(1, int(source_canvas[0]) // 8),
        max(1, int(source_canvas[1]) // 8),
    ]
    prompt_parts = [
        f"single full-body {character_prompt}",
        generation.get("projection", ""),
        animation_data.get("prompt", animation),
        generation.get("stylePrompt", ""),
        "trace the reference pose and anatomy exactly; preserve the same identity, face, silhouette, costume, colors, proportions and equipment",
        generation.get("backgroundPrompt", ""),
    ]
    prompt = ", ".join(str(part).strip() for part in prompt_parts if str(part).strip())
    plan = {
        "schemaVersion": 1,
        "profile": profile.id,
        "profileFingerprint": profile.fingerprint,
        "project": str(profile.project_root),
        "character": character,
        "kind": profile.kind,
        "tier": tier,
        "animation": animation,
        "direction": direction,
        "provider": provider,
        "reference": str(Path(reference).expanduser().resolve()) if reference else None,
        "contract": {
            "canvasPolicy": tier_data.get("canvasPolicy", "fixed"),
            "canvas": tier_data.get("canvas"),
            "anchor": tier_data.get("anchor"),
            "filtering": tier_data.get("filtering", "nearest"),
            "minFrames": animation_data.get("minFrames", 1),
            "maxFrames": animation_data.get("maxFrames", 1),
            "loop": animation_data.get("loop", False),
            "prompt": prompt,
            "negativePrompt": generation.get("negativePrompt", ""),
            "sourceCanvas": source_canvas,
            "referenceBackground": generation.get("referenceBackground", [236, 244, 241]),
            "referenceFill": float(generation.get("referenceFill", 0.94)),
            "checkpoint": provider_config.get("checkpoint", ""),
            "lora": provider_config.get("lora", ""),
            "loraStrengthModel": float(provider_config.get("loraStrengthModel", 1.0)),
            "loraStrengthClip": float(provider_config.get("loraStrengthClip", 1.0)),
            "steps": int(generation.get("steps", 24)),
            "cfg": float(generation.get("cfg", 6.5)),
            "denoise": float(generation.get("denoise", 0.28)),
            "pixelCanvas": pixel_canvas,
            "seed": int.from_bytes(
                hashlib.sha256(
                    f"{profile.fingerprint}:{character}:{tier}:{animation}:{direction}".encode("utf-8")
                ).digest()[:8],
                "big",
            ) % 2_147_483_647,
        },
        "gates": [
            "reference identity approved",
            "provider output ingested through deterministic canvas and palette normalization",
            "profile validation returns ok=true",
            "human contact-sheet approval before project export",
        ],
        "providerState": provider_state if provider == "comfy_local" else {"configured": False},
    }
    plan["executable"] = bool(provider != "comfy_local" or provider_state.get("generationReady"))
    if not plan["executable"]:
        plan["next"] = "Install/start ComfyUI and configure the profile workflow template; ingest remains usable now."
    return plan


def write_plan(plan: dict[str, Any], path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def _replace_tokens(value: Any, mapping: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _replace_tokens(child, mapping) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_tokens(child, mapping) for child in value]
    if isinstance(value, str):
        if value in mapping:
            return mapping[value]
        for token, replacement in mapping.items():
            value = value.replace(token, str(replacement))
        return value
    return value


TOKEN_PATTERN = re.compile(r"\$\{[A-Z0-9_]+\}")


def _unresolved_tokens(value: Any) -> list[str]:
    return sorted(set(TOKEN_PATTERN.findall(json.dumps(value, ensure_ascii=False))))


def validate_comfy_workflow(workflow: dict[str, Any]) -> None:
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("ComfyUI workflow must be a non-empty API-format node object")
    invalid = []
    for node_id, node in workflow.items():
        if not isinstance(node_id, str) or not isinstance(node, dict):
            invalid.append(str(node_id))
            continue
        if not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), dict):
            invalid.append(node_id)
    if invalid:
        raise ValueError(
            "ComfyUI workflow is not API format; nodes need class_type and inputs: "
            + ", ".join(invalid[:8])
        )


def compile_comfy_request(plan: dict[str, Any], workflow: dict[str, Any], reference_name: str | None = None) -> dict[str, Any]:
    validate_comfy_workflow(workflow)
    canvas = plan["contract"].get("sourceCanvas") or [512, 512]
    resolved_reference = reference_name or (Path(plan["reference"]).name if plan.get("reference") else "")
    if "${REFERENCE_IMAGE}" in json.dumps(workflow) and not resolved_reference:
        raise ValueError("this ComfyUI workflow requires a reference image")
    mapping = {
        "${PROMPT}": plan["contract"].get("prompt", ""),
        "${NEGATIVE_PROMPT}": plan["contract"].get("negativePrompt", ""),
        "${REFERENCE_IMAGE}": resolved_reference,
        "${SEED}": int(plan["contract"].get("seed", 1)),
        "${WIDTH}": int(canvas[0]),
        "${HEIGHT}": int(canvas[1]),
        "${CHECKPOINT}": plan["contract"].get("checkpoint", ""),
        "${LORA}": plan["contract"].get("lora", ""),
        "${LORA_STRENGTH_MODEL}": float(plan["contract"].get("loraStrengthModel", 1.0)),
        "${LORA_STRENGTH_CLIP}": float(plan["contract"].get("loraStrengthClip", 1.0)),
        "${STEPS}": int(plan["contract"].get("steps", 24)),
        "${CFG}": float(plan["contract"].get("cfg", 6.5)),
        "${DENOISE}": float(plan["contract"].get("denoise", 0.28)),
        "${PIXEL_WIDTH}": int((plan["contract"].get("pixelCanvas") or canvas)[0]),
        "${PIXEL_HEIGHT}": int((plan["contract"].get("pixelCanvas") or canvas)[1]),
    }
    compiled = _replace_tokens(workflow, mapping)
    unresolved = _unresolved_tokens(compiled)
    if unresolved:
        raise ValueError(f"unresolved ComfyUI workflow tokens: {', '.join(unresolved)}")
    return {"prompt": compiled, "client_id": f"assetforge-{uuid.uuid4().hex[:12]}"}


def _decode_json_response(response: Any, url: str) -> dict[str, Any]:
    try:
        return strict_json_loads(response.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid JSON response from ComfyUI: {url}") from exc


def _post_json(url: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _decode_json_response(response, url)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"ComfyUI HTTP {exc.code} at {url}: {body}") from exc


def _get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return _decode_json_response(response, url)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"ComfyUI HTTP {exc.code} at {url}: {body}") from exc


def prepare_comfy_reference(plan: dict[str, Any], output_dir: str | Path) -> Path:
    reference = plan.get("reference")
    if not reference:
        raise ValueError("the generation plan has no reference image")
    source_path = Path(reference).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"reference image not found: {source_path}")
    canvas = plan["contract"].get("sourceCanvas") or [512, 512]
    width, height = int(canvas[0]), int(canvas[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid source canvas: {canvas}")
    background = plan["contract"].get("referenceBackground", [236, 244, 241])
    if not isinstance(background, list) or len(background) != 3:
        raise ValueError("referenceBackground must be an RGB array")
    color = tuple(max(0, min(255, int(channel))) for channel in background) + (255,)
    with Image.open(source_path) as opened:
        source = opened.convert("RGBA")
    fill = max(0.1, min(1.0, float(plan["contract"].get("referenceFill", 0.94))))
    scale = min((width * fill) / source.width, (height * fill) / source.height)
    if abs(scale - 1.0) > 1e-9:
        source = source.resize(
            (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
            Image.Resampling.NEAREST,
        )
    prepared = Image.new("RGBA", (width, height), color)
    prepared.alpha_composite(source, ((width - source.width) // 2, (height - source.height) // 2))
    # ComfyUI keeps uploaded inputs in a shared folder.  A fixed filename lets
    # concurrent character jobs overwrite each other's reference before the
    # queued workflow reads it, so key the name by the prepared pixel content.
    reference_digest = hashlib.sha256(prepared.tobytes()).hexdigest()[:16]
    target = (
        Path(output_dir).expanduser().resolve()
        / "_input"
        / f"reference-{reference_digest}.png"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    prepared.convert("RGB").save(target)
    return target


def is_runtime_pixel_reference(
    image_path: str | Path,
    max_side: int = 512,
    max_colors: int = 256,
    min_alpha: int = 20,
) -> bool:
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"reference image not found: {path}")
    with Image.open(path) as opened:
        image = opened.convert("RGBA")
    if max(image.size) > max_side:
        return False
    colors: set[tuple[int, int, int]] = set()
    has_transparency = False
    pixels = (
        image.get_flattened_data()
        if hasattr(image, "get_flattened_data")
        else image.getdata()
    )
    for red, green, blue, alpha in pixels:
        if alpha < 255:
            has_transparency = True
        if alpha > min_alpha:
            colors.add((red, green, blue))
            if len(colors) > max_colors:
                return False
    return has_transparency and bool(colors)


def preserve_reference_frame(plan: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    reference = plan.get("reference")
    if not reference:
        raise ValueError("the generation plan has no reference image")
    source = Path(reference).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"reference image not found: {source}")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    animation = str(plan["animation"])
    for stale in output.glob(f"{animation}_*.png"):
        stale.unlink()
    target = output / f"{animation}_00.png"
    with Image.open(source) as opened:
        opened.convert("RGBA").save(target)
    result = {
        "ok": True,
        "mode": "reference-preserve",
        "profile": plan["profile"],
        "reference": str(source),
        "output": str(output),
        "images": [{"path": str(target), "sourceFilename": source.name}],
    }
    manifest = output / "_reference-preserve.json"
    result["manifest"] = str(manifest)
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _multipart_body(fields: dict[str, str], file_path: Path) -> tuple[str, bytes]:
    boundary = f"assetforge-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    safe_name = file_path.name.replace('"', "")
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="image"; filename="{safe_name}"\r\n'.encode(),
            b"Content-Type: image/png\r\n\r\n",
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return boundary, b"".join(chunks)


def upload_comfy_image(profile: Profile, image_path: str | Path) -> dict[str, Any]:
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"upload image not found: {path}")
    endpoint = _comfy_endpoint(profile).rstrip("/")
    boundary, body = _multipart_body(
        {"type": "input", "subfolder": "assetforge", "overwrite": "true"}, path
    )
    url = f"{endpoint}/upload/image"
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = _decode_json_response(response, url)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"ComfyUI upload HTTP {exc.code}: {error_body}") from exc
    name = result.get("name")
    if not name:
        raise RuntimeError(f"ComfyUI upload returned no image name: {result}")
    subfolder = str(result.get("subfolder", "")).strip("/")
    result["referenceName"] = f"{subfolder}/{name}" if subfolder else str(name)
    return result


def wait_for_comfy_result(
    profile: Profile,
    prompt_id: str,
    timeout: float = 300,
    poll_interval: float = 0.75,
) -> dict[str, Any]:
    endpoint = _comfy_endpoint(profile).rstrip("/")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = _get_json(f"{endpoint}/history/{urllib.parse.quote(prompt_id)}", timeout=30)
        record = history.get(prompt_id)
        if record:
            status = record.get("status", {})
            status_text = status.get("status_str")
            if status_text == "error":
                messages = status.get("messages", [])
                raise RuntimeError(f"ComfyUI generation failed: {messages[-1] if messages else status}")
            outputs = record.get("outputs", {})
            if outputs and status.get("completed", False):
                return record
        time.sleep(max(0.1, poll_interval))
    raise TimeoutError(f"ComfyUI generation timed out after {timeout:.0f}s (prompt {prompt_id})")


def download_comfy_outputs(
    profile: Profile,
    history: dict[str, Any],
    output_dir: str | Path,
    stem: str,
) -> list[dict[str, str]]:
    endpoint = _comfy_endpoint(profile).rstrip("/")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, Any]] = []
    node_ids = sorted(
        history.get("outputs", {}),
        key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)),
    )
    for node_id in node_ids:
        for item in history["outputs"][node_id].get("images", []):
            images.append({"nodeId": str(node_id), **item})
    if not images:
        raise RuntimeError("ComfyUI completed without downloadable image outputs")
    downloaded: list[dict[str, str]] = []
    for index, item in enumerate(images):
        query = urllib.parse.urlencode(
            {
                "filename": item.get("filename", ""),
                "subfolder": item.get("subfolder", ""),
                "type": item.get("type", "output"),
            }
        )
        url = f"{endpoint}/view?{query}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"ComfyUI output download HTTP {exc.code}: {url}") from exc
        suffix = Path(str(item.get("filename", "output.png"))).suffix.lower()
        if suffix not in {".png", ".webp", ".jpg", ".jpeg"}:
            suffix = ".png"
        target = output / f"{stem}_{index:02d}{suffix}"
        target.write_bytes(body)
        downloaded.append(
            {
                "path": str(target),
                "nodeId": str(item.get("nodeId", "")),
                "sourceFilename": str(item.get("filename", "")),
            }
        )
    return downloaded


def submit_comfy_request(profile: Profile, request: dict[str, Any]) -> dict[str, Any]:
    state = doctor(profile)["providers"]["comfyLocal"]
    if not state["reachable"]:
        raise RuntimeError(f"ComfyUI is not reachable at {state['endpoint']}")
    result = _post_json(f"{state['endpoint'].rstrip('/')}/prompt", request)
    if result.get("node_errors"):
        raise RuntimeError(f"ComfyUI rejected workflow nodes: {result['node_errors']}")
    if not result.get("prompt_id"):
        raise RuntimeError(f"ComfyUI returned no prompt_id: {result}")
    return {"ok": True, "endpoint": state["endpoint"], "result": result}


def run_comfy_request(
    profile: Profile,
    plan: dict[str, Any],
    workflow: dict[str, Any],
    output_dir: str | Path,
    timeout: float = 300,
    poll_interval: float = 0.75,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob(f"{plan['animation']}_*.png"):
        stale.unlink()
    prepared = prepare_comfy_reference(plan, output)
    uploaded = upload_comfy_image(profile, prepared)
    request = compile_comfy_request(plan, workflow, uploaded["referenceName"])
    request_path = output / "_request.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    submitted = submit_comfy_request(profile, request)
    prompt_id = submitted["result"]["prompt_id"]
    history = wait_for_comfy_result(profile, prompt_id, timeout, poll_interval)
    downloaded = download_comfy_outputs(profile, history, output, str(plan["animation"]))
    result = {
        "ok": True,
        "profile": profile.id,
        "promptId": prompt_id,
        "endpoint": submitted["endpoint"],
        "preparedReference": str(prepared),
        "uploadedReference": uploaded["referenceName"],
        "request": str(request_path),
        "output": str(output),
        "images": downloaded,
    }
    manifest = output / "_comfy-run.json"
    result["manifest"] = str(manifest)
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
