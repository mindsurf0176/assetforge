from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .exporters import export_assets
from .frames import ingest_frames
from .json_utils import strict_json_loads
from .local_animation import (
    parse_clips,
    parse_frame_counts,
    run_animation_spec,
    run_local_animation,
)
from .profile import ProfileError, list_profiles, load_profile
from .providers import (
    compile_comfy_request,
    doctor,
    generation_plan,
    is_runtime_pixel_reference,
    load_comfy_workflow,
    preserve_reference_frame,
    run_comfy_request,
    submit_comfy_request,
    write_plan,
)
from .validation import validate_frames
from .rig_build import archetypes, extract_part_sheet


def emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="assetforge", description="Regulated 2D game asset factory")
    root.add_argument("--version", action="version", version=f"AssetForge {__version__}")
    sub = root.add_subparsers(dest="command", required=True)

    sub.add_parser("profiles", help="list installed project profiles")
    sub.add_parser("rig-archetypes", help="list local cutout rig archetypes")

    p = sub.add_parser("rig-extract", help="extract disconnected parts from one part-sheet PNG")
    p.add_argument("--sheet", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--background-tolerance", type=int, default=42)
    p.add_argument("--min-area-ratio", type=float, default=0.0005)

    p = sub.add_parser(
        "animate",
        help="generate real sprite frames locally with a deterministic cutout rig",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", help="strict AnimationSpec JSON")
    source.add_argument("--rig", help="compiled RigSpec JSON")
    source.add_argument("--parts", help="directory of named transparent part PNGs")
    source.add_argument("--part-sheet", help="one PNG containing disconnected rig parts")
    source.add_argument(
        "--reference",
        help="one assembled character PNG; assisted coarse auto-rig with explicit limitations",
    )
    p.add_argument("--mapping", help="blob-to-slot JSON required by --part-sheet")
    p.add_argument("--archetype", choices=archetypes())
    p.add_argument("--character")
    p.add_argument("--direction")
    p.add_argument("--clips")
    p.add_argument("--frames", help="per-clip overrides, for example idle=4,walk=8")
    p.add_argument("--height", type=int)
    p.add_argument("--resample", choices=("nearest", "bicubic"))
    p.add_argument("--work", help="build directory; required unless --spec supplies the default")
    p.add_argument("--profile", help="optional AssetForge profile for normalize/validate/export")
    p.add_argument("--tier", help="profile tier, required with --profile")
    p.add_argument("--resource-prefix", help="base web path or res:// path for engine export")
    p.add_argument("--deploy-dir", help="explicit game asset root; requires --resource-prefix")
    p.add_argument("--no-mirror", action="store_true", help="do not synthesize missing back-side limbs")

    p = sub.add_parser("doctor", help="check profile, project, provider and toolchain readiness")
    p.add_argument("--profile", required=True)

    p = sub.add_parser("plan", help="write a provider-independent generation plan")
    p.add_argument("--profile", required=True)
    p.add_argument("--character", required=True)
    p.add_argument("--tier", required=True)
    p.add_argument("--animation", required=True)
    p.add_argument("--direction", required=True)
    p.add_argument("--provider", default="comfy_local")
    p.add_argument("--reference")
    p.add_argument("--write")

    p = sub.add_parser("ingest", help="normalize generated frames into the profile contract")
    p.add_argument("--profile", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tier", required=True)
    p.add_argument("--animation", required=True)
    p.add_argument("--direction", required=True)

    p = sub.add_parser("validate", help="validate normalized frames against a profile")
    p.add_argument("--profile", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--tier", required=True)
    p.add_argument("--animation")
    p.add_argument("--report")

    p = sub.add_parser("export", help="export normalized frames for the profile engine")
    p.add_argument("--profile", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--character", required=True)
    p.add_argument("--tier", required=True)
    p.add_argument("--animation", required=True)
    p.add_argument("--direction", required=True)
    p.add_argument("--resource-prefix")
    p.add_argument(
        "--deploy-dir",
        help="explicit frame deployment directory; omitted builds an output-adjacent artifact",
    )

    p = sub.add_parser("build", help="ingest, validate and export in one deterministic pass")
    p.add_argument("--profile", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--character", required=True)
    p.add_argument("--tier", required=True)
    p.add_argument("--animation", required=True)
    p.add_argument("--direction", required=True)
    p.add_argument("--resource-prefix")
    p.add_argument(
        "--deploy-dir",
        help="explicit frame deployment directory; omitted builds an output-adjacent artifact",
    )

    p = sub.add_parser("comfy-compile", help="compile a ComfyUI API workflow with an AssetForge plan")
    p.add_argument("--profile", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--workflow", help="override the profile's configured API workflow")
    p.add_argument("--output", required=True)

    p = sub.add_parser("comfy-submit", help="submit a compiled ComfyUI request; requires --execute")
    p.add_argument("--profile", required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--execute", action="store_true")

    p = sub.add_parser("comfy-run", help="upload, generate, wait and download through local ComfyUI")
    p.add_argument("--profile", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--workflow", help="override the profile's configured API workflow")
    p.add_argument("--output", required=True)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--poll-interval", type=float, default=0.75)
    p.add_argument("--execute", action="store_true")

    p = sub.add_parser(
        "comfy-build",
        help="plan, generate, normalize, validate and export a local ComfyUI asset",
    )
    p.add_argument("--profile", required=True)
    p.add_argument("--character", required=True)
    p.add_argument("--tier", required=True)
    p.add_argument("--animation", required=True)
    p.add_argument("--direction", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument(
        "--reference-mode",
        choices=["auto", "diffuse", "preserve"],
        default="auto",
        help="auto preserves low-color runtime pixel art and diffuses high-resolution masters",
    )
    p.add_argument("--raw", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workflow", help="override the profile's configured API workflow")
    p.add_argument("--resource-prefix")
    p.add_argument(
        "--deploy-dir",
        help="explicit frame deployment directory; omitted builds an output-adjacent artifact",
    )
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--poll-interval", type=float, default=0.75)
    p.add_argument("--execute", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "profiles":
            emit({"profiles": list_profiles()})
            return 0
        if args.command == "rig-archetypes":
            emit({"archetypes": list(archetypes())})
            return 0
        if args.command == "rig-extract":
            emit(
                extract_part_sheet(
                    args.sheet,
                    args.output,
                    background_tolerance=args.background_tolerance,
                    min_area_ratio=args.min_area_ratio,
                )
            )
            return 0
        if args.command == "animate":
            if args.spec:
                if any(
                    value is not None
                    for value in (
                        args.character,
                        args.direction,
                        args.clips,
                        args.frames,
                        args.height,
                        args.resample,
                        args.profile,
                        args.tier,
                        args.mapping,
                        args.archetype,
                        args.resource_prefix,
                        args.deploy_dir,
                    )
                ) or args.no_mirror:
                    raise ValueError(
                        "--spec owns all animation settings; only --work may override "
                        "its build directory"
                    )
                result = run_animation_spec(args.spec, args.work)
            else:
                if not args.character:
                    raise ValueError("--character is required unless --spec is used")
                if not args.work:
                    raise ValueError("--work is required unless --spec is used")
                if args.rig and args.height is not None:
                    raise ValueError("--height cannot override a compiled --rig")
                result = run_local_animation(
                    work_dir=args.work,
                    character=args.character,
                    direction=args.direction or "east",
                    clips=(parse_clips(args.clips) if args.clips is not None else None),
                    frame_overrides=parse_frame_counts(args.frames),
                    rig_path=args.rig,
                    parts_dir=args.parts,
                    part_sheet=args.part_sheet,
                    reference=args.reference,
                    mapping=args.mapping,
                    archetype=args.archetype,
                    height=512 if args.height is None else args.height,
                    resample=args.resample or "bicubic",
                    profile_name=args.profile,
                    tier=args.tier,
                    resource_prefix=args.resource_prefix,
                    deploy_dir=args.deploy_dir,
                    no_mirror=args.no_mirror,
                )
            emit(result)
            return 0 if result.get("ok") else 1
        profile = load_profile(args.profile)
        if args.command == "doctor":
            result = doctor(profile)
            emit(result)
            return 0 if result["ok"] else 1
        if args.command == "plan":
            result = generation_plan(
                profile, args.character, args.tier, args.animation, args.direction, args.provider, args.reference
            )
            if args.write:
                result["writtenTo"] = str(write_plan(result, args.write))
            emit(result)
            return 0
        if args.command == "ingest":
            emit(ingest_frames(profile, args.input, args.output, args.tier, args.animation, args.direction))
            return 0
        if args.command == "validate":
            result = validate_frames(profile, args.input, args.tier, args.animation, args.report)
            emit(result)
            return 0 if result["ok"] else 1
        if args.command == "export":
            emit(
                export_assets(
                    profile,
                    args.input,
                    args.output,
                    args.character,
                    args.tier,
                    args.animation,
                    args.direction,
                    args.resource_prefix,
                    args.deploy_dir,
                )
            )
            return 0
        if args.command == "build":
            manifest = ingest_frames(
                profile, args.input, args.work, args.tier, args.animation, args.direction
            )
            report_path = Path(args.work).expanduser().resolve() / "validation-report.json"
            validation = validate_frames(profile, args.work, args.tier, args.animation, report_path)
            if not validation["ok"]:
                emit({"ok": False, "stage": "validate", "manifest": manifest, "validation": validation})
                return 1
            exported = export_assets(
                profile,
                args.work,
                args.output,
                args.character,
                args.tier,
                args.animation,
                args.direction,
                args.resource_prefix,
                args.deploy_dir,
            )
            emit({"ok": True, "manifest": manifest, "validation": validation, "export": exported})
            return 0
        if args.command == "comfy-compile":
            plan_data = strict_json_loads(Path(args.plan).expanduser().read_text(encoding="utf-8"))
            workflow_path, workflow_data = load_comfy_workflow(profile, args.workflow)
            request = compile_comfy_request(plan_data, workflow_data)
            target = Path(args.output).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            emit(
                {
                    "ok": True,
                    "output": str(target),
                    "workflow": str(workflow_path),
                    "nodes": len(request["prompt"]),
                }
            )
            return 0
        if args.command == "comfy-submit":
            request_data = strict_json_loads(Path(args.request).expanduser().read_text(encoding="utf-8"))
            if not args.execute:
                emit({"ok": True, "dryRun": True, "profile": profile.id, "nodes": len(request_data["prompt"])})
                return 0
            emit(submit_comfy_request(profile, request_data))
            return 0
        if args.command == "comfy-run":
            plan_data = strict_json_loads(Path(args.plan).expanduser().read_text(encoding="utf-8"))
            workflow_path, workflow_data = load_comfy_workflow(profile, args.workflow)
            if not args.execute:
                reference_name = Path(plan_data["reference"]).name if plan_data.get("reference") else None
                request = compile_comfy_request(plan_data, workflow_data, reference_name)
                emit(
                    {
                        "ok": True,
                        "dryRun": True,
                        "profile": profile.id,
                        "workflow": str(workflow_path),
                        "nodes": len(request["prompt"]),
                        "output": str(Path(args.output).expanduser().resolve()),
                    }
                )
                return 0
            result = run_comfy_request(
                profile,
                plan_data,
                workflow_data,
                args.output,
                args.timeout,
                args.poll_interval,
            )
            result["workflow"] = str(workflow_path)
            emit(result)
            return 0
        if args.command == "comfy-build":
            plan_data = generation_plan(
                profile,
                args.character,
                args.tier,
                args.animation,
                args.direction,
                "comfy_local",
                args.reference,
            )
            if int(plan_data["contract"].get("minFrames", 1)) > 1:
                raise ValueError(
                    f"{profile.id}:{args.animation} requires at least "
                    f"{plan_data['contract']['minFrames']} coherent frames; the reference img2img workflow "
                    "currently builds single-frame masters only"
                )
            reference_mode = args.reference_mode
            if reference_mode == "auto":
                reference_mode = (
                    "preserve" if is_runtime_pixel_reference(args.reference) else "diffuse"
                )
            plan_data["contract"]["referenceMode"] = reference_mode
            workflow_path = None
            workflow_data = None
            if reference_mode == "preserve":
                plan_data["provider"] = "reference_preserve"
                plan_data["providerState"] = {"configured": True, "modelBypassed": True}
                plan_data["executable"] = True
            else:
                workflow_path, workflow_data = load_comfy_workflow(profile, args.workflow)
            if not args.execute:
                request = (
                    compile_comfy_request(plan_data, workflow_data, Path(args.reference).name)
                    if workflow_data is not None
                    else None
                )
                emit(
                    {
                        "ok": True,
                        "dryRun": True,
                        "profile": profile.id,
                        "referenceMode": reference_mode,
                        "workflow": str(workflow_path) if workflow_path else None,
                        "nodes": len(request["prompt"]) if request else 0,
                        "raw": str(Path(args.raw).expanduser().resolve()),
                        "work": str(Path(args.work).expanduser().resolve()),
                        "output": str(Path(args.output).expanduser().resolve()),
                        "seed": plan_data["contract"]["seed"],
                    }
                )
                return 0
            raw_path = Path(args.raw).expanduser().resolve()
            plan_path = raw_path / "_plan.json"
            write_plan(plan_data, plan_path)
            generated = (
                preserve_reference_frame(plan_data, raw_path)
                if reference_mode == "preserve"
                else run_comfy_request(
                    profile,
                    plan_data,
                    workflow_data,
                    raw_path,
                    args.timeout,
                    args.poll_interval,
                )
            )
            manifest = ingest_frames(
                profile,
                raw_path,
                args.work,
                args.tier,
                args.animation,
                args.direction,
            )
            report_path = Path(args.work).expanduser().resolve() / "validation-report.json"
            validation = validate_frames(profile, args.work, args.tier, args.animation, report_path)
            if not validation["ok"]:
                emit(
                    {
                        "ok": False,
                        "stage": "validate",
                        "generation": generated,
                        "manifest": manifest,
                        "validation": validation,
                    }
                )
                return 1
            exported = export_assets(
                profile,
                args.work,
                args.output,
                args.character,
                args.tier,
                args.animation,
                args.direction,
                args.resource_prefix,
                args.deploy_dir,
            )
            emit(
                {
                    "ok": True,
                    "referenceMode": reference_mode,
                    "workflow": str(workflow_path) if workflow_path else None,
                    "plan": str(plan_path),
                    "generation": generated,
                    "manifest": manifest,
                    "validation": validation,
                    "export": exported,
                }
            )
            return 0
    except (ProfileError, FileNotFoundError, ValueError, RuntimeError, OSError, ArithmeticError) as exc:
        emit({"ok": False, "error": str(exc)})
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
