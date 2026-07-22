from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .exporters import export_assets
from .frames import ingest_frames
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


def emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="assetforge", description="Regulated 2D game asset factory")
    root.add_argument("--version", action="version", version=f"AssetForge {__version__}")
    sub = root.add_subparsers(dest="command", required=True)

    sub.add_parser("profiles", help="list installed project profiles")

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
            plan_data = json.loads(Path(args.plan).expanduser().read_text(encoding="utf-8"))
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
            request_data = json.loads(Path(args.request).expanduser().read_text(encoding="utf-8"))
            if not args.execute:
                emit({"ok": True, "dryRun": True, "profile": profile.id, "nodes": len(request_data["prompt"])})
                return 0
            emit(submit_comfy_request(profile, request_data))
            return 0
        if args.command == "comfy-run":
            plan_data = json.loads(Path(args.plan).expanduser().read_text(encoding="utf-8"))
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
    except (ProfileError, FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        emit({"ok": False, "error": str(exc)})
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
