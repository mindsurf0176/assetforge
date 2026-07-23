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
from .mflux_backend import (
    DEFAULT_BASE_MODEL as MFLUX_DEFAULT_BASE_MODEL,
    DEFAULT_MODEL as MFLUX_DEFAULT_MODEL,
    DEFAULT_PROMPT as MFLUX_DEFAULT_PROMPT,
    mflux_doctor,
    paired_board_edit_plan,
    run_mflux_plan,
)
from .mflux_checkpoint import extract_mflux_lora_adapter
from .mflux_training import (
    MIN_LOCAL_TRAINING_FREE_DISK_GIB,
    build_mflux_training_plan,
    compile_mflux_train_command,
    create_portable_training_bundle,
    mflux_training_doctor,
    prepare_training_data,
    run_mflux_training_plan,
    write_mflux_training_config,
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
from .redraw_dataset import build_redraw_dataset
from .redraw_delivery import evaluate_redraw_holdout_batch, export_redraw_board_frames
from .redraw_quality import evaluate_redraw_sample


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

    p = sub.add_parser(
        "redraw-dataset",
        help="build paired identity-plus-pose boards for a local full-frame redraw model",
    )
    p.add_argument("--spec", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("redraw-quality", help="grade a generated board against one dataset holdout")
    p.add_argument("--manifest", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--generated", required=True)

    p = sub.add_parser(
        "redraw-quality-batch",
        help="grade one generated <sample-id>.png board for every validation holdout",
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--generated-dir", required=True)

    p = sub.add_parser(
        "redraw-board-export",
        help="split one passing redraw board into native transparent sprite frames",
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--generated", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--background-tolerance", type=int, default=42)

    p = sub.add_parser("mflux-doctor", help="check local FLUX.2 edit inference readiness")
    p.add_argument("--executable")
    p.add_argument("--model", default=MFLUX_DEFAULT_MODEL)
    p.add_argument("--base-model", default=MFLUX_DEFAULT_BASE_MODEL)
    p.add_argument("--model-path")
    p.add_argument("--cache-dir")
    p.add_argument("--lora", action="append")
    p.add_argument("--disk-path")
    p.add_argument("--minimum-free-gib", type=float, default=6.0)

    p = sub.add_parser(
        "mflux-redraw",
        help="plan or execute one local FLUX.2 full-board redraw",
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prompt", default=MFLUX_DEFAULT_PROMPT)
    p.add_argument("--executable")
    p.add_argument("--model", default=MFLUX_DEFAULT_MODEL)
    p.add_argument("--base-model", default=MFLUX_DEFAULT_BASE_MODEL)
    p.add_argument("--model-path")
    p.add_argument("--cache-dir")
    p.add_argument("--lora", action="append")
    p.add_argument("--lora-scale", action="append", type=float)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--quantize", type=int, choices=(3, 4, 5, 6, 8))
    p.add_argument(
        "--low-ram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use MFLUX's lower-memory inference path (default: enabled)",
    )
    p.add_argument("--mlx-cache-limit-gib", type=float, default=2.5)
    p.add_argument("--metadata", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--minimum-free-gib", type=float, default=6.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--execute", action="store_true")

    p = sub.add_parser(
        "mflux-train-doctor",
        help="check local FLUX.2 edit-LoRA training readiness without loading the model",
    )
    p.add_argument("--model-path", required=True)
    p.add_argument("--executable")
    p.add_argument("--data-path")
    p.add_argument("--checkpoint-path")
    p.add_argument("--cache-path")
    p.add_argument(
        "--minimum-free-disk-gib",
        type=float,
        default=MIN_LOCAL_TRAINING_FREE_DISK_GIB,
    )

    p = sub.add_parser(
        "mflux-train-prepare",
        help="copy a deterministic train-only subset for an edit-LoRA smoke test",
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sample-limit", required=True, type=int)

    p = sub.add_parser(
        "mflux-train-bundle",
        help="export a byte-pinned train-only bundle for another MFLUX host",
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    bundle_model = p.add_mutually_exclusive_group(required=True)
    bundle_model.add_argument("--model-path")
    bundle_model.add_argument(
        "--model-lock",
        help="trusted unquantized model fingerprint JSON when the large model stays remote",
    )

    p = sub.add_parser(
        "mflux-train-plan",
        help="validate, dry-run, or explicitly train a FLUX.2 edit-LoRA",
    )
    training_source = p.add_mutually_exclusive_group(required=True)
    training_source.add_argument("--manifest")
    training_source.add_argument(
        "--bundle",
        help="portable bundle directory or assetforge-mflux-bundle.json copied from another host",
    )
    p.add_argument(
        "--expected-bundle-sha256",
        help="required with --bundle; compare against the hash printed by mflux-train-bundle",
    )
    p.add_argument("--model-path", required=True)
    p.add_argument("--config-output", required=True)
    p.add_argument("--prepared-data-path")
    p.add_argument("--sample-limit", type=int)
    p.add_argument("--checkpoint-output")
    p.add_argument("--executable")
    p.add_argument(
        "--low-ram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="must remain disabled; managed training rejects MFLUX's recursive cache cleanup",
    )
    p.add_argument("--max-resolution", type=int, default=576)
    schedule = p.add_mutually_exclusive_group()
    schedule.add_argument(
        "--target-updates",
        type=int,
        help="derive whole epochs to reach at least this many optimizer updates (default: 1500)",
    )
    schedule.add_argument(
        "--epochs",
        type=int,
        help="use an explicit epoch count instead of the target-update schedule",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--checkpoint-frequency", type=int, default=250)
    p.add_argument("--plot-frequency", type=int, default=25)
    p.add_argument("--generate-image-frequency", type=int, default=250)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--minimum-free-disk-gib",
        type=float,
        default=MIN_LOCAL_TRAINING_FREE_DISK_GIB,
    )
    p.add_argument(
        "--write-config",
        action="store_true",
        help="write the validated config with exclusive creation",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="on a compatible host, run parser validation and then start training",
    )

    p = sub.add_parser(
        "mflux-train-extract",
        help="safely extract the manifest-selected LoRA from an MFLUX checkpoint",
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
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
        if args.command == "redraw-dataset":
            emit(build_redraw_dataset(args.spec, args.output))
            return 0
        if args.command == "redraw-quality":
            result = evaluate_redraw_sample(args.manifest, args.sample, args.generated)
            emit(result)
            return 0 if result["ok"] else 1
        if args.command == "redraw-quality-batch":
            result = evaluate_redraw_holdout_batch(args.manifest, args.generated_dir)
            emit(result)
            return 0 if result["ok"] else 1
        if args.command == "redraw-board-export":
            result = export_redraw_board_frames(
                args.manifest,
                args.sample,
                args.generated,
                args.output,
                background_tolerance=args.background_tolerance,
            )
            emit(result)
            return 0
        if args.command == "mflux-doctor":
            result = mflux_doctor(
                executable=args.executable,
                model=args.model,
                base_model=args.base_model,
                model_path=args.model_path,
                cache_dir=args.cache_dir,
                lora=args.lora,
                disk_path=args.disk_path,
                minimum_free_gib=args.minimum_free_gib,
            )
            emit(result)
            return 0 if result["ok"] else 1
        if args.command == "mflux-redraw":
            lora_scale: float | list[float] = args.lora_scale or 1.0
            plan_data = paired_board_edit_plan(
                args.input,
                args.output,
                prompt=args.prompt,
                executable=args.executable,
                model=args.model,
                base_model=args.base_model,
                model_path=args.model_path,
                cache_dir=args.cache_dir,
                lora=args.lora,
                lora_scale=lora_scale,
                seed=args.seed,
                steps=args.steps,
                guidance=args.guidance,
                quantize=args.quantize,
                low_ram=args.low_ram,
                mlx_cache_limit_gib=args.mlx_cache_limit_gib,
                metadata=args.metadata,
                overwrite=args.overwrite,
                minimum_free_gib=args.minimum_free_gib,
            )
            if not args.execute:
                plan_data["dryRun"] = True
                emit(plan_data)
                return 0 if plan_data["ready"] else 1
            emit(run_mflux_plan(plan_data, execute=True))
            return 0
        if args.command == "mflux-train-doctor":
            result = mflux_training_doctor(
                model_path=args.model_path,
                executable=args.executable,
                data_path=args.data_path,
                checkpoint_path=args.checkpoint_path,
                cache_path=args.cache_path,
                minimum_free_disk_gib=args.minimum_free_disk_gib,
            )
            emit(result)
            return 0 if result["localTrainingReady"] else 1
        if args.command == "mflux-train-prepare":
            result = prepare_training_data(
                args.manifest,
                args.output,
                sample_limit=args.sample_limit,
            )
            emit(result)
            return 0
        if args.command == "mflux-train-bundle":
            emit(
                create_portable_training_bundle(
                    args.manifest,
                    args.output,
                    model_path=args.model_path,
                    model_lock_path=args.model_lock,
                )
            )
            return 0
        if args.command == "mflux-train-plan":
            plan_data = build_mflux_training_plan(
                args.manifest,
                portable_bundle=args.bundle,
                expected_bundle_sha256=args.expected_bundle_sha256,
                model_path=args.model_path,
                config_output=args.config_output,
                prepared_data_path=args.prepared_data_path,
                sample_limit=args.sample_limit,
                checkpoint_output=args.checkpoint_output,
                executable=args.executable,
                low_ram=args.low_ram,
                max_resolution=args.max_resolution,
                epochs=args.epochs,
                target_updates=args.target_updates,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                checkpoint_frequency=args.checkpoint_frequency,
                plot_frequency=args.plot_frequency,
                generate_image_frequency=args.generate_image_frequency,
                lora_rank=args.lora_rank,
                seed=args.seed,
                minimum_free_disk_gib=args.minimum_free_disk_gib,
                allow_existing_config=args.execute,
            )
            config_path = Path(plan_data["configOutput"])
            config_exists = config_path.is_file() and not config_path.is_symlink()
            if args.execute and not args.write_config and not config_exists:
                raise ValueError(
                    "--execute requires --write-config or an existing config that exactly "
                    "matches the newly validated plan"
                )
            if args.execute and not plan_data["ready"]:
                plan_data["writtenConfig"] = (
                    str(config_path.resolve()) if plan_data["configReused"] else None
                )
                plan_data["dryRunCommand"] = None
                plan_data["training"] = None
                emit(plan_data)
                return 1
            if args.write_config or args.execute:
                plan_data["writtenConfig"] = (
                    str(config_path.resolve())
                    if plan_data["configReused"]
                    else str(write_mflux_training_config(plan_data))
                )
                plan_data["dryRunCommand"] = compile_mflux_train_command(plan_data)
                plan_data["training"] = (
                    run_mflux_training_plan(plan_data, execute=True)
                    if args.execute
                    else None
                )
            else:
                plan_data["writtenConfig"] = None
                plan_data["dryRunCommand"] = None
                plan_data["training"] = None
            emit(plan_data)
            return 0 if plan_data["ready"] else 1
        if args.command == "mflux-train-extract":
            emit(extract_mflux_lora_adapter(args.checkpoint, args.output))
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
