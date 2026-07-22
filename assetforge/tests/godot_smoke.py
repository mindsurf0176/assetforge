#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Load an AssetForge SpriteFrames export in a scratch Godot project")
    parser.add_argument("--resource", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument(
        "--artifact-root",
        help="copy a self-contained export tree and load the resource at its original relative path",
    )
    parser.add_argument("--animation", required=True)
    parser.add_argument("--expected", required=True, type=int)
    parser.add_argument("--godot", default="godot")
    parser.add_argument("--report")
    args = parser.parse_args()

    resource = Path(args.resource).resolve()
    frame_dir = Path(args.frames).resolve()
    if not resource.is_file() or not frame_dir.is_dir():
        raise SystemExit("resource or frame directory missing")

    with tempfile.TemporaryDirectory(prefix="assetforge-godot-") as temp:
        root = Path(temp)
        if args.artifact_root:
            artifact_root = Path(args.artifact_root).expanduser().resolve()
            if not artifact_root.is_dir() or not resource.is_relative_to(artifact_root):
                raise SystemExit("resource must be inside artifact root")
            shutil.copytree(artifact_root, root, dirs_exist_ok=True)
            resource_target = root / resource.relative_to(artifact_root)
        else:
            (root / "frames").mkdir()
            for path in frame_dir.glob("*.png"):
                if not path.name.startswith("_"):
                    shutil.copy2(path, root / "frames" / path.name)
            resource_target = root / "asset.tres"
            shutil.copy2(resource, resource_target)
        (root / "project.godot").write_text(
            '[application]\nconfig/name="AssetForge smoke"\n[rendering]\nrenderer/rendering_method="gl_compatibility"\n',
            encoding="utf-8",
        )
        (root / "smoke.gd").write_text(
            "extends SceneTree\n"
            "func _initialize():\n"
            f"    var frames = load(\"res://{resource_target.relative_to(root).as_posix()}\")\n"
            "    if frames == null:\n"
            "        push_error(\"AssetForge SpriteFrames failed to load\")\n"
            "        quit(2)\n"
            f"    var count = frames.get_frame_count(&\"{args.animation}\")\n"
            f"    if count != {args.expected}:\n"
            "        push_error(\"unexpected frame count: %s\" % count)\n"
            "        quit(3)\n"
            "    print(\"ASSETFORGE_GODOT_OK frames=%s\" % count)\n"
            "    quit(0)\n",
            encoding="utf-8",
        )
        imported = subprocess.run(
            [args.godot, "--headless", "--path", str(root), "--import"],
            text=True,
            capture_output=True,
            timeout=120,
        )
        process = subprocess.run(
            [args.godot, "--headless", "--path", str(root), "--script", "smoke.gd"],
            text=True,
            capture_output=True,
            timeout=120,
        )
        result = {
            "ok": (
                imported.returncode == 0
                and process.returncode == 0
                and "ASSETFORGE_GODOT_OK" in process.stdout
                and "ERROR:" not in imported.stderr
                and "ERROR:" not in process.stderr
            ),
            "importReturncode": imported.returncode,
            "importStdout": imported.stdout.strip(),
            "importStderr": imported.stderr.strip(),
            "returncode": process.returncode,
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip(),
        }
        if args.report:
            report = Path(args.report).expanduser().resolve()
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
