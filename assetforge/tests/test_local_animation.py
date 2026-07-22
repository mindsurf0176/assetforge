from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from copy import deepcopy
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from assetforge.cli import main
from assetforge.frames import alpha_bbox, remove_corner_background
from assetforge.local_animation import (
    _character_palette,
    parse_frame_counts,
    run_animation_spec,
    run_local_animation,
)
from assetforge.profile import Profile, load_profile
from assetforge.rig_build import (
    autorig_reference,
    build_rig,
    extract_part_sheet,
    load_named_parts,
    load_sheet_parts,
)
from assetforge.rig_core import RigError, frame_times, load_rig, world_transforms


def _digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _part(path: Path, size: tuple[int, int], color: tuple[int, int, int, int], kind: str = "limb") -> None:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    if kind == "head":
        draw.ellipse((2, 2, size[0] - 3, size[1] - 3), fill=color, outline=(28, 34, 46, 255), width=2)
        draw.ellipse(
            (round(size[0] * 0.65), round(size[1] * 0.38), round(size[0] * 0.72), round(size[1] * 0.48)),
            fill=(20, 22, 26, 255),
        )
    elif kind == "body":
        draw.ellipse((2, 2, size[0] - 3, size[1] - 3), fill=color, outline=(28, 34, 46, 255), width=2)
    elif kind == "tail":
        draw.polygon(
            [(size[0] - 3, size[1] // 2), (4, 3), (13, size[1] // 2), (4, size[1] - 4)],
            fill=color,
            outline=(28, 34, 46, 255),
        )
    elif kind == "wing":
        draw.polygon(
            [(3, size[1] - 4), (size[0] - 4, 4), (size[0] - 12, size[1] - 8)],
            fill=color,
            outline=(28, 34, 46, 255),
        )
    else:
        draw.rounded_rectangle(
            (2, 2, size[0] - 3, size[1] - 3),
            radius=max(2, min(size) // 3),
            fill=color,
            outline=(28, 34, 46, 255),
            width=2,
        )
        draw.line(
            (size[0] // 2, 3, size[0] // 2, size[1] - 4),
            fill=color,
            width=2,
        )
    image.save(path)


def _biped_parts(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _part(root / "head.png", (42, 36), (244, 190, 120, 255), "head")
    _part(root / "torso.png", (40, 70), (75, 130, 205, 255), "body")
    for name in ("uarm_f", "uarm_b"):
        _part(root / f"{name}.png", (15, 36), (102, 158, 224, 255))
    for name in ("farm_f", "farm_b"):
        _part(root / f"{name}.png", (13, 34), (238, 181, 118, 255))
    for name in ("thigh_f", "thigh_b"):
        _part(root / f"{name}.png", (18, 43), (58, 70, 112, 255))
    for name in ("shin_f", "shin_b"):
        _part(root / f"{name}.png", (17, 46), (48, 58, 92, 255))
    return root


def _quadruped_parts(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _part(root / "body.png", (88, 50), (190, 126, 220, 255), "body")
    _part(root / "head.png", (48, 42), (238, 183, 106, 255), "head")
    _part(root / "tail.png", (70, 30), (190, 126, 220, 255), "tail")
    _part(root / "wing_f.png", (66, 56), (126, 190, 230, 255), "wing")
    _part(root / "foreleg_f.png", (18, 52), (150, 102, 190, 255))
    _part(root / "hindleg_f.png", (19, 53), (150, 102, 190, 255))
    return root


class LocalAnimationTests(unittest.TestCase):
    def test_non_loop_time_includes_the_last_authored_pose(self) -> None:
        self.assertEqual(frame_times(4, False), [0.0, 1 / 3, 2 / 3, 1.0])
        self.assertEqual(frame_times(4, True), [0.0, 0.25, 0.5, 0.75])

    def test_animation_spec_owns_direction_and_rejects_cli_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = load_named_parts(_biped_parts(root / "parts"), "biped-side")
            build_rig(
                parts,
                root / "rig",
                archetype="biped-side",
                character="demo",
                direction="south",
                height=128,
                clips=["idle"],
                resample="nearest",
            )
            spec = {
                "schemaVersion": 1,
                "id": "demo-south",
                "character": "demo",
                "direction": "south",
                "rig": "rig/rig.json",
                "clips": {"idle": {"frames": 3}},
                "render": {
                    "renderer": "local-cutout-v1",
                    "fit": "shared-motion-bounds",
                    "resample": "nearest",
                },
            }
            spec_path = root / "animation.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            result = run_animation_spec(spec_path)
            self.assertEqual(result["direction"], "south")
            self.assertEqual(result["frameCounts"], {"idle": 3})
            self.assertEqual(result["raw"]["resample"], "nearest")
            self.assertEqual(result["animationSpec"]["id"], "demo-south")
            self.assertEqual(
                json.loads(Path(result["manifest"]).read_text(encoding="utf-8")),
                result,
            )

            shutil.rmtree(root / "build")
            outside = root / "outside"
            outside.mkdir()
            (root / "build").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                run_animation_spec(spec_path)
            self.assertEqual(list(outside.iterdir()), [])
            (root / "build").unlink()

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    ["animate", "--spec", str(spec_path), "--direction", "east"]
                )
            self.assertEqual(exit_code, 2)
            self.assertIn("only --work may override", stdout.getvalue())

    def test_animation_spec_clip_member_order_does_not_change_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = load_named_parts(_biped_parts(root / "parts"), "biped-side")
            build_rig(
                parts,
                root / "rig",
                archetype="biped-side",
                character="demo",
                direction="east",
                height=128,
                clips=["attack", "death"],
                resample="nearest",
            )

            def write_spec(name: str, clip_items: list[tuple[str, dict[str, int]]]) -> Path:
                spec = {
                    "schemaVersion": 1,
                    "id": name,
                    "character": "demo",
                    "direction": "east",
                    "rig": "rig/rig.json",
                    "clips": dict(clip_items),
                    "render": {
                        "renderer": "local-cutout-v1",
                        "fit": "shared-motion-bounds",
                        "resample": "nearest",
                    },
                }
                target = root / f"{name}.json"
                target.write_text(json.dumps(spec), encoding="utf-8")
                return target

            forward = run_animation_spec(
                write_spec("forward", [("attack", {"frames": 3}), ("death", {"frames": 3})])
            )
            reverse = run_animation_spec(
                write_spec("reverse", [("death", {"frames": 3}), ("attack", {"frames": 3})])
            )
            self.assertEqual(forward["raw"]["referenceClip"], "attack")
            self.assertEqual(forward["raw"]["motionAnchor"], reverse["raw"]["motionAnchor"])
            for clip in ("attack", "death"):
                self.assertEqual(
                    [frame["sha256"] for frame in forward["raw"]["clips"][clip]["frames"]],
                    [frame["sha256"] for frame in reverse["raw"]["clips"][clip]["frames"]],
                )

    def test_successful_rerun_removes_stale_raw_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = load_named_parts(_biped_parts(root / "parts"), "biped-side")
            report = build_rig(
                parts,
                root / "rig",
                archetype="biped-side",
                character="demo",
                height=128,
            )
            work = root / "build"
            run_local_animation(
                work_dir=work,
                character="demo",
                direction="east",
                clips=["idle", "walk"],
                rig_path=report["rig"],
            )
            self.assertTrue((work / "raw" / "east" / "walk").is_dir())
            latest = run_local_animation(
                work_dir=work,
                character="demo",
                direction="east",
                clips=["idle"],
                rig_path=report["rig"],
            )
            self.assertEqual(set(latest["raw"]["clips"]), {"idle"})
            self.assertFalse((work / "raw" / "east" / "walk").exists())
            self.assertFalse((work / "raw" / "east" / "walk.gif").exists())

    def test_non_finite_rig_json_returns_structured_cli_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = load_named_parts(_biped_parts(root / "parts"), "biped-side")
            report = build_rig(
                parts,
                root / "rig",
                archetype="biped-side",
                character="demo",
                height=128,
                clips=["idle"],
            )
            data = json.loads(Path(report["rig"]).read_text(encoding="utf-8"))
            data["skeleton"]["root"]["offset"][0] = float("inf")
            invalid = root / "rig" / "non-finite.json"
            invalid.write_text(json.dumps(data), encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "animate",
                        "--rig",
                        str(invalid),
                        "--character",
                        "demo",
                        "--direction",
                        "east",
                        "--clips",
                        "idle",
                        "--work",
                        str(root / "build"),
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 2)
            self.assertFalse(payload["ok"])
            self.assertIn("non-finite", payload["error"])

    def test_nearest_rig_build_preserves_source_part_palette(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _biped_parts(root / "parts")
            parts = load_named_parts(source, "biped-side")
            report = build_rig(
                parts,
                root / "rig",
                archetype="biped-side",
                character="demo",
                height=192,
                resample="nearest",
            )
            with Image.open(source / "head.png") as original:
                source_colors = set(original.convert("RGBA").getdata())
            with Image.open(Path(report["rig"]).parent / "parts" / "head.png") as scaled:
                scaled_colors = set(scaled.convert("RGBA").getdata())
            self.assertLessEqual(scaled_colors, source_colors)

    def test_frame_override_parser_rejects_invalid_values(self) -> None:
        self.assertEqual(parse_frame_counts("idle=4,walk=8"), {"idle": 4, "walk": 8})
        with self.assertRaises(RigError):
            parse_frame_counts("walk=0")
        with self.assertRaises(RigError):
            parse_frame_counts("walk")

    def test_output_identifiers_reject_path_escape_before_any_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = _biped_parts(root / "parts")
            protected = root / "target" / "idle"
            protected.mkdir(parents=True)
            marker = protected / "keep.png"
            Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(marker)

            with self.assertRaisesRegex(RigError, "direction must match"):
                run_local_animation(
                    work_dir=root / "nested" / "work",
                    character="demo",
                    direction="../../../target",
                    clips=["idle"],
                    parts_dir=parts,
                    archetype="biped-side",
                    height=128,
                )
            self.assertTrue(marker.is_file())
            with self.assertRaisesRegex(RigError, "character must match"):
                run_local_animation(
                    work_dir=root / "work",
                    character="../../escaped",
                    direction="east",
                    clips=["idle"],
                    parts_dir=parts,
                    archetype="biped-side",
                    height=128,
                )
            self.assertFalse((root / "escaped_east_idle.json").exists())

    def test_incomplete_parts_cannot_claim_production_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = load_named_parts(_biped_parts(root / "parts"), "biped-side")
            incomplete = {name: loaded[name] for name in ("head", "torso")}
            with self.assertRaisesRegex(RigError, "production rig is incomplete"):
                build_rig(
                    incomplete,
                    root / "rig",
                    archetype="biped-side",
                    character="demo",
                    height=128,
                )

    def test_compiled_rig_cannot_be_relabelled_as_another_character_or_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = load_named_parts(_biped_parts(root / "parts"), "biped-side")
            report = build_rig(
                parts,
                root / "rig",
                archetype="biped-side",
                character="hero",
                direction="east",
                height=128,
                clips=["idle"],
            )
            with self.assertRaisesRegex(RigError, "does not match requested character"):
                run_local_animation(
                    work_dir=root / "character-mismatch",
                    character="villain",
                    direction="east",
                    clips=["idle"],
                    rig_path=report["rig"],
                )
            with self.assertRaisesRegex(RigError, "cannot be relabeled"):
                run_local_animation(
                    work_dir=root / "direction-mismatch",
                    character="hero",
                    direction="south",
                    clips=["idle"],
                    rig_path=report["rig"],
                )

    def test_output_symlinks_cannot_escape_work_or_rig_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = _biped_parts(root / "parts")
            work = root / "work"
            outside = root / "outside"
            work.mkdir()
            outside.mkdir()
            marker = outside / "victim.png"
            Image.new("RGBA", (5, 5), (255, 0, 0, 255)).save(marker)
            original = _digest(marker)
            (work / "raw").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                run_local_animation(
                    work_dir=work,
                    character="hero",
                    direction="east",
                    clips=["idle"],
                    parts_dir=parts,
                    archetype="biped-side",
                    height=128,
                )
            self.assertEqual(_digest(marker), original)

            (work / "raw").unlink()
            rig_root = work / "rig"
            if rig_root.exists():
                shutil.rmtree(rig_root)
            rig_root.mkdir()
            (rig_root / "parts").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                run_local_animation(
                    work_dir=work,
                    character="hero",
                    direction="east",
                    clips=["idle"],
                    parts_dir=parts,
                    archetype="biped-side",
                    height=128,
                )
            self.assertEqual(_digest(marker), original)

    def test_tiny_opaque_image_fails_cleanly_without_index_error(self) -> None:
        cleaned = remove_corner_background(
            Image.new("RGB", (1, 1), (200, 200, 200)),
            42,
        )
        self.assertIsNone(cleaned.getchannel("A").getbbox())

    def test_production_biped_generates_five_real_clips_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = _biped_parts(root / "parts")
            first = run_local_animation(
                work_dir=root / "build-a",
                character="demo",
                direction="east",
                clips=["idle", "walk", "attack", "hit", "death"],
                parts_dir=parts,
                archetype="biped-side",
                height=192,
            )
            second = run_local_animation(
                work_dir=root / "build-b",
                character="demo",
                direction="east",
                clips=["idle", "walk", "attack", "hit", "death"],
                parts_dir=parts,
                archetype="biped-side",
                height=192,
            )
            self.assertTrue(first["ok"])
            self.assertEqual(first["quality"], "production")
            self.assertEqual(first["frameCounts"], {"idle": 6, "walk": 8, "attack": 6, "hit": 4, "death": 8})
            canvases = set()
            for clip, expected in first["frameCounts"].items():
                frames = first["raw"]["clips"][clip]["frames"]
                self.assertEqual(len(frames), expected)
                for frame in frames:
                    with Image.open(frame["file"]) as image:
                        canvases.add(image.size)
            self.assertEqual(len(canvases), 1)
            self.assertNotEqual(
                _digest(first["raw"]["clips"]["death"]["frames"][0]["file"]),
                _digest(first["raw"]["clips"]["death"]["frames"][-1]["file"]),
            )
            self.assertGreater(
                len({frame["sha256"] for frame in first["raw"]["clips"]["walk"]["frames"]}),
                2,
            )
            first_hashes = {
                clip: [frame["sha256"] for frame in data["frames"]]
                for clip, data in first["raw"]["clips"].items()
            }
            second_hashes = {
                clip: [frame["sha256"] for frame in data["frames"]]
                for clip, data in second["raw"]["clips"].items()
            }
            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual(
                json.loads(Path(first["manifest"]).read_text(encoding="utf-8")),
                first,
            )
            with Image.open(first["raw"]["clips"]["idle"]["gif"]) as idle_gif:
                self.assertEqual(idle_gif.info.get("loop"), 0)
            with Image.open(first["raw"]["clips"]["death"]["gif"]) as death_gif:
                self.assertNotIn("loop", death_gif.info)

    def test_non_loop_death_transform_clamps_at_t_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = load_named_parts(_biped_parts(root / "parts"), "biped-side")
            report = build_rig(parts, root / "rig", archetype="biped-side", character="demo", height=192)
            rig, _ = load_rig(report["rig"])
            death = rig["clips"]["death"]
            start = world_transforms(rig["skeleton"], death, 0.0)["root"]
            end = world_transforms(rig["skeleton"], death, 1.0)["root"]
            self.assertNotEqual(start, end)
            self.assertAlmostEqual(
                end[2] - start[2],
                death["tracks"]["root"]["offset_y"][-1][1],
                places=5,
            )

    def test_rig_rejects_path_escape_and_skeleton_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = load_named_parts(_biped_parts(root / "parts"), "biped-side")
            report = build_rig(parts, root / "rig", archetype="biped-side", character="demo", height=128)
            data = json.loads(Path(report["rig"]).read_text(encoding="utf-8"))
            data["parts"]["head"]["image"] = "../outside.png"
            escaped = root / "rig" / "escaped.json"
            escaped.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(RigError):
                load_rig(escaped)

            data = json.loads(Path(report["rig"]).read_text(encoding="utf-8"))
            data["skeleton"]["root"]["parent"] = "spine"
            cycled = root / "rig" / "cycled.json"
            cycled.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(RigError):
                load_rig(cycled)

            data = json.loads(Path(report["rig"]).read_text(encoding="utf-8"))
            data["source"]["mode"] = "auto-segmented-flat"
            data["source"]["quality"] = "production"
            dishonest = root / "rig" / "dishonest.json"
            dishonest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(RigError, "'coarse' was expected"):
                load_rig(dishonest)

    def test_loaded_production_rig_cannot_bypass_semantic_part_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = load_named_parts(_biped_parts(root / "parts"), "biped-side")
            report = build_rig(
                parts,
                root / "rig",
                archetype="biped-side",
                character="demo",
                height=128,
            )
            original = json.loads(Path(report["rig"]).read_text(encoding="utf-8"))

            incomplete = json.loads(json.dumps(original))
            only = incomplete["parts"]["head"]
            only["joint"] = "root"
            incomplete["parts"] = {"only": only}
            incomplete_path = root / "rig" / "incomplete-production.json"
            incomplete_path.write_text(json.dumps(incomplete), encoding="utf-8")
            with self.assertRaisesRegex(RigError, "production rig is incomplete"):
                load_rig(incomplete_path)

            wrong_binding = json.loads(json.dumps(original))
            wrong_binding["parts"]["head"]["joint"] = "root"
            binding_path = root / "rig" / "wrong-binding.json"
            binding_path.write_text(json.dumps(wrong_binding), encoding="utf-8")
            with self.assertRaisesRegex(RigError, "invalid semantic joint bindings"):
                load_rig(binding_path)

    def test_recovering_action_rejects_endpoint_only_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = load_named_parts(_biped_parts(root / "parts"), "biped-side")
            report = build_rig(
                parts,
                root / "rig",
                archetype="biped-side",
                character="demo",
                height=128,
            )
            with self.assertRaisesRegex(RigError, "requires at least 3 frames"):
                run_local_animation(
                    work_dir=root / "build",
                    character="demo",
                    direction="east",
                    clips=["attack"],
                    frame_overrides={"attack": 2},
                    rig_path=report["rig"],
                )

    def test_extreme_motion_fails_instead_of_emitting_clipped_or_empty_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = load_named_parts(_biped_parts(root / "parts"), "biped-side")
            report = build_rig(
                parts,
                root / "rig",
                archetype="biped-side",
                character="demo",
                height=128,
                clips=["hit"],
            )
            data = json.loads(Path(report["rig"]).read_text(encoding="utf-8"))
            data["clips"]["hit"]["tracks"]["root"]["offset_x"] = [
                [0.0, 0.0],
                [1.0, 10000.0],
            ]
            extreme = root / "rig" / "extreme.json"
            extreme.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(RigError, "rendered empty|temporary canvas edge"):
                run_local_animation(
                    work_dir=root / "build",
                    character="demo",
                    direction="east",
                    clips=["hit"],
                    frame_overrides={"hit": 3},
                    rig_path=extreme,
                )

    def test_profile_content_min_rejects_collapsed_custom_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = load_named_parts(_biped_parts(root / "parts"), "biped-side")
            report = build_rig(
                parts,
                root / "rig",
                archetype="biped-side",
                character="hero",
                height=192,
            )
            data = json.loads(Path(report["rig"]).read_text(encoding="utf-8"))
            for clip in data["clips"].values():
                clip.pop("grounded", None)
            custom = root / "rig" / "ungrounded.json"
            custom.write_text(json.dumps(data), encoding="utf-8")
            result = run_local_animation(
                work_dir=root / "build",
                character="hero",
                direction="east",
                clips=["idle", "walk", "attack", "hit", "death"],
                rig_path=custom,
                profile_name="godot-pixel-demo",
                tier="battle-generated",
            )
            self.assertFalse(result["ok"])
            self.assertEqual(result["stage"], "validate")
            self.assertFalse(result["exports"])
            self.assertTrue(
                any(
                    "contentMin" in error
                    for validation in result["validation"].values()
                    for error in validation["errors"]
                )
            )

    def test_part_sheet_extract_and_mapping_build_production_rig(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_parts = _quadruped_parts(root / "source-parts")
            sheet = Image.new("RGBA", (520, 260), (207, 207, 207, 255))
            placements = {
                "body": (20, 20),
                "head": (150, 20),
                "tail": (240, 20),
                "wing_f": (350, 20),
                "foreleg_f": (80, 150),
                "hindleg_f": (180, 150),
            }
            for name, position in placements.items():
                sheet.alpha_composite(Image.open(source_parts / f"{name}.png").convert("RGBA"), position)
            sheet_path = root / "sheet.png"
            sheet.convert("RGB").save(sheet_path)
            extraction = extract_part_sheet(sheet_path, root / "extract")
            self.assertEqual(extraction["count"], 6)
            mapping: dict[str, str] = {}
            for component in extraction["components"]:
                left, top, _width, _height = component["bbox"]
                name = min(placements, key=lambda slot: abs(placements[slot][0] - left) + abs(placements[slot][1] - top))
                mapping[component["id"]] = name
            mapping_path = root / "mapping.json"
            mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
            incomplete_mapping = dict(mapping)
            incomplete_mapping.pop(next(iter(incomplete_mapping)))
            incomplete_path = root / "mapping-incomplete.json"
            incomplete_path.write_text(json.dumps(incomplete_mapping), encoding="utf-8")
            with self.assertRaisesRegex(RigError, "unmapped"):
                load_sheet_parts(
                    sheet_path,
                    incomplete_path,
                    root / "extract-incomplete",
                    "winged-quadruped-side",
                )

            alias_mapping = dict(mapping)
            alias_mapping["blob_1"] = "IGNORE"
            alias_path = root / "mapping-alias.json"
            alias_path.write_text(json.dumps(alias_mapping), encoding="utf-8")
            with self.assertRaisesRegex(RigError, "aliases component"):
                load_sheet_parts(
                    sheet_path,
                    alias_path,
                    root / "extract-alias",
                    "winged-quadruped-side",
                )
            parts, _ = load_sheet_parts(
                sheet_path,
                mapping_path,
                root / "extract-2",
                "winged-quadruped-side",
            )
            report = build_rig(
                parts,
                root / "rig",
                archetype="winged-quadruped-side",
                character="pet",
                height=192,
                source_mode="part-sheet",
            )
            self.assertEqual(report["quality"], "production")
            self.assertTrue(Path(report["rig"]).is_file())
            self.assertIn("wing_b", report["mirroredParts"])

    def test_flat_reference_is_explicitly_coarse_and_emits_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = load_named_parts(_quadruped_parts(root / "parts"), "winged-quadruped-side")
            production = build_rig(
                parts,
                root / "production",
                archetype="winged-quadruped-side",
                character="pet",
                height=192,
            )
            coarse = autorig_reference(
                production["bindPose"],
                root / "coarse",
                archetype="winged-quadruped-side",
                character="pet-coarse",
                height=192,
            )
            self.assertEqual(coarse["quality"], "coarse")
            self.assertFalse(coarse["occlusionSynthesis"])
            self.assertGreater(coarse["restAlphaReconstructionIoU"], 0.99)
            self.assertEqual(coarse["semanticConfidence"], "unscored")
            self.assertTrue(Path(coarse["overlay"]).is_file())
            self.assertTrue(coarse["warnings"])

    def test_non_winged_quadruped_coarse_rig_does_not_invent_a_wing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = load_named_parts(_quadruped_parts(root / "parts"), "quadruped-side")
            parts.pop("wing_f")
            production = build_rig(
                parts,
                root / "production",
                archetype="quadruped-side",
                character="pet",
                height=192,
            )
            coarse = autorig_reference(
                production["bindPose"],
                root / "coarse",
                archetype="quadruped-side",
                character="pet-coarse",
                height=192,
            )
            self.assertNotIn("wing_f", coarse["parts"])
            rig, _ = load_rig(coarse["rig"])
            self.assertNotIn("wing_f", rig["parts"])

    def test_profile_finalize_preserves_motion_and_exports_all_requested_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = _quadruped_parts(root / "parts")
            result = run_local_animation(
                work_dir=root / "build",
                character="pet",
                direction="east",
                clips=["idle", "walk", "attack", "hit"],
                parts_dir=parts,
                archetype="winged-quadruped-side",
                height=192,
                profile_name="web-pixel-demo",
                tier="village",
            )
            self.assertTrue(result["ok"])
            self.assertEqual(set(result["exports"]), {"idle", "walk", "attack", "hit"})
            self.assertTrue(all(report["ok"] for report in result["validation"].values()))
            for normalized in result["normalized"].values():
                self.assertEqual(normalized["placementMode"], "shared-motion")
                self.assertEqual(normalized["canvas"], [40, 40])
            palettes = []
            for clip in result["normalized"]:
                paths = list(Path(result["normalized"][clip]["output"]).glob(f"{clip}_*.png"))
                for path in paths:
                    array = np.asarray(Image.open(path).convert("RGBA"))
                    palettes.append({tuple(color) for color in np.unique(array[array[:, :, 3] > 20, :3], axis=0)})
            union = set().union(*palettes)
            self.assertLessEqual(len(union), 24)

    def test_godot_five_clip_build_starts_on_anchor_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_local_animation(
                work_dir=root / "build",
                character="hero",
                direction="east",
                clips=["idle", "walk", "attack", "hit", "death"],
                parts_dir=_biped_parts(root / "parts"),
                archetype="biped-side",
                height=192,
                resample="nearest",
                profile_name="godot-pixel-demo",
                tier="battle-generated",
            )
            self.assertTrue(result["ok"], result)
            self.assertEqual(
                set(result["exports"]),
                {"idle", "walk", "attack", "hit", "death"},
            )
            profile = load_profile("godot-pixel-demo")
            self.assertEqual(result["raw"]["timingSource"], "profile:godot-pixel-demo")
            for clip in result["raw"]["clips"]:
                self.assertEqual(
                    result["raw"]["clips"][clip]["fps"],
                    float(profile.animation(clip)["fps"]),
                )
                self.assertEqual(
                    result["raw"]["clips"][clip]["loop"],
                    bool(profile.animation(clip)["loop"]),
                )
            attack_resource = Path(result["exports"]["attack"]["output"]).read_text(
                encoding="utf-8"
            )
            self.assertIn('"loop": false', attack_resource)
            self.assertIn('"speed": 12.0', attack_resource)
            for clip, manifest in result["normalized"].items():
                first = manifest["frames"][0]
                self.assertLessEqual(abs(first["foot"][1] - 243), 2, clip)
                self.assertTrue(
                    all(min(frame["contentSize"]) >= 16 for frame in manifest["frames"]),
                    clip,
                )
                self.assertEqual(
                    result["validation"][clip]["enclosedTransparencyPolicy"],
                    "report-only",
                )

    def test_profile_rejects_rig_loop_contract_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = load_named_parts(_biped_parts(root / "parts"), "biped-side")
            report = build_rig(
                parts,
                root / "rig",
                archetype="biped-side",
                character="hero",
                height=128,
                clips=["idle"],
            )
            data = json.loads(Path(report["rig"]).read_text(encoding="utf-8"))
            data["clips"]["idle"]["loop"] = False
            mismatch = root / "rig" / "loop-mismatch.json"
            mismatch.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(RigError, "loop contract mismatch"):
                run_local_animation(
                    work_dir=root / "build",
                    character="hero",
                    direction="east",
                    clips=["idle"],
                    rig_path=mismatch,
                    profile_name="godot-pixel-demo",
                    tier="battle-generated",
                )

    def test_profile_defaults_use_only_supported_local_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_local_animation(
                work_dir=root / "build",
                character="pet",
                direction="east",
                clips=None,
                parts_dir=_quadruped_parts(root / "parts"),
                archetype="winged-quadruped-side",
                height=192,
                profile_name="web-pixel-demo",
                tier="village",
            )
            self.assertTrue(result["ok"], result)
            self.assertEqual(set(result["exports"]), {"idle", "walk", "attack", "hit"})

    def test_profile_mirror_direction_can_reuse_a_bound_rig(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = load_named_parts(_biped_parts(root / "parts"), "biped-side")
            report = build_rig(
                parts,
                root / "rig",
                archetype="biped-side",
                character="hero",
                direction="east",
                height=192,
                clips=["idle"],
            )
            mirrored = run_local_animation(
                work_dir=root / "build",
                character="hero",
                direction="west",
                clips=["idle"],
                rig_path=report["rig"],
                profile_name="godot-pixel-demo",
                tier="battle-generated",
            )
            self.assertTrue(mirrored["ok"], mirrored)
            self.assertTrue(mirrored["raw"]["mirroredX"])

    def test_profile_part_alpha_gate_blocks_large_source_holes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = _biped_parts(root / "parts")
            with Image.open(parts / "torso.png") as opened:
                torso = opened.convert("RGBA")
            ImageDraw.Draw(torso).rectangle((12, 18, 28, 42), fill=(0, 0, 0, 0))
            torso.save(parts / "torso.png")
            with self.assertRaisesRegex(RigError, "source parts violate"):
                run_local_animation(
                    work_dir=root / "build",
                    character="hero",
                    direction="east",
                    clips=["idle"],
                    parts_dir=parts,
                    archetype="biped-side",
                    height=192,
                    resample="nearest",
                    profile_name="godot-pixel-demo",
                    tier="battle-generated",
                )

    def test_character_palette_respects_disabled_cross_clip_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clip = root / "idle"
            clip.mkdir()
            Image.new("RGBA", (8, 8), (200, 80, 60, 255)).save(clip / "idle_00.png")
            packaged = load_profile("web-pixel-demo")
            data = deepcopy(packaged.data)
            data["quality"]["palette"]["lockAcrossClip"] = False
            unlocked = Profile(path=packaged.path, data=data)
            rendered = {"clips": {"idle": {"directory": str(clip)}}}
            self.assertIsNone(_character_palette(rendered, unlocked))

    def test_animation_spec_resolves_relative_profile_from_spec_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = load_named_parts(_biped_parts(root / "parts"), "biped-side")
            build_rig(
                parts,
                root / "rig",
                archetype="biped-side",
                character="demo",
                direction="south",
                height=128,
                clips=["idle"],
            )
            packaged_profile = Path(__file__).parents[1] / "profiles" / "web-pixel-demo.json"
            shutil.copy2(packaged_profile, root / "profile.json")
            spec = {
                "schemaVersion": 1,
                "id": "demo-south-profile",
                "character": "demo",
                "direction": "south",
                "profile": "./profile.json",
                "tier": "village",
                "rig": "rig/rig.json",
                "clips": {"idle": {"frames": 3}},
                "render": {
                    "renderer": "local-cutout-v1",
                    "fit": "shared-motion-bounds",
                    "resample": "nearest",
                },
            }
            spec_path = root / "animation.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            elsewhere = root / "elsewhere"
            elsewhere.mkdir()
            previous = Path.cwd()
            try:
                os.chdir(elsewhere)
                result = run_animation_spec(spec_path)
            finally:
                os.chdir(previous)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["profile"], "web-pixel-demo")


if __name__ == "__main__":
    unittest.main()
