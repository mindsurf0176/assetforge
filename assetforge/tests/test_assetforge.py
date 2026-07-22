from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from jsonschema import Draft202012Validator

from assetforge.exporters import export_assets
from assetforge.frames import alpha_bbox, ingest_frames, remove_corner_background
from assetforge.profile import (
    ProfileError,
    SCHEMA_PATH,
    list_profiles,
    load_profile,
    validate_profile_data,
)
from assetforge.providers import (
    compile_comfy_request,
    doctor,
    generation_plan,
    is_runtime_pixel_reference,
    load_comfy_workflow,
    prepare_comfy_reference,
    preserve_reference_frame,
)
from assetforge.validation import validate_frames


def synthetic_frame(path: Path, width: int, height: int, offset: int, extra_colors: bool = True) -> None:
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    x0 = 32 - width // 2 + offset
    y0 = 56 - height
    draw.rounded_rectangle((x0, y0, x0 + width - 1, 55), radius=3, fill=(240, 178, 93, 255))
    draw.rectangle((x0 + 3, y0 + 3, x0 + width - 4, y0 + 7), fill=(108, 71, 74, 255))
    draw.point((x0 + width // 3, y0 + height // 3), fill=(253, 247, 218, 255))
    if extra_colors:
        for i in range(min(width, 18)):
            draw.point((x0 + i, y0 + height // 2), fill=((i * 31) % 255, (i * 53) % 255, (i * 79) % 255, 255))
    image.save(path)


def synthetic_godot_frame(
    path: Path,
    width: int,
    height: int,
    offset: int,
) -> None:
    """Create a fixture that obeys the fixed source-canvas placement contract."""
    image = Image.new("RGBA", (244, 247), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    x0 = 123 - width // 2 + offset
    y0 = 244 - height
    draw.rounded_rectangle(
        (x0, y0, x0 + width - 1, 243),
        radius=3,
        fill=(240, 178, 93, 255),
    )
    draw.rectangle(
        (x0 + 3, y0 + 3, x0 + width - 4, y0 + 7),
        fill=(108, 71, 74, 255),
    )
    image.save(path)


class ProfileTests(unittest.TestCase):
    def test_profile_schema_is_valid_and_rejects_unknown_contract_fields(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        profile_data = json.loads(
            load_profile("web-pixel-demo").path.read_text(encoding="utf-8")
        )
        profile_data["unexpectedContract"] = True
        with self.assertRaises(ProfileError):
            validate_profile_data(profile_data, "test-profile")

    def test_packaged_profiles_are_discoverable_and_portable(self):
        ids = {entry["id"] for entry in list_profiles()}
        self.assertEqual(ids, {"godot-pixel-demo", "web-pixel-demo"})
        for profile_id in ids:
            result = doctor(load_profile(profile_id))
            self.assertTrue(result["projectRootExists"])
            self.assertTrue(all(tool["exists"] for tool in result["toolchain"]))

    def test_generation_plan_is_provider_independent_and_contract_bound(self):
        plan = generation_plan(
            load_profile("godot-pixel-demo"),
            "guardian",
            "battle-generated",
            "attack",
            "east",
            reference="/tmp/master.png",
        )
        self.assertEqual(plan["contract"]["minFrames"], 3)
        self.assertEqual(plan["contract"]["canvasPolicy"], "fixed")
        self.assertIn("human contact-sheet approval", plan["gates"][-1])

    def test_comfy_workflow_tokens_compile_from_the_same_project_plan(self):
        plan = generation_plan(load_profile("web-pixel-demo"), "companion", "village", "idle", "south")
        workflow = {
            "1": {"class_type": "ExampleText", "inputs": {"text": "${PROMPT}", "seed": "${SEED}"}},
            "2": {
                "class_type": "ExampleImage",
                "inputs": {"width": "${WIDTH}", "height": "${HEIGHT}", "image": "${REFERENCE_IMAGE}"},
            },
        }
        request = compile_comfy_request(plan, workflow, "companion-master.png")
        self.assertEqual(request["prompt"]["2"]["inputs"]["width"], 1024)
        self.assertEqual(request["prompt"]["2"]["inputs"]["image"], "companion-master.png")
        self.assertIsInstance(request["prompt"]["1"]["inputs"]["seed"], int)

    def test_godot_demo_comfy_workflow_is_api_format_and_identity_bound(self):
        profile = load_profile("godot-pixel-demo")
        plan = generation_plan(
            profile,
            "guardian",
            "battle-generated",
            "idle",
            "east",
            reference="/tmp/guardian.png",
        )
        path, workflow = load_comfy_workflow(profile)
        request = compile_comfy_request(plan, workflow, "assetforge/reference.png")
        encoded = json.dumps(request, ensure_ascii=False)
        self.assertEqual(path.name, "sdxl-pixel-art-reference-api.json")
        self.assertNotIn("${", encoded)
        self.assertIn("translucent crystal shield", plan["contract"]["prompt"])
        self.assertEqual(request["prompt"]["3"]["inputs"]["denoise"], 0.1)
        self.assertEqual(request["prompt"]["13"]["inputs"]["width"], 128)
        self.assertEqual(request["prompt"]["13"]["inputs"]["height"], 128)
        self.assertEqual(request["prompt"]["5"]["inputs"]["lora_name"], "pixel-art-xl.safetensors")

    def test_godot_generated_tier_accepts_comfy_pixel_canvas_output(self):
        profile = load_profile("godot-pixel-demo")
        plan = generation_plan(
            profile,
            "guardian",
            "battle-generated",
            "idle",
            "east",
        )
        _, workflow = load_comfy_workflow(profile)
        request = compile_comfy_request(plan, workflow, "assetforge/reference.png")
        generated_size = (
            request["prompt"]["13"]["inputs"]["width"],
            request["prompt"]["13"]["inputs"]["height"],
        )
        self.assertEqual(generated_size, (128, 128))

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            image = Image.new("RGB", generated_size, (236, 244, 241))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((42, 28, 86, 116), radius=8, fill=(108, 71, 74))
            image.save(raw / "idle_00.png")

            manifest = ingest_frames(
                profile,
                raw,
                root / "normalized",
                "battle-generated",
                "idle",
                "east",
            )
            self.assertFalse(manifest["preservePlacement"])
            self.assertEqual(manifest["canvas"], [244, 247])

    def test_comfy_reference_is_padded_to_the_generation_canvas_without_overwriting_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.png"
            Image.new("RGBA", (20, 24), (10, 20, 30, 128)).save(source)
            plan = generation_plan(
                load_profile("godot-pixel-demo"),
                "guardian",
                "battle-generated",
                "idle",
                "east",
                reference=str(source),
            )
            prepared = prepare_comfy_reference(plan, root / "raw")
            self.assertRegex(prepared.name, r"^reference-[0-9a-f]{16}\.png$")
            with Image.open(prepared) as image:
                self.assertEqual(image.size, (1024, 1024))
                self.assertEqual(image.mode, "RGB")
            with Image.open(source) as original:
                self.assertEqual(original.mode, "RGBA")

            other_source = root / "other.png"
            Image.new("RGBA", (20, 24), (200, 30, 40, 255)).save(other_source)
            other_plan = generation_plan(
                load_profile("godot-pixel-demo"),
                "guardian",
                "battle-generated",
                "idle",
                "east",
                reference=str(other_source),
            )
            other_prepared = prepare_comfy_reference(other_plan, root / "raw")
            self.assertNotEqual(prepared.name, other_prepared.name)
            self.assertTrue(prepared.is_file())
            self.assertTrue(other_prepared.is_file())

    def test_runtime_pixel_reference_is_preserved_instead_of_sent_through_lossy_vae(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "tiny.png"
            synthetic_frame(source, 18, 22, 0, extra_colors=False)
            self.assertTrue(is_runtime_pixel_reference(source))
            plan = generation_plan(
                load_profile("web-pixel-demo"),
                "companion",
                "village",
                "idle",
                "south",
                reference=str(source),
            )
            result = preserve_reference_frame(plan, root / "raw")
            self.assertEqual(result["mode"], "reference-preserve")
            with Image.open(result["images"][0]["path"]) as preserved:
                self.assertEqual(preserved.size, (64, 64))
                self.assertEqual(preserved.mode, "RGBA")

            high_resolution = root / "master.png"
            Image.new("RGBA", (700, 700), (120, 130, 140, 255)).save(high_resolution)
            self.assertFalse(is_runtime_pixel_reference(high_resolution))


class PipelineTests(unittest.TestCase):
    def test_background_removal_clears_border_connected_background(self):
        background = (236, 244, 241)
        image = Image.new("RGB", (20, 20), background)
        draw = ImageDraw.Draw(image)
        draw.rectangle((4, 4, 15, 15), fill=(108, 71, 74))

        removed = np.asarray(remove_corner_background(image, tolerance=42))

        self.assertTrue(np.all(removed[0, :, 3] == 0))
        self.assertTrue(np.all(removed[-1, :, 3] == 0))
        self.assertEqual(int(removed[5, 5, 3]), 255)

    def test_background_removal_keeps_enclosed_same_color_pixels_opaque(self):
        background = (236, 244, 241)
        image = Image.new("RGB", (20, 20), background)
        draw = ImageDraw.Draw(image)
        draw.rectangle((4, 4, 15, 15), fill=(108, 71, 74))
        draw.rectangle((8, 8, 11, 11), fill=background)

        removed = np.asarray(remove_corner_background(image, tolerance=42))

        self.assertEqual(int(removed[0, 0, 3]), 0)
        self.assertTrue(np.all(removed[8:12, 8:12, 3] == 255))

    def test_enclosed_face_colors_are_not_cut_out_as_background(self):
        background = (236, 244, 241)
        image = Image.new("RGB", (128, 128), background)
        draw = ImageDraw.Draw(image)
        draw.ellipse((28, 22, 100, 112), fill=(108, 71, 74))
        draw.ellipse((42, 40, 86, 78), fill=(225, 235, 230))
        face_pixels = [(50, 50), (60, 60), (80, 50)]

        removed = np.asarray(remove_corner_background(image, tolerance=42))

        self.assertEqual(int(removed[0, 0, 3]), 0)
        self.assertTrue(all(removed[y, x, 3] == 255 for x, y in face_pixels))

    def test_web_demo_ingest_locks_canvas_anchor_and_palette_then_exports_web(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            work = root / "work"
            raw.mkdir()
            for index, (width, height, offset) in enumerate([(18, 22, 0), (19, 23, 1), (18, 21, -1), (20, 22, 0)]):
                synthetic_frame(raw / f"frame_{index}.png", width, height, offset)

            profile = load_profile("web-pixel-demo")
            manifest = ingest_frames(profile, raw, work, "village", "walk", "south")
            self.assertEqual(manifest["canvas"], [40, 40])
            self.assertEqual(manifest["anchor"], [20, 37])
            self.assertTrue((work / "_contact.png").is_file())

            for path in sorted(work.glob("walk_*.png")):
                image = Image.open(path).convert("RGBA")
                self.assertEqual(image.size, (40, 40))
                box = alpha_bbox(image, 20)
                self.assertIsNotNone(box)
                self.assertEqual(box[3] - 1, 37)
                rgba = np.asarray(image)
                opaque = rgba[:, :, 3] > 20
                self.assertLessEqual(len(np.unique(rgba[opaque, :3], axis=0)), 24)

            report = validate_frames(profile, work, "village", "walk")
            self.assertTrue(report["ok"], report)
            exported = export_assets(
                profile,
                work,
                root / "registry.json",
                "companion",
                "village",
                "walk",
                "south",
                "./characters/companion/walk/south",
            )
            self.assertTrue(exported["ok"])
            registry = json.loads((root / "registry.json").read_text())
            frames = registry["characters"]["companion"]["tiers"]["village"]["animations"]["walk"]["directions"]["south"]
            self.assertEqual(len(frames), 4)

    def test_godot_fixed_reference_canvas_alignment_and_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            work = root / "work"
            raw.mkdir()
            for index, (width, height, offset) in enumerate([(24, 34, 0), (27, 36, 1), (25, 35, -1), (26, 34, 0)]):
                synthetic_godot_frame(raw / f"pose_{index}.png", width, height, offset)

            profile = load_profile("godot-pixel-demo")
            manifest = ingest_frames(profile, raw, work, "battle-approved", "walk", "east")
            self.assertEqual(manifest["canvasPolicy"], "fixed")
            self.assertEqual(manifest["canvas"], [244, 247])
            foot_lines = {frame["foot"][1] for frame in manifest["frames"]}
            self.assertEqual(foot_lines, {243})

            report = validate_frames(profile, work, "battle-approved", "walk")
            self.assertTrue(report["ok"], report)
            output = root / "guardian_walk.tres"
            exported = export_assets(
                profile,
                work,
                output,
                "guardian",
                "battle-approved",
                "walk",
                "east",
                "res://assets/sprites/guardian_generated",
            )
            self.assertEqual(exported["format"], "godot-spriteframes")
            text = output.read_text()
            self.assertIn('[gd_resource type="SpriteFrames"', text)
            self.assertIn('"name": &"walk"', text)
            self.assertEqual(text.count('[ext_resource type="Texture2D"'), 4)

    def test_validation_filters_a_mixed_runtime_directory_by_animation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            work = root / "work"
            raw.mkdir()
            for index in range(4):
                synthetic_godot_frame(raw / f"walk_{index}.png", 24, 34 + index % 2, 0)
            for index in range(6):
                synthetic_godot_frame(raw / f"attack_{index}.png", 25, 35, 0)
            profile = load_profile("godot-pixel-demo")
            ingest_frames(profile, raw, work, "battle-approved", "walk", "east")
            ingest_frames(profile, raw, work, "battle-approved", "attack", "east")
            report = validate_frames(profile, work, "battle-approved", "attack")
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["frameCount"], 6)


if __name__ == "__main__":
    unittest.main()
