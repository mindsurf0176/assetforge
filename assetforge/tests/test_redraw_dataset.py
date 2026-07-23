from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw

from assetforge.redraw_dataset import build_redraw_dataset, pose_guide


def _sprite(path: Path, offset: int = 0) -> None:
    image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((2 + offset, 2, 4 + offset, 7), fill=(90, 150, 180, 255))
    image.save(path)


class RedrawDatasetTests(unittest.TestCase):
    def test_builds_identity_pose_to_target_boards_with_character_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for character in ("train-creature", "holdout-creature"):
                source = root / character
                (source / "rotations").mkdir(parents=True)
                (source / "animations" / "walk" / "east").mkdir(parents=True)
                _sprite(source / "rotations" / "east.png")
                _sprite(source / "animations" / "walk" / "east" / "frame_00.png")
                _sprite(source / "animations" / "walk" / "east" / "frame_01.png", 1)
            spec = {
                "schemaVersion": 1,
                "id": "test-redraw",
                "nativeCanvas": [8, 8],
                "validationCharacters": ["holdout-creature"],
                "board": {
                    "cellSize": 32,
                    "columns": 3,
                    "rows": 1,
                    "trainingScale": 2,
                    "background": [236, 244, 241],
                },
                "characters": [
                    {
                        "id": character,
                        "root": character,
                        "referencePattern": "rotations/{direction}.png",
                        "directions": ["east"],
                        "clips": [
                            {
                                "id": "walk",
                                "framePattern": "animations/{clip}/{direction}/frame_*.png",
                                "expectedFrames": 2,
                                "loop": True,
                            }
                        ],
                    }
                    for character in ("train-creature", "holdout-creature")
                ],
            }
            spec_path = root / "dataset-spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")

            result = build_redraw_dataset(spec_path, root / "output")

            self.assertTrue(result["ok"])
            self.assertEqual(result["sampleCount"], 2)
            self.assertEqual(result["splits"], {"train": 1, "validation": 1})
            self.assertEqual(result["trainingSize"], [192, 64])
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual({sample["split"] for sample in manifest["samples"]}, {"train", "validation"})
            self.assertEqual(result["mflux"], {"train": 1, "holdout": 1, "promptCount": 8})
            self.assertEqual(manifest["mflux"]["train"]["path"], "mflux/train")
            self.assertEqual(manifest["mflux"]["holdout"]["path"], "mflux/holdout")
            self.assertEqual(manifest["mflux"]["train"]["sampleCount"], 1)
            self.assertEqual(manifest["mflux"]["holdout"]["sampleCount"], 1)
            self.assertEqual(
                sorted(path.name for path in (Path(result["output"]) / "mflux" / "train").iterdir()),
                ["0001_in.png", "0001_in.txt", "0001_out.png"],
            )
            self.assertEqual(
                sorted(path.name for path in (Path(result["output"]) / "mflux" / "holdout").iterdir()),
                ["0001_in.png", "0001_in.txt", "0001_out.png"],
            )
            self.assertTrue(manifest["mflux"]["train"]["entries"][0]["sample"].startswith("train-creature__"))
            self.assertTrue(
                manifest["mflux"]["holdout"]["entries"][0]["sample"].startswith("holdout-creature__")
            )
            for sample in manifest["samples"]:
                input_path = Path(result["output"]) / sample["input"]
                target_path = Path(result["output"]) / sample["target"]
                with Image.open(input_path) as input_image:
                    self.assertEqual(input_image.size, (192, 64))
                with Image.open(target_path) as target_image:
                    self.assertEqual(target_image.size, (192, 64))
            self.assertNotIn(str(root), Path(result["manifest"]).read_text(encoding="utf-8"))

            sentinel = Path(result["output"]) / "last-good.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            last_good_manifest = Path(result["manifest"]).read_bytes()
            with mock.patch(
                "assetforge.redraw_dataset._save_board",
                side_effect=OSError("simulated disk full"),
            ):
                with self.assertRaisesRegex(OSError, "simulated disk full"):
                    build_redraw_dataset(spec_path, root / "output")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(Path(result["manifest"]).read_bytes(), last_good_manifest)
            self.assertFalse(
                any(root.glob(".output.assetforge-staging-*")),
                "failed staged builds must be removed",
            )

            (root / "train-creature" / "animations" / "walk" / "east" / "frame_01.png").unlink()
            with self.assertRaisesRegex(ValueError, "expected 2 frames, found 1"):
                build_redraw_dataset(spec_path, root / "output")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            self.assertTrue(Path(result["manifest"]).is_file())

    def test_mflux_export_is_sorted_deterministic_and_resets_with_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directions = [f"dir{index:02d}" for index in range(8, -1, -1)]
            source = root / "train-creature"
            for direction in directions:
                (source / "rotations").mkdir(parents=True, exist_ok=True)
                (source / "animations" / "walk" / direction).mkdir(parents=True)
                _sprite(source / "rotations" / f"{direction}.png")
                _sprite(source / "animations" / "walk" / direction / "frame_00.png")
                _sprite(source / "animations" / "walk" / direction / "frame_01.png", 1)
            holdout = root / "holdout-creature"
            (holdout / "rotations").mkdir(parents=True)
            (holdout / "animations" / "walk" / "east").mkdir(parents=True)
            _sprite(holdout / "rotations" / "east.png")
            _sprite(holdout / "animations" / "walk" / "east" / "frame_00.png")
            _sprite(holdout / "animations" / "walk" / "east" / "frame_01.png", 1)
            characters = []
            for character, character_directions in (
                ("train-creature", directions),
                ("holdout-creature", ["east"]),
            ):
                characters.append(
                    {
                        "id": character,
                        "root": character,
                        "referencePattern": "rotations/{direction}.png",
                        "directions": character_directions,
                        "clips": [
                            {
                                "id": "walk",
                                "framePattern": "animations/{clip}/{direction}/frame_*.png",
                                "expectedFrames": 2,
                                "loop": True,
                            }
                        ],
                    }
                )
            spec = {
                "schemaVersion": 1,
                "id": "deterministic-redraw",
                "nativeCanvas": [8, 8],
                "validationCharacters": ["holdout-creature"],
                "board": {
                    "cellSize": 64,
                    "columns": 3,
                    "rows": 1,
                    "trainingScale": 1,
                    "background": [236, 244, 241],
                },
                "characters": characters,
            }
            spec_path = root / "dataset-spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            output = root / "output"

            first = build_redraw_dataset(spec_path, output)
            first_manifest = json.loads(Path(first["manifest"]).read_text(encoding="utf-8"))
            first_prompts = [
                (output / entry["prompt"]).read_text(encoding="utf-8")
                for entry in first_manifest["mflux"]["train"]["entries"]
            ]
            train_entries = first_manifest["mflux"]["train"]["entries"]
            self.assertEqual([entry["sample"] for entry in train_entries], sorted(entry["sample"] for entry in train_entries))
            self.assertEqual(first_prompts[0], first_prompts[8])
            self.assertEqual([entry["promptIndex"] for entry in train_entries], [0, 1, 2, 3, 4, 5, 6, 7, 0])
            self.assertEqual(first_manifest["mflux"]["train"]["sampleCount"], 9)
            self.assertEqual(first_manifest["mflux"]["holdout"]["sampleCount"], 1)
            self.assertEqual(len(list((output / "mflux" / "train").iterdir())), 27)
            for entry in train_entries:
                sample = next(sample for sample in first_manifest["samples"] if sample["id"] == entry["sample"])
                self.assertEqual((output / entry["input"]).read_bytes(), (output / sample["input"]).read_bytes())
                self.assertEqual((output / entry["target"]).read_bytes(), (output / sample["target"]).read_bytes())

            (output / "stale.txt").write_text("remove me", encoding="utf-8")
            second = build_redraw_dataset(spec_path, output)
            second_manifest = json.loads(Path(second["manifest"]).read_text(encoding="utf-8"))
            second_prompts = [
                (output / entry["prompt"]).read_text(encoding="utf-8")
                for entry in second_manifest["mflux"]["train"]["entries"]
            ]
            self.assertFalse((output / "stale.txt").exists())
            self.assertEqual(first_manifest["mflux"], second_manifest["mflux"])
            self.assertEqual(first_prompts, second_prompts)

    def test_pose_guide_removes_character_palette(self) -> None:
        image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        ImageDraw.Draw(image).rectangle((2, 2, 5, 7), fill=(255, 0, 255, 255))

        guide = pose_guide(image)
        colors = {pixel[:3] for pixel in guide.getdata() if pixel[3]}

        self.assertNotIn((255, 0, 255), colors)
        self.assertTrue(colors.issubset({(70, 104, 138), (244, 238, 192)}))

    def test_spec_requires_nonempty_train_and_validation_splits_before_output_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = {
                "schemaVersion": 1,
                "id": "invalid-splits",
                "nativeCanvas": [8, 8],
                "board": {
                    "cellSize": 32,
                    "columns": 3,
                    "rows": 1,
                    "trainingScale": 2,
                    "background": [236, 244, 241],
                },
                "characters": [
                    {
                        "id": "only-creature",
                        "root": "missing",
                        "referencePattern": "{direction}.png",
                        "directions": ["east"],
                        "clips": [
                            {
                                "id": "walk",
                                "framePattern": "{clip}/{direction}/*.png",
                                "expectedFrames": 2,
                                "loop": True,
                            }
                        ],
                    }
                ],
            }
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validationCharacters"):
                build_redraw_dataset(spec_path, root / "output")
            self.assertFalse((root / "output").exists())

            base["validationCharacters"] = ["only-creature"]
            spec_path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train split"):
                build_redraw_dataset(spec_path, root / "output")
            self.assertFalse((root / "output").exists())

    def test_spec_rejects_board_dimensions_incompatible_with_mflux(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = {
                "schemaVersion": 1,
                "id": "invalid-board",
                "nativeCanvas": [8, 8],
                "validationCharacters": ["holdout"],
                "board": {
                    "cellSize": 10,
                    "columns": 3,
                    "rows": 1,
                    "trainingScale": 1,
                    "background": [236, 244, 241],
                },
                "characters": [
                    {
                        "id": character,
                        "root": "missing",
                        "referencePattern": "{direction}.png",
                        "directions": ["east"],
                        "clips": [
                            {
                                "id": "walk",
                                "framePattern": "{clip}/{direction}/*.png",
                                "expectedFrames": 2,
                                "loop": True,
                            }
                        ],
                    }
                    for character in ("train", "holdout")
                ],
            }
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "64..4096"):
                build_redraw_dataset(spec_path, root / "output")
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
