from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from assetforge.redraw_quality import evaluate_redraw_images, evaluate_redraw_sample


_BACKGROUND = (236, 244, 241)


def _boards(root: Path) -> tuple[Path, Path, Path]:
    input_image = Image.new("RGB", (36, 36), _BACKGROUND)
    target_image = Image.new("RGB", (36, 36), _BACKGROUND)
    input_draw = ImageDraw.Draw(input_image)
    target_draw = ImageDraw.Draw(target_image)

    # Cell 0 is the immutable identity reference.
    for draw in (input_draw, target_draw):
        draw.rectangle((3, 3, 8, 9), fill=(86, 47, 42))
        draw.rectangle((7, 5, 9, 7), fill=(226, 165, 88))

    # Cells 1 and 2 are guide silhouettes in the input and full sprites in target.
    input_draw.rectangle((15, 3, 20, 9), fill=(70, 104, 138))
    input_draw.rectangle((27, 4, 33, 9), fill=(70, 104, 138))
    target_draw.rectangle((15, 3, 20, 9), fill=(86, 47, 42))
    target_draw.rectangle((19, 5, 21, 7), fill=(226, 165, 88))
    target_draw.rectangle((27, 4, 33, 9), fill=(86, 47, 42))
    target_draw.rectangle((26, 6, 28, 8), fill=(226, 165, 88))

    input_path = root / "input.png"
    target_path = root / "target.png"
    partial_path = root / "partial.png"
    input_image.save(input_path)
    target_image.save(target_path)
    partial = target_image.copy()
    ImageDraw.Draw(partial).rectangle((24, 0, 35, 11), fill=_BACKGROUND)
    partial.save(partial_path)
    return input_path, target_path, partial_path


def _manifest(root: Path, input_path: Path, target_path: Path) -> Path:
    def pixel_digest(path: Path) -> str:
        with Image.open(path) as opened:
            return hashlib.sha256(opened.convert("RGB").tobytes()).hexdigest()

    manifest = {
        "schemaVersion": 1,
        "id": "quality-test",
        "board": {
            "nativeSize": [36, 36],
            "trainingSize": [36, 36],
            "background": list(_BACKGROUND),
        },
        "samples": [
            {
                "id": "creature__east__walk",
                "split": "validation",
                "frameCount": 2,
                "input": input_path.name,
                "target": target_path.name,
                "inputPixelSha256": pixel_digest(input_path),
                "targetPixelSha256": pixel_digest(target_path),
            }
        ],
    }
    path = root / "dataset.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class RedrawQualityTests(unittest.TestCase):
    def test_exact_target_passes_manifest_sample_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path, target_path, _ = _boards(root)
            manifest = _manifest(root, input_path, target_path)

            report = evaluate_redraw_sample(manifest, "creature__east__walk", target_path)

            self.assertTrue(report["ok"])
            self.assertEqual(report["blockers"], [])
            self.assertEqual(report["board"]["columns"], 3)
            self.assertEqual(report["board"]["rows"], 3)
            self.assertEqual(report["metrics"]["identity"]["score"], 1.0)
            self.assertEqual(report["metrics"]["expectedCells"]["completed"], 2)
            self.assertTrue(report["metrics"]["unusedCells"]["passed"])
            self.assertTrue(report["metrics"]["background"]["passed"])

    def test_guide_input_fails_pose_completion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path, target_path, _ = _boards(root)
            manifest = _manifest(root, input_path, target_path)

            report = evaluate_redraw_sample(manifest, "creature__east__walk", input_path)

            self.assertFalse(report["ok"])
            self.assertGreater(report["metrics"]["poseGuide"]["residualRatio"], 0.5)
            self.assertTrue(any("pose-guide residual" in blocker for blocker in report["blockers"]))
            self.assertTrue(any("expected cell 1 is incomplete" in blocker for blocker in report["blockers"]))

    def test_partial_generation_fails_explicit_path_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path, target_path, partial_path = _boards(root)

            report = evaluate_redraw_images(
                input_path,
                target_path,
                partial_path,
                columns=3,
                rows=3,
                expected_frames=2,
                background=_BACKGROUND,
            )

            self.assertFalse(report["ok"])
            second = report["metrics"]["expectedCells"]["cells"][1]
            self.assertFalse(second["completed"])
            self.assertEqual(second["generatedForegroundPixels"], 0)
            self.assertTrue(any("expected cell 2 is incomplete" in blocker for blocker in report["blockers"]))

    def test_manifest_sample_paths_cannot_escape_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path, target_path, _ = _boards(root)
            manifest = _manifest(root, input_path, target_path)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["samples"][0]["input"] = "../input.png"
            manifest.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "escapes"):
                evaluate_redraw_sample(manifest, "creature__east__walk", target_path)

    def test_manifest_gate_rejects_train_samples_and_tampered_holdout_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path, target_path, _ = _boards(root)
            manifest = _manifest(root, input_path, target_path)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["samples"][0]["split"] = "train"
            manifest.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "validation holdout"):
                evaluate_redraw_sample(manifest, "creature__east__walk", target_path)

            data["samples"][0]["split"] = "validation"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            Image.new("RGB", (36, 36), _BACKGROUND).save(target_path)
            with self.assertRaisesRegex(ValueError, "pixels do not match"):
                evaluate_redraw_sample(manifest, "creature__east__walk", target_path)


if __name__ == "__main__":
    unittest.main()
