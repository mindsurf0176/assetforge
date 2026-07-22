from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from assetforge.profile import Profile
from assetforge.validation import enclosed_transparent_hole_areas, validate_frames


def _profile(root: Path, max_hole_pixels: int | None) -> Profile:
    background: dict[str, object] = {"mode": "transparent"}
    if max_hole_pixels is not None:
        background["maxEnclosedTransparentPixels"] = max_hole_pixels
    return Profile(
        path=root / "profile.json",
        data={
            "schemaVersion": 1,
            "id": "alpha-hole-test",
            "kind": "pixel-character",
            "projectRoot": str(root),
            "tiers": {"runtime": {"canvasPolicy": "fixed", "canvas": [16, 16]}},
            "animations": {"idle": {"minFrames": 1, "maxFrames": 1}},
            "quality": {
                "background": background,
                "alphaThreshold": 20,
                "palette": {"maxColors": 8},
            },
            "export": {"engine": "web"},
        },
    )


def _solid_sprite_with_cutout(path: Path, open_to_exterior: bool) -> None:
    image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((3, 3, 12, 12), fill=(220, 180, 120, 255))
    draw.rectangle((7, 7, 8, 8), fill=(0, 0, 0, 0))
    if open_to_exterior:
        draw.rectangle((7, 3, 7, 6), fill=(0, 0, 0, 0))
    image.save(path)


class TransparentHoleValidationTests(unittest.TestCase):
    def test_enclosed_transparency_fails_a_zero_pixel_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = root / "idle_00.png"
            _solid_sprite_with_cutout(frame, open_to_exterior=False)

            self.assertEqual(enclosed_transparent_hole_areas(Image.open(frame), 20), [4])

            report = validate_frames(_profile(root, 0), root, "runtime", "idle")

            self.assertFalse(report["ok"])
            self.assertEqual(
                report["transparentHoles"],
                [
                    {
                        "file": "idle_00.png",
                        "componentCount": 1,
                        "pixelCount": 4,
                        "largestComponentPixels": 4,
                    }
                ],
            )
            self.assertIn(
                "idle_00.png: 4 enclosed transparent pixels across 1 component(s) exceeds "
                "background.maxEnclosedTransparentPixels=0",
                report["errors"],
            )

    def test_transparency_with_a_path_to_the_canvas_edge_is_not_a_hole(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = root / "idle_00.png"
            _solid_sprite_with_cutout(frame, open_to_exterior=True)

            self.assertEqual(enclosed_transparent_hole_areas(Image.open(frame), 20), [])

            report = validate_frames(_profile(root, 0), root, "runtime", "idle")

            self.assertTrue(report["ok"], report)
            self.assertEqual(report["transparentHoles"][0]["pixelCount"], 0)

    def test_unconfigured_profiles_report_holes_without_rejecting_them(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = root / "idle_00.png"
            _solid_sprite_with_cutout(frame, open_to_exterior=False)

            report = validate_frames(_profile(root, None), root, "runtime", "idle")

            self.assertTrue(report["ok"], report)
            self.assertEqual(report["transparentHoles"][0]["pixelCount"], 4)


if __name__ == "__main__":
    unittest.main()
