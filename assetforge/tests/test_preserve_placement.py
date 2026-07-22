from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from assetforge.frames import ingest_frames
from assetforge.profile import Profile


BODY = (205, 166, 112, 255)
ACCENT = (73, 50, 58, 255)


def _profile(root: Path, policy: str = "fixed") -> Profile:
    tier: dict[str, object] = {
        "canvasPolicy": policy,
        "preservePlacement": True,
        "anchor": [8, 14],
        "padding": 1,
        "previewScale": 1,
    }
    if policy == "fixed":
        tier["canvas"] = [16, 16]
    return Profile(
        path=root / "profile.json",
        data={
            "schemaVersion": 1,
            "id": "preserve-placement-test",
            "kind": "pixel-character",
            "projectRoot": str(root),
            "tiers": {"runtime": tier},
            "animations": {"walk": {"minFrames": 1, "maxFrames": 4}},
            "quality": {
                "background": {
                    "mode": "transparent",
                    "maxRepairableEnclosedComponentPixels": 1,
                    "maxEnclosedTransparentPixels": 0,
                },
                "alphaThreshold": 20,
                "palette": {"maxColors": 8, "lockAcrossClip": True},
                "anchor": {"maxFootDrift": 8},
                "identity": {"maxHeightDriftRatio": 1.0, "maxWidthDriftRatio": 1.0},
            },
            "export": {"engine": "web"},
        },
    )


def _frame(path: Path, box: tuple[int, int, int, int], hole: tuple[int, int] | None = None) -> None:
    image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, fill=BODY)
    draw.point((box[0], box[1]), fill=ACCENT)
    if hole:
        draw.point(hole, fill=(0, 0, 0, 0))
    image.save(path)


class PreservePlacementTests(unittest.TestCase):
    def test_fixed_canvas_frames_keep_original_coordinates_and_skip_unneeded_palette(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            work = root / "work"
            raw.mkdir()
            first = raw / "walk_00.png"
            second = raw / "walk_01.png"
            _frame(first, (1, 2, 7, 10))
            _frame(second, (7, 4, 14, 13), hole=(10, 8))
            first_before = np.asarray(Image.open(first).convert("RGBA")).copy()
            second_before = np.asarray(Image.open(second).convert("RGBA")).copy()

            manifest = ingest_frames(
                _profile(root), raw, work, "runtime", "walk", "south"
            )
            first_after = np.asarray(Image.open(work / "walk_00.png").convert("RGBA"))
            second_after = np.asarray(Image.open(work / "walk_01.png").convert("RGBA"))

            self.assertTrue(manifest["preservePlacement"])
            np.testing.assert_array_equal(first_after, first_before)
            unchanged = np.ones(second_before.shape[:2], dtype=bool)
            unchanged[8, 10] = False
            np.testing.assert_array_equal(second_after[unchanged], second_before[unchanged])
            self.assertEqual(tuple(int(value) for value in second_after[8, 10]), BODY)

    def test_source_canvas_mismatch_fails_instead_of_resizing_or_recentering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            image = Image.new("RGBA", (15, 16), (0, 0, 0, 0))
            ImageDraw.Draw(image).rectangle((3, 3, 10, 12), fill=BODY)
            image.save(raw / "walk_00.png")

            with self.assertRaisesRegex(
                ValueError,
                r"source canvas \[15, 16\] must equal configured canvas \[16, 16\]",
            ):
                ingest_frames(
                    _profile(root), raw, root / "work", "runtime", "walk", "south"
                )

    def test_preserve_placement_rejects_union_canvas_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            _frame(raw / "walk_00.png", (3, 3, 10, 12))

            with self.assertRaisesRegex(
                ValueError, r"preservePlacement requires canvasPolicy='fixed'"
            ):
                ingest_frames(
                    _profile(root, policy="union"),
                    raw,
                    root / "work",
                    "runtime",
                    "walk",
                    "south",
                )


if __name__ == "__main__":
    unittest.main()
