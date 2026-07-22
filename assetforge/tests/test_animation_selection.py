from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from assetforge.frames import ingest_frames
from assetforge.profile import Profile
from assetforge.validation import validate_frames


def _profile(root: Path) -> Profile:
    return Profile(
        path=root / "profile.json",
        data={
            "schemaVersion": 1,
            "id": "animation-selection-test",
            "kind": "pixel-character",
            "projectRoot": str(root / "project"),
            "tiers": {
                "runtime": {
                    "canvasPolicy": "fixed",
                    "canvas": [8, 8],
                    "anchor": [4, 7],
                    "padding": 0,
                    "filtering": "nearest",
                    "allowUpscale": False,
                }
            },
            "animations": {
                "walk": {"minFrames": 1, "maxFrames": 4},
                "attack": {"minFrames": 1, "maxFrames": 4},
            },
            "quality": {"alphaThreshold": 20},
            "export": {"engine": "web", "resourcePrefix": "./frames"},
        },
    )


def _frame(path: Path) -> None:
    image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((2, 2, 5, 7), fill=(120, 140, 160, 255))
    image.save(path)


class AnimationSelectionTests(unittest.TestCase):
    def test_ingest_rejects_other_known_clip_instead_of_falling_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            _frame(source / "attack_00.png")

            with self.assertRaisesRegex(
                ValueError,
                r"no PNG frames found for requested animation 'walk'",
            ):
                ingest_frames(
                    _profile(root),
                    source,
                    root / "work",
                    "runtime",
                    "walk",
                    "south",
                )

    def test_validate_reports_zero_requested_frames_for_other_known_clip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            _frame(source / "attack_00.png")

            result = validate_frames(_profile(root), source, "runtime", "walk")

            self.assertFalse(result["ok"])
            self.assertEqual(result["frameCount"], 0)
            self.assertTrue(
                any("no PNG frames found for requested animation 'walk'" in error for error in result["errors"]),
                result,
            )

    def test_generic_provider_names_remain_valid_ingest_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            _frame(source / "frame_0.png")
            _frame(source / "pose_1.png")

            manifest = ingest_frames(
                _profile(root),
                source,
                root / "work",
                "runtime",
                "walk",
                "south",
            )

            self.assertEqual(len(manifest["frames"]), 2)
            self.assertTrue((root / "work" / "walk_00.png").is_file())
            self.assertTrue((root / "work" / "walk_01.png").is_file())


if __name__ == "__main__":
    unittest.main()
