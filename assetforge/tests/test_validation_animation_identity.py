from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from assetforge.profile import Profile
from assetforge.validation import validate_frames


def _profile(root: Path) -> Profile:
    return Profile(
        path=root / "profile.json",
        data={
            "schemaVersion": 1,
            "id": "animation-identity-test",
            "kind": "pixel-character",
            "projectRoot": str(root),
            "tiers": {
                "runtime": {
                    "canvasPolicy": "fixed",
                    "canvas": [20, 20],
                    "anchor": [10, 15],
                }
            },
            "animations": {
                "idle": {"minFrames": 2, "maxFrames": 2},
                "death": {
                    "minFrames": 2,
                    "maxFrames": 2,
                    "identity": {"maxHeightDriftRatio": 0.6},
                    "anchor": {"maxFootDrift": 2},
                },
            },
            "quality": {
                "background": {"mode": "transparent"},
                "alphaThreshold": 20,
                "palette": {"maxColors": 8},
                "anchor": {"maxFootDrift": 1},
                "identity": {"maxHeightDriftRatio": 0.2, "maxWidthDriftRatio": 0.2},
            },
            "export": {"engine": "web"},
        },
    )


def _frame(path: Path, height: int, foot: int = 15) -> None:
    image = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((7, foot - height + 1, 12, foot), fill=(180, 140, 110, 255))
    image.save(path)


class AnimationIdentityOverrideTests(unittest.TestCase):
    def test_animation_override_relaxes_only_that_clips_height_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for animation in ("idle", "death"):
                _frame(root / f"{animation}_00.png", 10)
                _frame(root / f"{animation}_01.png", 6)

            profile = _profile(root)
            death = validate_frames(profile, root, "runtime", "death")
            idle = validate_frames(profile, root, "runtime", "idle")

            self.assertEqual(death["heightDriftRatio"], 0.5)
            self.assertTrue(death["ok"], death)
            self.assertFalse(idle["ok"])
            self.assertIn("content height drift 0.500 exceeds limit", idle["errors"])

    def test_animation_anchor_override_relaxes_only_that_clips_foot_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for animation in ("idle", "death"):
                _frame(root / f"{animation}_00.png", 8, foot=13)
                _frame(root / f"{animation}_01.png", 8, foot=13)

            profile = _profile(root)
            death = validate_frames(profile, root, "runtime", "death")
            idle = validate_frames(profile, root, "runtime", "idle")

            self.assertTrue(death["ok"], death)
            self.assertFalse(idle["ok"])
            self.assertIn(
                "idle_00.png: foot y=13 exceeds anchor y=15 tolerance=1",
                idle["errors"],
            )


if __name__ == "__main__":
    unittest.main()
