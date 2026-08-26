from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from assetforge.profile import Profile
from assetforge.release import package_release, verify_release


def _profile(root: Path) -> Profile:
    data = {
            "schemaVersion": 1,
            "id": "release-test",
            "kind": "pixel-character",
            "projectRoot": ".",
            "directions": ["east"],
            "tiers": {
                "runtime": {
                    "canvasPolicy": "fixed",
                    "canvas": [20, 20],
                    "anchor": [10, 15],
                    "filtering": "nearest",
                }
            },
            "animations": {
                "idle": {"minFrames": 2, "maxFrames": 2, "fps": 8, "loop": True},
                "walk": {"minFrames": 2, "maxFrames": 2, "fps": 12, "loop": True},
            },
            "quality": {
                "background": {"mode": "transparent", "maxEnclosedTransparentPixels": 0},
                "alphaThreshold": 20,
                "palette": {"maxColors": 8},
                "anchor": {"maxFootDrift": 1},
                "identity": {"maxHeightDriftRatio": 0.2, "maxWidthDriftRatio": 0.2},
            },
            "export": {"engine": "web"},
    }
    profile_path = root / "profile.json"
    profile_path.write_text(json.dumps(data), encoding="utf-8")
    return Profile(path=profile_path, data=data)


def _frame(path: Path, offset: int = 0) -> None:
    image = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((6 + offset, 8, 13 + offset, 15), fill=(180, 140, 110, 255))
    image.save(path)


class ReleaseTests(unittest.TestCase):
    def test_package_and_verify_produce_portable_hashed_release(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile = _profile(root)
            source = root / "source"
            for clip in ("idle", "walk"):
                clip_dir = source / clip / "frames"
                clip_dir.mkdir(parents=True)
                _frame(clip_dir / f"{clip}_0.png")
                _frame(clip_dir / f"{clip}_1.png", 0)

            output = root / "release"
            result = package_release(
                profile,
                source,
                output,
                character="demo",
                direction="east",
                tier="runtime",
            )
            self.assertTrue(result["ok"])
            manifest_path = output / "release.json"
            manifest_text = manifest_path.read_text(encoding="utf-8")
            self.assertNotIn(str(root), manifest_text)
            self.assertEqual(result["fileCount"], 4)
            verified = verify_release(manifest_path)
            self.assertTrue(verified["ok"], verified)
            self.assertEqual(verified["verifiedFiles"], 4)
            self.assertEqual(json.loads(manifest_text)["clips"]["walk"]["fps"], 12.0)

    def test_failed_release_does_not_create_or_replace_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile = _profile(root)
            source = root / "source" / "idle" / "frames"
            source.mkdir(parents=True)
            _frame(source / "idle_0.png")
            output = root / "release"
            with self.assertRaises(ValueError):
                package_release(
                    profile,
                    root / "source",
                    output,
                    character="demo",
                    direction="east",
                    tier="runtime",
                )
            self.assertFalse(output.exists())

    def test_release_rejects_rgb_frames_without_transparency(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile = _profile(root)
            source = root / "source" / "idle" / "frames"
            source.mkdir(parents=True)
            image = Image.new("RGB", (20, 20), (0, 0, 0))
            image.save(source / "idle_0.png")
            image.save(source / "idle_1.png")
            with self.assertRaisesRegex(ValueError, "alpha channel"):
                package_release(
                    profile,
                    root / "source",
                    root / "release",
                    character="demo",
                    direction="east",
                    tier="runtime",
                )

    def test_verify_detects_tampered_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile = _profile(root)
            source = root / "source" / "idle" / "frames"
            source.mkdir(parents=True)
            _frame(source / "idle_0.png")
            _frame(source / "idle_1.png")
            output = root / "release"
            package_release(
                profile,
                root / "source",
                output,
                character="demo",
                direction="east",
                tier="runtime",
                clips=["idle"],
            )
            target = output / "frames" / "idle" / "idle_00.png"
            target.write_bytes(target.read_bytes() + b"tampered")
            verified = verify_release(output / "release.json")
            self.assertFalse(verified["ok"])
            self.assertTrue(any("hash mismatch" in error for error in verified["errors"]))


if __name__ == "__main__":
    unittest.main()
