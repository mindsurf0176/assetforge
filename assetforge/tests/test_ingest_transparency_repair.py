from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from assetforge.cli import main
from assetforge.frames import ingest_frames, repair_small_enclosed_transparent_components
from assetforge.profile import Profile
from assetforge.validation import enclosed_transparent_hole_areas, validate_frames


BODY = (205, 166, 112, 255)
ACCENT = (73, 50, 58, 255)


def _profile(root: Path, repair_limit: int = 1) -> Profile:
    return Profile(
        path=root / "profile.json",
        data={
            "schemaVersion": 1,
            "id": "transparency-repair-test",
            "kind": "pixel-character",
            "projectRoot": str(root),
            "tiers": {
                "runtime": {
                    "canvasPolicy": "fixed",
                    "canvas": [16, 16],
                    "anchor": [8, 14],
                    "padding": 1,
                    "filtering": "nearest",
                    "allowUpscale": False,
                    "previewScale": 1,
                }
            },
            "animations": {"walk": {"minFrames": 4, "maxFrames": 4, "fps": 8, "loop": True}},
            "quality": {
                "background": {
                    "mode": "transparent",
                    "maxRepairableEnclosedComponentPixels": repair_limit,
                    "maxEnclosedTransparentPixels": 0,
                },
                "alphaThreshold": 20,
                "palette": {"maxColors": 8, "lockAcrossClip": True},
                "anchor": {"type": "bottom-center", "maxFootDrift": 1},
                "identity": {"maxHeightDriftRatio": 0.1, "maxWidthDriftRatio": 0.1},
            },
            "export": {"engine": "web", "resourcePrefix": "./frames"},
        },
    )


def _sprite(path: Path, hole_size: int = 1, body: tuple[int, int, int, int] = BODY) -> None:
    image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((3, 3, 12, 12), fill=body)
    if hole_size == 1:
        draw.point((8, 8), fill=(0, 0, 0, 0))
        draw.point((7, 7), fill=ACCENT)
        draw.point((9, 9), fill=ACCENT)
    else:
        draw.rectangle((7, 7, 8, 8), fill=(0, 0, 0, 0))
    image.save(path)


class IngestTransparencyRepairTests(unittest.TestCase):
    def test_one_pixel_component_uses_existing_majority_boundary_color(self) -> None:
        image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((3, 3, 12, 12), fill=BODY)
        draw.point((8, 8), fill=(0, 0, 0, 0))
        draw.point((7, 7), fill=ACCENT)
        draw.point((9, 9), fill=ACCENT)

        repaired = repair_small_enclosed_transparent_components(image, 1, 20)
        rgba = np.asarray(repaired)

        self.assertEqual(tuple(int(value) for value in rgba[8, 8]), BODY)
        self.assertEqual(int(rgba[0, 0, 3]), 0)
        self.assertEqual(enclosed_transparent_hole_areas(repaired, 20), [])

    def test_component_above_limit_is_untouched(self) -> None:
        image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((3, 3, 12, 12), fill=BODY)
        draw.rectangle((7, 7, 8, 8), fill=(0, 0, 0, 0))

        repaired = repair_small_enclosed_transparent_components(image, 1, 20)

        self.assertEqual(enclosed_transparent_hole_areas(repaired, 20), [4])

    def test_existing_build_command_handles_a_repaired_multiframe_clip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            work = root / "work"
            raw.mkdir()
            for index in range(4):
                _sprite(raw / f"walk_{index:02d}.png", body=(205 - index, 166, 112, 255))

            profile = _profile(root)
            profile.path.write_text(json.dumps(profile.data), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = main(
                    [
                        "build",
                        "--profile",
                        str(profile.path),
                        "--input",
                        str(raw),
                        "--work",
                        str(work),
                        "--output",
                        str(root / "registry.json"),
                        "--character",
                        "companion",
                        "--tier",
                        "runtime",
                        "--animation",
                        "walk",
                        "--direction",
                        "south",
                        "--deploy-dir",
                        str(root / "frames"),
                    ]
                )

            self.assertEqual(status, 0, stdout.getvalue())
            self.assertEqual(len(list(work.glob("walk_*.png"))), 4)
            for frame in sorted(work.glob("walk_*.png")):
                self.assertEqual(enclosed_transparent_hole_areas(Image.open(frame), 20), [])
            registry = json.loads((root / "registry.json").read_text(encoding="utf-8"))
            exported = registry["characters"]["companion"]["tiers"]["runtime"]["animations"]["walk"]
            self.assertEqual(len(exported["directions"]["south"]), 4)

    def test_larger_ingested_hole_remains_for_the_validation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            work = root / "work"
            raw.mkdir()
            _sprite(raw / "walk_00.png", hole_size=4)
            for index in range(1, 4):
                _sprite(raw / f"walk_{index:02d}.png")

            profile = _profile(root)
            ingest_frames(profile, raw, work, "runtime", "walk", "south")
            report = validate_frames(profile, work, "runtime", "walk")

            self.assertFalse(report["ok"])
            self.assertEqual(report["transparentHoles"][0]["largestComponentPixels"], 4)
            self.assertIn("4 enclosed transparent pixels", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
