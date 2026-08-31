from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from assetforge.local_animation import run_local_animation
from assetforge.rig_core import RigError


CANVAS = (16, 16)


def _alpha_bbox(path: Path) -> tuple[int, int, int, int] | None:
    with Image.open(path) as image:
        return image.getchannel("A").getbbox()


def _write_profile(root: Path, canvas: tuple[int, int] = CANVAS) -> Path:
    profile = {
        "schemaVersion": 1,
        "id": "preserved-rig-test",
        "kind": "pixel-character",
        "projectRoot": str(root),
        "directions": ["east"],
        "mirrorDirections": {},
        "tiers": {
            "runtime": {
                "canvasPolicy": "fixed",
                "canvas": list(canvas),
                "anchor": [5, 9],
                "preservePlacement": True,
                "padding": 0,
                "filtering": "nearest",
                "downscaleFiltering": "nearest",
                "allowUpscale": False,
                "previewScale": 1,
            }
        },
        "animations": {
            "idle": {
                "minFrames": 1,
                "maxFrames": 4,
                "fps": 4,
                "loop": True,
            }
        },
        "quality": {
            "background": {
                "mode": "transparent",
                "maxRepairableEnclosedComponentPixels": 0,
                "maxEnclosedTransparentPixels": 0,
            },
            "alphaThreshold": 20,
            "palette": {"maxColors": 8, "lockAcrossClip": True},
            "anchor": {"type": "bottom-center", "maxFootDrift": 0},
            "identity": {
                "maxHeightDriftRatio": 0,
                "maxWidthDriftRatio": 0,
            },
        },
        "export": {
            "engine": "web",
            "format": "registry-json",
            "resourcePrefix": "./generated",
        },
    }
    path = root / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


def _write_rig(root: Path, offset_track: list[list[float]]) -> Path:
    rig_root = root / "rig"
    rig_root.mkdir()
    Image.new("RGBA", (4, 6), (92, 164, 214, 255)).save(rig_root / "part.png")
    rig = {
        "schemaVersion": 1,
        "id": "preserved-hero",
        "direction": "east",
        "archetype": "biped-side",
        "canvas": list(CANVAS),
        "source": {
            "mode": "auto-segmented-flat",
            "quality": "coarse",
            "occlusionSynthesis": False,
        },
        "parts": {
            "sprite": {
                "image": "part.png",
                "pivot": [0, 0],
                "joint": "root",
                "z": 0,
            }
        },
        "skeleton": {"root": {"parent": None, "offset": [3, 4]}},
        "clips": {
            "idle": {
                "fps": 4,
                "loop": True,
                "tracks": {"root": {"offset_x": offset_track}},
            }
        },
    }
    path = rig_root / "rig.json"
    path.write_text(json.dumps(rig), encoding="utf-8")
    return path


class PreserveRigCanvasTests(unittest.TestCase):
    def test_profile_preserves_hand_authored_rig_canvas_and_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _write_profile(root)
            rig = _write_rig(
                root,
                [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]],
            )

            result = run_local_animation(
                work_dir=root / "build",
                character="preserved-hero",
                direction="east",
                clips=["idle"],
                frame_overrides={"idle": 2},
                rig_path=rig,
                resample="nearest",
                profile_name=str(profile),
                tier="runtime",
            )

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["raw"]["fit"], "rig-canvas")
            self.assertEqual(result["raw"]["canvas"], [16, 16])
            self.assertTrue(result["raw"]["preserveRigCanvas"])
            self.assertEqual(result["raw"]["overflowPolicy"], "error")
            normalized = result["normalized"]["idle"]
            self.assertTrue(normalized["preservePlacement"])
            self.assertEqual(normalized["canvas"], [16, 16])
            self.assertIsNone(normalized["sourceAnchor"])
            self.assertIsNone(normalized["sourceBounds"])

            raw_paths = [
                Path(frame["file"])
                for frame in result["raw"]["clips"]["idle"]["frames"]
            ]
            normalized_paths = sorted(
                Path(normalized["output"]).glob("idle_*.png")
            )
            self.assertEqual(
                [_alpha_bbox(path) for path in raw_paths],
                [(3, 4, 7, 10), (4, 4, 8, 10)],
            )
            self.assertEqual(
                [_alpha_bbox(path) for path in normalized_paths],
                [(3, 4, 7, 10), (4, 4, 8, 10)],
            )

    def test_preserved_profile_rejects_a_different_rig_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _write_profile(root, (17, 16))
            rig = _write_rig(root, [[0.0, 0.0]])

            with self.assertRaisesRegex(
                RigError,
                r"preservePlacement requires RigSpec canvas \[16, 16\] "
                r"to exactly match profile canvas \[17, 16\]",
            ):
                run_local_animation(
                    work_dir=root / "build",
                    character="preserved-hero",
                    direction="east",
                    clips=["idle"],
                    frame_overrides={"idle": 1},
                    rig_path=rig,
                    resample="nearest",
                    profile_name=str(profile),
                    tier="runtime",
                )

    def test_preserved_canvas_reports_motion_overflow_before_writing_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _write_profile(root)
            rig = _write_rig(
                root,
                [[0.0, 0.0], [0.5, 20.0], [1.0, 0.0]],
            )

            with self.assertRaisesRegex(
                RigError,
                r"clip 'idle' frame 1 overflows preserved RigSpec canvas "
                r"\[16, 16\]: right=11px",
            ):
                run_local_animation(
                    work_dir=root / "build",
                    character="preserved-hero",
                    direction="east",
                    clips=["idle"],
                    frame_overrides={"idle": 2},
                    rig_path=rig,
                    resample="nearest",
                    profile_name=str(profile),
                    tier="runtime",
                )
            self.assertFalse((root / "build" / "raw" / "east").exists())


if __name__ == "__main__":
    unittest.main()
