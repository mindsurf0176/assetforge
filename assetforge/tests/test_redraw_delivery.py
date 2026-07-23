from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw

from assetforge import redraw_delivery as redraw_delivery_module
from assetforge.redraw_dataset import pose_guide
from assetforge.redraw_delivery import (
    evaluate_redraw_holdout_batch,
    export_redraw_board_frames,
)


_BACKGROUND = (236, 244, 241)
_SPRITE = (86, 47, 42, 255)
_ACCENT = (226, 165, 88, 255)


def _sprite(offset: int = 0) -> Image.Image:
    image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((2 + offset, 1, 4 + offset, 7), fill=_SPRITE)
    draw.rectangle((4 + offset, 3, 5 + offset, 4), fill=_ACCENT)
    return image


def _place(board: Image.Image, image: Image.Image, index: int) -> None:
    cell_size = 10
    x = index * cell_size + (cell_size - image.width) // 2
    y = (cell_size - image.height) // 2
    board.alpha_composite(image, (x, y))


def _pixel_digest(path: Path) -> str:
    with Image.open(path) as opened:
        return hashlib.sha256(opened.convert("RGB").tobytes()).hexdigest()


def _fixture(root: Path, sample_ids: list[str] | None = None) -> dict[str, object]:
    sample_ids = sample_ids or ["creature__east__walk"]
    input_board = Image.new("RGBA", (30, 10), _BACKGROUND + (255,))
    target_board = Image.new("RGBA", (30, 10), _BACKGROUND + (255,))
    reference = _sprite()
    frames = [_sprite(), _sprite(1)]
    _place(input_board, reference, 0)
    _place(target_board, reference, 0)
    for index, frame in enumerate(frames, start=1):
        _place(input_board, pose_guide(frame), index)
        _place(target_board, frame, index)

    input_path = root / "input.png"
    target_path = root / "target.png"
    input_board.convert("RGB").resize((60, 20), Image.Resampling.NEAREST).save(input_path)
    target_board.convert("RGB").resize((60, 20), Image.Resampling.NEAREST).save(target_path)
    samples = [
        {
            "id": sample_id,
            "split": "validation",
            "character": sample_id.split("__", 1)[0],
            "direction": "east",
            "clip": "walk",
            "loop": True,
            "frameCount": 2,
            "input": input_path.name,
            "target": target_path.name,
            "inputPixelSha256": _pixel_digest(input_path),
            "targetPixelSha256": _pixel_digest(target_path),
        }
        for sample_id in sample_ids
    ]
    manifest_data = {
        "schemaVersion": 1,
        "id": "redraw-delivery-test",
        "nativeCanvas": [8, 8],
        "board": {
            "nativeSize": [30, 10],
            "trainingSize": [60, 20],
            "columns": 3,
            "rows": 1,
            "cellSize": 20,
            "background": list(_BACKGROUND),
        },
        "samples": samples,
    }
    manifest = root / "dataset.json"
    manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
    return {
        "manifest": manifest,
        "manifestData": manifest_data,
        "input": input_path,
        "target": target_path,
        "frames": frames,
    }


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class RedrawDeliveryTests(unittest.TestCase):
    def test_holdout_batch_reports_pass_missing_failed_and_unreadable_boards(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ids = [
                "failed__east__walk",
                "good__east__walk",
                "missing__east__walk",
                "unreadable__east__walk",
            ]
            fixture = _fixture(root, ids)
            generated = root / "generated"
            generated.mkdir()
            shutil.copyfile(fixture["target"], generated / "good__east__walk.png")
            shutil.copyfile(fixture["input"], generated / "failed__east__walk.png")
            (generated / "unreadable__east__walk.png").write_text("not an image", encoding="utf-8")

            report = evaluate_redraw_holdout_batch(fixture["manifest"], generated)

            self.assertFalse(report["ok"])
            self.assertEqual(report["summary"], {"passed": 1, "required": 4, "failed": 3})
            reports = {item["sampleId"]: item for item in report["reports"]}
            self.assertTrue(reports["good__east__walk"]["ok"])
            self.assertFalse(reports["failed__east__walk"]["ok"])
            self.assertTrue(
                any("pose-guide residual" in blocker for blocker in reports["failed__east__walk"]["blockers"])
            )
            self.assertEqual(
                reports["missing__east__walk"]["blockers"],
                ["generated holdout board is missing"],
            )
            self.assertIn(
                "not a readable image",
                reports["unreadable__east__walk"]["blockers"][0],
            )

    def test_export_splits_frame_cells_with_nearest_resize_and_transparency(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = _fixture(root)
            output = root / "frames"

            result = export_redraw_board_frames(
                fixture["manifest"],
                "creature__east__walk",
                fixture["target"],
                output,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["frameCount"], 2)
            self.assertEqual(result["canvas"], [8, 8])
            expected_bboxes = [list(frame.getbbox() or ()) for frame in fixture["frames"]]
            for index, frame_path in enumerate(map(Path, result["frames"])):
                with Image.open(frame_path) as opened:
                    frame = opened.convert("RGBA")
                    self.assertEqual(frame.size, (8, 8))
                    self.assertEqual(frame.getpixel((0, 0))[3], 0)
                    self.assertEqual(list(frame.getbbox() or ()), expected_bboxes[index])
                    opaque_colors = {pixel[:3] for pixel in frame.getdata() if pixel[3]}
                    self.assertEqual(opaque_colors, {_SPRITE[:3], _ACCENT[:3]})

    def test_frames_manifest_is_portable_after_export_tree_moves(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = _fixture(root)
            output = root / "frames"
            export_redraw_board_frames(
                fixture["manifest"],
                "creature__east__walk",
                fixture["target"],
                output,
            )
            descriptor_text = (output / "frames.json").read_text(encoding="utf-8")
            descriptor = json.loads(descriptor_text)

            self.assertNotIn(str(root), descriptor_text)
            self.assertEqual(descriptor["sourceDataset"], "dataset.json")
            self.assertEqual(descriptor["sourceBoard"], "target.png")
            self.assertEqual([item["path"] for item in descriptor["frames"]], ["frame_00.png", "frame_01.png"])
            moved = root / "elsewhere" / "portable-frames"
            moved.parent.mkdir()
            shutil.copytree(output, moved)
            moved_descriptor = json.loads((moved / "frames.json").read_text(encoding="utf-8"))
            for item in moved_descriptor["frames"]:
                path = moved / item["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])

    def test_invalid_frame_count_and_board_geometry_preserve_previous_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = _fixture(root)
            output = root / "frames"
            export_redraw_board_frames(
                fixture["manifest"],
                "creature__east__walk",
                fixture["target"],
                output,
            )
            (output / "sentinel.txt").write_text("last good", encoding="utf-8")
            baseline = _tree_bytes(output)
            original = fixture["manifestData"]
            mutations = (
                ("frameCount", lambda data: data["samples"][0].__setitem__("frameCount", 0), "positive integer"),
                ("capacity", lambda data: data["samples"][0].__setitem__("frameCount", 3), "exceeds"),
                ("native grid", lambda data: data["board"].__setitem__("nativeSize", [31, 10]), "not divisible"),
                (
                    "training scale",
                    lambda data: data["board"].__setitem__("trainingSize", [60, 21]),
                    "nearest-neighbor scale",
                ),
                ("canvas fit", lambda data: data.__setitem__("nativeCanvas", [11, 8]), "does not fit"),
            )
            for label, mutate, expected in mutations:
                with self.subTest(label=label):
                    data = json.loads(json.dumps(original))
                    mutate(data)
                    fixture["manifest"].write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, expected):
                        export_redraw_board_frames(
                            fixture["manifest"],
                            "creature__east__walk",
                            fixture["target"],
                            output,
                        )
                    self.assertEqual(_tree_bytes(output), baseline)

    def test_missing_or_quality_failed_board_does_not_touch_previous_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = _fixture(root)
            output = root / "frames"
            export_redraw_board_frames(
                fixture["manifest"],
                "creature__east__walk",
                fixture["target"],
                output,
            )
            baseline = _tree_bytes(output)

            with self.assertRaises(FileNotFoundError):
                export_redraw_board_frames(
                    fixture["manifest"],
                    "creature__east__walk",
                    root / "missing.png",
                    output,
                )
            self.assertEqual(_tree_bytes(output), baseline)
            with self.assertRaisesRegex(ValueError, "failed quality gate"):
                export_redraw_board_frames(
                    fixture["manifest"],
                    "creature__east__walk",
                    fixture["input"],
                    output,
                )
            self.assertEqual(_tree_bytes(output), baseline)

    def test_staging_failure_preserves_last_good_output_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = _fixture(root)
            output = root / "frames"
            export_redraw_board_frames(
                fixture["manifest"],
                "creature__east__walk",
                fixture["target"],
                output,
            )
            (output / "sentinel.txt").write_text("approved", encoding="utf-8")
            baseline = _tree_bytes(output)
            real_save = Image.Image.save

            def fail_second_frame(image: Image.Image, path: object, *args: object, **kwargs: object) -> None:
                if Path(path).name == "frame_01.png":
                    raise OSError("simulated staging write failure")
                real_save(image, path, *args, **kwargs)

            with mock.patch.object(Image.Image, "save", new=fail_second_frame):
                with self.assertRaisesRegex(OSError, "simulated staging write failure"):
                    export_redraw_board_frames(
                        fixture["manifest"],
                        "creature__east__walk",
                        fixture["target"],
                        output,
                    )

            self.assertEqual(_tree_bytes(output), baseline)
            self.assertFalse(
                any(path.name.startswith(".frames.stage-") for path in output.parent.iterdir())
            )

    def test_publish_swap_failure_rolls_back_last_good_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = _fixture(root)
            output = root / "frames"
            export_redraw_board_frames(
                fixture["manifest"],
                "creature__east__walk",
                fixture["target"],
                output,
            )
            (output / "sentinel.txt").write_text("approved", encoding="utf-8")
            baseline = _tree_bytes(output)
            real_replace = redraw_delivery_module.os.replace

            def fail_stage_swap(source: object, destination: object) -> None:
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    source_path.name.startswith(".frames.stage-")
                    and destination_path.resolve() == output.resolve()
                ):
                    raise OSError("simulated atomic swap failure")
                real_replace(source, destination)

            with mock.patch.object(
                redraw_delivery_module.os,
                "replace",
                side_effect=fail_stage_swap,
            ):
                with self.assertRaisesRegex(OSError, "simulated atomic swap failure"):
                    export_redraw_board_frames(
                        fixture["manifest"],
                        "creature__east__walk",
                        fixture["target"],
                        output,
                    )

            self.assertEqual(_tree_bytes(output), baseline)
            self.assertFalse(
                any(
                    path.name.startswith((".frames.stage-", ".frames.backup-"))
                    for path in output.parent.iterdir()
                )
            )

    def test_nonempty_unowned_destination_is_rejected_without_deleting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = _fixture(root)
            output = root / "frames"
            output.mkdir()
            sentinel = output / "user-file.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not AssetForge-owned"):
                export_redraw_board_frames(
                    fixture["manifest"],
                    "creature__east__walk",
                    fixture["target"],
                    output,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
