from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import assetforge.exporters as exporters
from assetforge.exporters import (
    deploy_animation_direction,
    export_godot_spriteframes,
    export_web_registry,
)
from assetforge.path_safety import reset_output_directory
from assetforge.profile import Profile


def _profile(root: Path, engine: str) -> Profile:
    return Profile(
        path=root / "profile.json",
        data={
            "schemaVersion": 1,
            "id": f"release-{engine}",
            "kind": "pixel-character",
            "projectRoot": str(root),
            "tiers": {"runtime": {"canvasPolicy": "fixed", "canvas": [8, 8]}},
            "animations": {
                "walk": {
                    "minFrames": 1,
                    "maxFrames": 4,
                    "fps": 8,
                    "loop": True,
                }
            },
            "quality": {},
            "export": {"engine": engine, "resourcePrefix": "./frames"},
        },
    )


def _frame(path: Path, color: tuple[int, int, int, int]) -> bytes:
    Image.new("RGBA", (8, 8), color).save(path)
    return path.read_bytes()


class ReleaseSafetyTests(unittest.TestCase):
    def test_generated_reset_refuses_unowned_nonempty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            user_output = root / "documents"
            user_output.mkdir()
            only_copy = user_output / "only-copy.txt"
            only_copy.write_text("preserve me", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not marked as AssetForge-owned"):
                reset_output_directory(user_output, label="test output")

            self.assertEqual(only_copy.read_text(encoding="utf-8"), "preserve me")

    def test_generated_reset_cleans_only_marked_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary) / "generated"
            reset_output_directory(generated, label="test output")
            stale = generated / "stale.txt"
            stale.write_text("old", encoding="utf-8")

            reset_output_directory(generated, label="test output")

            self.assertFalse(stale.exists())
            self.assertTrue((generated / ".assetforge-output.json").is_file())

    def test_exact_godot_root_prefix_stays_res_scheme(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _frame(source / "walk_00.png", (80, 100, 140, 255))
            output = root / "artifact" / "walk.tres"

            result = export_godot_spriteframes(
                _profile(root, "godot"),
                source,
                output,
                "walk",
                "res://",
            )

            resource = output.read_text(encoding="utf-8")
            self.assertEqual(result["resourcePrefix"], "res://")
            self.assertIn('path="res://walk_00.png"', resource)
            self.assertNotIn("res:/walk", resource)

    def test_invalid_descriptor_path_cannot_partially_deploy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            source = root / "source"
            destination = project / "assets" / "runtime"
            invalid_output = root / "artifact" / "descriptor.json"
            for directory in (source, destination, invalid_output):
                directory.mkdir(parents=True)
            old_zero = _frame(destination / "walk_00.png", (20, 30, 40, 255))
            old_one = _frame(destination / "walk_01.png", (40, 50, 60, 255))
            _frame(source / "walk_00.png", (200, 100, 60, 255))

            with self.assertRaisesRegex(ValueError, "not a file path"):
                export_web_registry(
                    _profile(project, "web"),
                    source,
                    invalid_output,
                    "hero",
                    "runtime",
                    "walk",
                    "east",
                    "./assets/runtime",
                    destination,
                )

            self.assertEqual((destination / "walk_00.png").read_bytes(), old_zero)
            self.assertEqual((destination / "walk_01.png").read_bytes(), old_one)

    def test_descriptor_commit_failure_rolls_back_live_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            source = root / "source"
            artifact = root / "artifact"
            destination = project / "assets" / "runtime"
            for directory in (source, artifact, destination):
                directory.mkdir(parents=True)
            old_zero = _frame(destination / "walk_00.png", (20, 30, 40, 255))
            old_one = _frame(destination / "walk_01.png", (40, 50, 60, 255))
            _frame(source / "walk_00.png", (200, 100, 60, 255))
            descriptor = artifact / "walk.json"
            descriptor.write_text("old descriptor\n", encoding="utf-8")
            real_replace = os.replace
            failed = False

            def fail_descriptor_once(source_path: str | Path, target_path: str | Path) -> None:
                nonlocal failed
                if Path(target_path).resolve() == descriptor.resolve() and not failed:
                    failed = True
                    raise OSError("simulated descriptor commit failure")
                real_replace(source_path, target_path)

            with patch.object(exporters.os, "replace", side_effect=fail_descriptor_once):
                with self.assertRaisesRegex(OSError, "simulated descriptor commit failure"):
                    export_web_registry(
                        _profile(project, "web"),
                        source,
                        descriptor,
                        "hero",
                        "runtime",
                        "walk",
                        "east",
                        "./assets/runtime",
                        destination,
                    )

            self.assertEqual((destination / "walk_00.png").read_bytes(), old_zero)
            self.assertEqual((destination / "walk_01.png").read_bytes(), old_one)
            self.assertEqual(descriptor.read_text(encoding="utf-8"), "old descriptor\n")
            self.assertFalse(any(root.rglob(".assetforge-deploy-*")))
            self.assertFalse(any(root.rglob(".assetforge-descriptor-*")))

    def test_interrupt_after_descriptor_replace_restores_descriptor_and_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            source = root / "source"
            artifact = root / "artifact"
            destination = project / "assets" / "runtime"
            for directory in (source, artifact, destination):
                directory.mkdir(parents=True)
            old_frame = _frame(destination / "walk_00.png", (20, 30, 40, 255))
            _frame(source / "walk_00.png", (200, 100, 60, 255))
            descriptor = artifact / "walk.json"
            descriptor.write_text("old descriptor\n", encoding="utf-8")
            real_replace = os.replace
            interrupted = False

            def interrupt_after_replace(source_path: str | Path, target_path: str | Path) -> None:
                nonlocal interrupted
                real_replace(source_path, target_path)
                if Path(target_path).resolve() == descriptor.resolve() and not interrupted:
                    interrupted = True
                    raise KeyboardInterrupt()

            with patch.object(exporters.os, "replace", side_effect=interrupt_after_replace):
                with self.assertRaises(KeyboardInterrupt):
                    export_web_registry(
                        _profile(project, "web"),
                        source,
                        descriptor,
                        "hero",
                        "runtime",
                        "walk",
                        "east",
                        "./assets/runtime",
                        destination,
                    )

            self.assertEqual((destination / "walk_00.png").read_bytes(), old_frame)
            self.assertEqual(descriptor.read_text(encoding="utf-8"), "old descriptor\n")

    def test_windows_lock_failure_does_not_unlock_or_strand_thread_lock(self) -> None:
        class FakeMsvcrt:
            LK_NBLCK = 1
            LK_UNLCK = 2

            def __init__(self) -> None:
                self.acquire_attempts = 0
                self.unlock_attempts = 0

            def locking(self, _fd: int, mode: int, _length: int) -> None:
                if mode == self.LK_NBLCK:
                    self.acquire_attempts += 1
                    if self.acquire_attempts == 1:
                        raise OSError("simulated Windows lock contention")
                elif mode == self.LK_UNLCK:
                    self.unlock_attempts += 1

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "project"
            fake = FakeMsvcrt()
            with (
                patch.object(exporters, "fcntl", None),
                patch.object(exporters, "msvcrt", fake, create=True),
            ):
                with self.assertRaisesRegex(RuntimeError, "deployment already in progress"):
                    with exporters._deployment_lock(destination):
                        self.fail("contended Windows lock must not be entered")

                self.assertEqual(fake.unlock_attempts, 0)
                with exporters._deployment_lock(destination):
                    pass
                self.assertEqual(fake.acquire_attempts, 2)
                self.assertEqual(fake.unlock_attempts, 1)

    def test_direction_transaction_removes_superseded_clip_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            source = root / "source"
            source.mkdir()
            idle = source / "idle_00.png"
            walk = source / "walk_00.png"
            _frame(idle, (70, 90, 130, 255))
            _frame(walk, (100, 120, 160, 255))
            profile = _profile(project, "web")
            deploy_root = project / "assets"

            deploy_animation_direction(
                profile,
                {"idle": [idle], "walk": [walk]},
                "hero",
                "east",
                "./assets",
                deploy_root,
            )
            direction_root = deploy_root / "hero" / "east"
            self.assertTrue((direction_root / "walk" / "walk_00.png").is_file())

            result = deploy_animation_direction(
                profile,
                {"idle": [idle]},
                "hero",
                "east",
                "./assets",
                deploy_root,
            )

            self.assertEqual(set(result["clips"]), {"idle"})
            self.assertTrue((direction_root / "idle" / "idle_00.png").is_file())
            self.assertFalse((direction_root / "walk").exists())

    def test_direction_swap_failure_restores_all_previous_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            source = root / "source"
            source.mkdir()
            old_idle = source / "idle_00.png"
            old_walk = source / "walk_00.png"
            _frame(old_idle, (70, 90, 130, 255))
            _frame(old_walk, (100, 120, 160, 255))
            profile = _profile(project, "web")
            deploy_root = project / "assets"
            deploy_animation_direction(
                profile,
                {"idle": [old_idle], "walk": [old_walk]},
                "hero",
                "east",
                "./assets",
                deploy_root,
            )
            direction_root = deploy_root / "hero" / "east"
            previous_idle = (direction_root / "idle" / "idle_00.png").read_bytes()
            previous_walk = (direction_root / "walk" / "walk_00.png").read_bytes()
            _frame(old_idle, (220, 100, 60, 255))
            real_replace = os.replace
            failed = False

            def fail_new_tree_once(source_path: str | Path, target_path: str | Path) -> None:
                nonlocal failed
                source_name = Path(source_path).name
                if (
                    Path(target_path).resolve() == direction_root.resolve()
                    and ".assetforge-stage-" in source_name
                    and not failed
                ):
                    failed = True
                    raise OSError("simulated direction swap failure")
                real_replace(source_path, target_path)

            with patch.object(exporters.os, "replace", side_effect=fail_new_tree_once):
                with self.assertRaisesRegex(OSError, "simulated direction swap failure"):
                    deploy_animation_direction(
                        profile,
                        {"idle": [old_idle]},
                        "hero",
                        "east",
                        "./assets",
                        deploy_root,
                    )

            self.assertEqual(
                (direction_root / "idle" / "idle_00.png").read_bytes(),
                previous_idle,
            )
            self.assertEqual(
                (direction_root / "walk" / "walk_00.png").read_bytes(),
                previous_walk,
            )

    def test_direction_finalize_failure_restores_previous_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            source = root / "source"
            source.mkdir()
            idle = source / "idle_00.png"
            _frame(idle, (70, 90, 130, 255))
            profile = _profile(project, "web")
            deploy_root = project / "assets"
            deploy_animation_direction(
                profile,
                {"idle": [idle]},
                "hero",
                "east",
                "./assets",
                deploy_root,
            )
            deployed = deploy_root / "hero" / "east" / "idle" / "idle_00.png"
            previous = deployed.read_bytes()
            _frame(idle, (220, 100, 60, 255))

            def fail_finalize(_deployment: dict[str, object]) -> None:
                raise OSError("simulated final manifest failure")

            with self.assertRaisesRegex(OSError, "simulated final manifest failure"):
                deploy_animation_direction(
                    profile,
                    {"idle": [idle]},
                    "hero",
                    "east",
                    "./assets",
                    deploy_root,
                    fail_finalize,
                )

            self.assertEqual(deployed.read_bytes(), previous)

    def test_concurrent_direction_deploy_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            source = root / "source"
            source.mkdir()
            idle_a = source / "idle_a.png"
            idle_b = source / "idle_b.png"
            _frame(idle_a, (70, 90, 130, 255))
            _frame(idle_b, (220, 100, 60, 255))
            profile = _profile(project, "web")
            deploy_root = project / "assets"
            entered = threading.Event()
            release = threading.Event()
            errors: list[BaseException] = []

            def hold_finalize(_deployment: dict[str, object]) -> None:
                entered.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test deployment lock was not released")

            def first_deploy() -> None:
                try:
                    deploy_animation_direction(
                        profile,
                        {"idle": [idle_a]},
                        "hero",
                        "east",
                        "./assets",
                        deploy_root,
                        hold_finalize,
                    )
                except BaseException as exc:
                    errors.append(exc)

            worker = threading.Thread(target=first_deploy)
            worker.start()
            self.assertTrue(entered.wait(timeout=5))
            try:
                with self.assertRaisesRegex(RuntimeError, "deployment already in progress"):
                    deploy_animation_direction(
                        profile,
                        {"idle": [idle_b]},
                        "hero",
                        "east",
                        "./assets",
                        deploy_root,
                    )
            finally:
                release.set()
                worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            deployed = deploy_root / "hero" / "east" / "idle" / "idle_a.png"
            self.assertTrue(deployed.is_file())

    def test_direct_clip_export_cannot_race_or_enter_managed_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            source = root / "source"
            source.mkdir()
            idle = source / "idle_00.png"
            walk = source / "walk_00.png"
            _frame(idle, (70, 90, 130, 255))
            _frame(walk, (220, 100, 60, 255))
            profile = _profile(project, "web")
            deploy_root = project / "assets"
            direction_root = deploy_root / "hero" / "east"
            clip_destination = direction_root / "walk"
            descriptor = root / "artifact" / "walk.json"
            entered = threading.Event()
            release = threading.Event()
            errors: list[BaseException] = []

            def hold_finalize(_deployment: dict[str, object]) -> None:
                entered.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test deployment lock was not released")

            def direction_deploy() -> None:
                try:
                    deploy_animation_direction(
                        profile,
                        {"idle": [idle]},
                        "hero",
                        "east",
                        "./assets",
                        deploy_root,
                        hold_finalize,
                    )
                except BaseException as exc:
                    errors.append(exc)

            worker = threading.Thread(target=direction_deploy)
            worker.start()
            self.assertTrue(entered.wait(timeout=5))
            try:
                with self.assertRaisesRegex(RuntimeError, "deployment already in progress"):
                    export_web_registry(
                        profile,
                        source,
                        descriptor,
                        "hero",
                        "runtime",
                        "walk",
                        "east",
                        "./assets/hero/east/walk",
                        clip_destination,
                    )
            finally:
                release.set()
                worker.join(timeout=5)

            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            with self.assertRaisesRegex(ValueError, "managed by a direction transaction"):
                export_web_registry(
                    profile,
                    source,
                    descriptor,
                    "hero",
                    "runtime",
                    "walk",
                    "east",
                    "./assets/hero/east/walk",
                    clip_destination,
                )
            self.assertFalse(clip_destination.exists())
            self.assertFalse(descriptor.exists())

    def test_direction_rollback_failure_preserves_previous_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            source = root / "source"
            source.mkdir()
            idle = source / "idle_00.png"
            _frame(idle, (70, 90, 130, 255))
            profile = _profile(project, "web")
            deploy_root = project / "assets"
            deploy_animation_direction(
                profile,
                {"idle": [idle]},
                "hero",
                "east",
                "./assets",
                deploy_root,
            )
            _frame(idle, (220, 100, 60, 255))
            direction_root = deploy_root / "hero" / "east"
            real_replace = os.replace
            destination_failures = 0

            def fail_swap_and_restore(source_path: str | Path, target_path: str | Path) -> None:
                nonlocal destination_failures
                if Path(target_path).resolve() == direction_root.resolve():
                    destination_failures += 1
                    if destination_failures <= 2:
                        raise OSError(f"simulated destination failure {destination_failures}")
                real_replace(source_path, target_path)

            with patch.object(exporters.os, "replace", side_effect=fail_swap_and_restore):
                with self.assertRaisesRegex(RuntimeError, "recovery trees are preserved"):
                    deploy_animation_direction(
                        profile,
                        {"idle": [idle]},
                        "hero",
                        "east",
                        "./assets",
                        deploy_root,
                    )

            backups = list((deploy_root / "hero").glob(".east.assetforge-backup-*"))
            self.assertEqual(len(backups), 1)
            self.assertTrue((backups[0] / "idle" / "idle_00.png").is_file())

    def test_keyboard_interrupt_during_direction_swap_restores_previous_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            source = root / "source"
            source.mkdir()
            idle = source / "idle_00.png"
            old_bytes = _frame(idle, (70, 90, 130, 255))
            profile = _profile(project, "web")
            deploy_root = project / "assets"
            deploy_animation_direction(
                profile,
                {"idle": [idle]},
                "hero",
                "east",
                "./assets",
                deploy_root,
            )
            _frame(idle, (220, 100, 60, 255))
            direction_root = deploy_root / "hero" / "east"
            real_replace = os.replace
            interrupted = False

            def interrupt_new_tree(source_path: str | Path, target_path: str | Path) -> None:
                nonlocal interrupted
                if (
                    Path(target_path).resolve() == direction_root.resolve()
                    and ".assetforge-stage-" in Path(source_path).name
                    and not interrupted
                ):
                    interrupted = True
                    raise KeyboardInterrupt()
                real_replace(source_path, target_path)

            with patch.object(exporters.os, "replace", side_effect=interrupt_new_tree):
                with self.assertRaises(KeyboardInterrupt):
                    deploy_animation_direction(
                        profile,
                        {"idle": [idle]},
                        "hero",
                        "east",
                        "./assets",
                        deploy_root,
                    )

            self.assertEqual(
                (direction_root / "idle" / "idle_00.png").read_bytes(),
                old_bytes,
            )

    def test_clip_rollback_failure_preserves_recovery_copies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            source = root / "source"
            artifact = root / "artifact"
            destination = project / "assets" / "runtime"
            for directory in (source, artifact, destination):
                directory.mkdir(parents=True)
            old_bytes = _frame(destination / "walk_00.png", (20, 30, 40, 255))
            _frame(source / "walk_00.png", (200, 100, 60, 255))
            descriptor = artifact / "walk.json"
            descriptor.write_text("old descriptor\n", encoding="utf-8")
            real_replace = os.replace
            descriptor_failed = False

            def fail_commit_and_restore(source_path: str | Path, target_path: str | Path) -> None:
                nonlocal descriptor_failed
                source = Path(source_path)
                target = Path(target_path)
                if target.resolve() == descriptor.resolve() and not descriptor_failed:
                    descriptor_failed = True
                    raise OSError("simulated descriptor failure")
                if (
                    target.resolve() == (destination / "walk_00.png").resolve()
                    and source.parent.name == "previous"
                ):
                    raise OSError("simulated frame restore failure")
                real_replace(source_path, target_path)

            with patch.object(exporters.os, "replace", side_effect=fail_commit_and_restore):
                with self.assertRaisesRegex(RuntimeError, "recovery copies are preserved"):
                    export_web_registry(
                        _profile(project, "web"),
                        source,
                        descriptor,
                        "hero",
                        "runtime",
                        "walk",
                        "east",
                        "./assets/runtime",
                        destination,
                    )

            recovery = list((project / "assets").glob(".assetforge-deploy-*"))
            self.assertEqual(len(recovery), 1)
            self.assertEqual(
                (recovery[0] / "previous" / "walk_00.png").read_bytes(),
                old_bytes,
            )


if __name__ == "__main__":
    unittest.main()
