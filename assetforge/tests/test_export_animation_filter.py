from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from assetforge.exporters import export_godot_spriteframes, export_web_registry
from assetforge.profile import Profile


def _profile(root: Path, engine: str) -> Profile:
    return Profile(
        path=root / "profile.json",
        data={
            "schemaVersion": 1,
            "id": f"mixed-export-{engine}",
            "kind": "pixel-character",
            "projectRoot": str(root),
            "tiers": {"runtime": {"canvasPolicy": "fixed", "canvas": [8, 8]}},
            "animations": {
                "walk": {"minFrames": 2, "maxFrames": 2, "fps": 8, "loop": True},
                "attack": {"minFrames": 1, "maxFrames": 1, "fps": 10, "loop": False},
            },
            "quality": {},
            "export": {"engine": engine, "resourcePrefix": "./frames"},
        },
    )


def _mixed_frames(root: Path) -> None:
    for name in ("walk_00.png", "walk_01.png", "attack_00.png"):
        Image.new("RGBA", (8, 8), (120, 140, 160, 255)).save(root / name)


class AnimationFilteredExportTests(unittest.TestCase):
    def test_web_registry_exports_only_requested_animation_from_mixed_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _mixed_frames(root)

            result = export_web_registry(
                _profile(root, "web"),
                root,
                root / "registry.json",
                "companion",
                "runtime",
                "walk",
                "south",
                deploy_dir=root / "frames",
            )
            registry = json.loads((root / "registry.json").read_text(encoding="utf-8"))
            paths = registry["characters"]["companion"]["tiers"]["runtime"]["animations"]["walk"]

            self.assertEqual(result["frames"], 2)
            self.assertEqual(
                paths["directions"]["south"],
                ["./frames/walk_00.png", "./frames/walk_01.png"],
            )
            self.assertEqual(result["verifiedReferences"], 2)
            self.assertTrue(all(Path(path).is_file() for path in result["localReferencePaths"]))
            self.assertEqual(
                (root / "walk_00.png").read_bytes(),
                (root / "frames" / "walk_00.png").read_bytes(),
            )

    def test_godot_spriteframes_exports_only_requested_animation_from_mixed_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _mixed_frames(root)

            result = export_godot_spriteframes(
                _profile(root, "godot"),
                root,
                root / "walk.tres",
                "walk",
                deploy_dir=root / "frames",
            )
            resource = (root / "walk.tres").read_text(encoding="utf-8")

            self.assertEqual(result["frames"], 2)
            self.assertIn("walk_00.png", resource)
            self.assertIn("walk_01.png", resource)
            self.assertNotIn("attack_00.png", resource)
            self.assertEqual(resource.count('[ext_resource type="Texture2D"'), 2)
            self.assertEqual(result["verifiedReferences"], 2)
            self.assertTrue(all(Path(path).is_file() for path in result["localReferencePaths"]))

    def test_default_export_creates_isolated_output_adjacent_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            artifact = root / "artifact"
            source.mkdir()
            artifact.mkdir()
            _mixed_frames(source)
            profile = _profile(root / "project", "godot")

            result = export_godot_spriteframes(
                profile,
                source,
                artifact / "walk.tres",
                "walk",
                "res://frames",
            )

            self.assertEqual(result["deploymentMode"], "artifact")
            self.assertEqual(Path(result["artifactRoot"]), artifact.resolve())
            self.assertEqual(Path(result["deployDir"]), (artifact / "frames").resolve())
            self.assertTrue(all(Path(path).is_file() for path in result["localReferencePaths"]))
            references = [
                line.split('path="', 1)[1].split('"', 1)[0]
                for line in (artifact / "walk.tres").read_text(encoding="utf-8").splitlines()
                if line.startswith("[ext_resource")
            ]
            self.assertTrue(
                all(
                    (artifact / reference.removeprefix("res://")).is_file()
                    for reference in references
                ),
                references,
            )

    def test_project_asset_write_requires_explicit_deploy_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            _mixed_frames(source)

            with self.assertRaisesRegex(ValueError, r"implicit export would modify project assets"):
                export_godot_spriteframes(
                    _profile(root, "godot"),
                    source,
                    root / "walk.tres",
                    "walk",
                    "res://frames",
                )

    def test_explicit_web_deploy_resolves_prefix_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "project"
            source = root / "source"
            artifact = root / "artifact"
            project.mkdir()
            source.mkdir()
            artifact.mkdir()
            _mixed_frames(source)
            deploy_dir = project / "assets" / "runtime" / "companion"

            result = export_web_registry(
                _profile(project, "web"),
                source,
                artifact / "companion-walk.json",
                "companion",
                "runtime",
                "walk",
                "south",
                "./assets/runtime/companion",
                deploy_dir,
            )

            self.assertEqual(result["deploymentMode"], "explicit")
            self.assertEqual(Path(result["artifactRoot"]), project.resolve())
            self.assertEqual(Path(result["deployDir"]), deploy_dir.resolve())
            self.assertTrue((deploy_dir / "walk_00.png").is_file())
            self.assertTrue((deploy_dir / "walk_01.png").is_file())

    def test_export_fails_when_requested_animation_has_no_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.new("RGBA", (8, 8), (120, 140, 160, 255)).save(
                root / "attack_00.png"
            )

            with self.assertRaisesRegex(ValueError, r"no PNG frames found for animation 'walk'"):
                export_web_registry(
                    _profile(root, "web"),
                    root,
                    root / "registry.json",
                    "companion",
                    "runtime",
                    "walk",
                    "south",
                )


if __name__ == "__main__":
    unittest.main()
