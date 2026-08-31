from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from assetforge.pipeline import build_pipeline
from assetforge.tests.test_release import _profile


class PipelineTests(unittest.TestCase):
    def test_sheet_pipeline_is_generator_independent_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _profile(root)
            sheet = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
            draw = ImageDraw.Draw(sheet)
            draw.rectangle((6, 5, 13, 15), fill=(180, 140, 110, 255))
            draw.rectangle((26, 4, 33, 15), fill=(180, 140, 110, 255))
            sheet_path = root / "generated-by-any-model.png"
            sheet.save(sheet_path)
            metadata = root / "generation.json"
            metadata.write_text(json.dumps({"backend": "codex_imagegen", "seed": 42}), encoding="utf-8")

            result = build_pipeline(
                profile,
                sheet_path,
                root / "work",
                root / "export.json",
                character="demo",
                tier="runtime",
                animation="walk",
                direction="east",
                input_kind="sheet",
                columns=2,
                backend="codex_imagegen",
                generation_manifest=metadata,
            )

            self.assertTrue(result["ok"], result)
            pipeline_manifest = json.loads((root / "work" / "pipeline-manifest.json").read_text())
            self.assertEqual(pipeline_manifest["backend"], "codex_imagegen")
            self.assertEqual(pipeline_manifest["frameCount"], 2)
            self.assertEqual(pipeline_manifest["generationManifest"]["data"]["backend"], "codex_imagegen")
            self.assertTrue((root / "export.json").is_file())

    def test_sheet_pipeline_rejects_missing_grid_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _profile(root)
            sheet = root / "sheet.png"
            Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(sheet)
            with self.assertRaisesRegex(ValueError, "columns"):
                build_pipeline(
                    profile,
                    sheet,
                    root / "work",
                    root / "export.json",
                    character="demo",
                    tier="runtime",
                    animation="walk",
                    direction="east",
                    input_kind="sheet",
                )


if __name__ == "__main__":
    unittest.main()
