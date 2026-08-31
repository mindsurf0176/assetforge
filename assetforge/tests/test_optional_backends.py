from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from assetforge.generation_backends import (
    SheetRequest,
    SheetResult,
    codex_imagegen_sheet_prompt,
    write_generation_manifest,
)
from assetforge.vision_backends import sam2_status


class OptionalBackendTests(unittest.TestCase):
    def test_generation_manifest_is_portable_and_records_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = SheetRequest("moa", "walk", "east", 12, 12, reference=root / "master.png")
            result = SheetResult(root / "sheet.png", "test", {"seed": 7})
            manifest = write_generation_manifest(result, request, root / "out")
            text = manifest.read_text(encoding="utf-8")
            self.assertIn('"frameCount": 12', text)
            self.assertNotIn(str(root.resolve()), text)

    def test_sam2_status_is_lazy(self) -> None:
        status = sam2_status()
        self.assertIn("available", status)

    def test_codex_prompt_requires_one_complete_sheet(self) -> None:
        prompt = codex_imagegen_sheet_prompt(SheetRequest("moa", "walk", "east", 12, 12))
        self.assertIn("exactly 12 unique poses", prompt)
        self.assertIn("genuinely transparent background", prompt)
        self.assertIn("one shared horizontal ground line", prompt)
