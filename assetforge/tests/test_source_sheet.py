from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from assetforge.frames import harden_alpha, split_source_sheet


class SourceSheetTests(unittest.TestCase):
    def test_split_source_sheet_pads_remainder_to_one_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sheet = Image.new("RGBA", (10, 4), (236, 244, 241, 255))
            draw = ImageDraw.Draw(sheet)
            for index, color in enumerate(((200, 50, 50, 255), (50, 200, 50, 255), (50, 50, 200, 255))):
                draw.rectangle((index * 3 + 1, 1, index * 3 + 2, 2), fill=color)
            sheet_path = root / "sheet.png"
            sheet.save(sheet_path)

            paths = split_source_sheet(sheet_path, root / "frames", columns=3, crop_height=4, prefix="attack")

            self.assertEqual(len(paths), 3)
            sizes = []
            for path in paths:
                with Image.open(path) as opened:
                    sizes.append(opened.size)
            self.assertEqual(sizes, [(4, 4)] * 3)

    def test_harden_alpha_removes_semitransparent_halo(self) -> None:
        image = Image.new("RGBA", (3, 1), (10, 20, 30, 0))
        image.putpixel((1, 0), (10, 20, 30, 128))
        image.putpixel((2, 0), (10, 20, 30, 255))

        hardened = harden_alpha(image, 20)

        self.assertEqual([hardened.getpixel((x, 0))[3] for x in range(3)], [0, 255, 255])
