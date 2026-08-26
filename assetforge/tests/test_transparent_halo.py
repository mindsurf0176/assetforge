import unittest

from PIL import Image, ImageDraw

from assetforge.frames import remove_neutral_edge_halo


class TransparentHaloTests(unittest.TestCase):
    def test_removes_border_connected_white_matte(self) -> None:
        image = Image.new("RGBA", (9, 9), (255, 255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((2, 2, 6, 6), fill=(45, 50, 58, 255))
        draw.point((4, 4), fill=(250, 250, 250, 255))

        cleaned = remove_neutral_edge_halo(image)

        self.assertEqual(cleaned.getpixel((0, 0))[3], 0)
        self.assertEqual(cleaned.getpixel((1, 1))[3], 0)
        self.assertEqual(cleaned.getpixel((4, 4)), (250, 250, 250, 255))
        self.assertEqual(cleaned.getpixel((4, 3)), (45, 50, 58, 255))

    def test_removes_semitransparent_border_matte_before_alpha_hardening(self) -> None:
        image = Image.new("RGBA", (7, 7), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 6, 6), outline=(255, 255, 255, 96), width=1)
        draw.rectangle((2, 2, 4, 4), fill=(45, 50, 58, 255))

        cleaned = remove_neutral_edge_halo(image)

        self.assertEqual(cleaned.getpixel((0, 3))[3], 0)
        self.assertEqual(cleaned.getpixel((3, 3)), (45, 50, 58, 255))
