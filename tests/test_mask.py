import os
import tempfile
import unittest
from unittest import mock

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.mask import (
    MaskSettings,
    MaskTool,
    apply_mask,
    binarize_mask,
    build_mask,
    load_image_file,
    selected_fraction,
)


def gradient() -> Image.Image:
    """4×1 grayscale image with values 0, 100, 200, 255."""
    image = Image.new("L", (4, 1))
    image.putdata([0, 100, 200, 255])
    return image


def row_mask(*values: int) -> Image.Image:
    mask = Image.new("L", (len(values), 1))
    mask.putdata(list(values))
    return mask


class BinarizeMaskTests(unittest.TestCase):
    def test_bright_pixels_become_selected(self) -> None:
        mask = binarize_mask(gradient(), (4, 1))
        self.assertEqual(mask.mode, "L")
        self.assertEqual(list(mask.getdata()), [0, 0, 255, 255])

    def test_different_size_is_resized(self) -> None:
        mask = Image.new("L", (2, 2), 0)
        mask.putpixel((0, 0), 255)  # top-left quarter selected
        resized = binarize_mask(mask, (8, 8))
        self.assertEqual(resized.size, (8, 8))
        self.assertEqual(resized.getpixel((0, 0)), 255)
        self.assertEqual(resized.getpixel((7, 7)), 0)
        self.assertEqual(set(resized.getdata()), {0, 255})

    def test_loads_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "mask.png")
            Image.new("RGB", (4, 1), "white").save(path)
            mask = load_image_file(path)
        self.assertEqual(mask.size, (4, 1))


class BuildMaskTests(unittest.TestCase):
    def test_resizes_and_inverts(self) -> None:
        settings = MaskSettings(mask_path="m.png", mask_image=row_mask(255, 0), invert=True)
        self.assertEqual(list(build_mask(settings, (4, 1)).getdata()), [0, 0, 255, 255])


class ApplyMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (2, 1), (10, 20, 30))
        self.mask = row_mask(255, 0)

    def test_keeps_selected_and_blacks_out_rest(self) -> None:
        output = apply_mask(self.image, self.mask)
        self.assertEqual(list(output.getdata()), [(10, 20, 30), (0, 0, 0)])

    def test_rest_is_opaque_black_when_image_has_alpha(self) -> None:
        output = apply_mask(Image.new("RGBA", (2, 1), (10, 20, 30, 100)), self.mask)
        self.assertEqual(list(output.getdata()), [(10, 20, 30, 100), (0, 0, 0, 255)])

    def test_rest_comes_from_fill_image(self) -> None:
        fill = Image.new("RGB", (2, 1), (200, 100, 50))
        output = apply_mask(self.image, self.mask, fill)
        self.assertEqual(list(output.getdata()), [(10, 20, 30), (200, 100, 50)])

    def test_fill_image_is_resized_and_converted(self) -> None:
        fill = Image.new("L", (8, 8), 77)  # different size and mode
        output = apply_mask(self.image, self.mask, fill)
        self.assertEqual(output.size, (2, 1))
        self.assertEqual(output.getpixel((1, 0)), (77, 77, 77))

    def test_does_not_mutate_input(self) -> None:
        apply_mask(self.image, self.mask)
        self.assertEqual(self.image.getpixel((1, 0)), (10, 20, 30))

    def test_selected_fraction(self) -> None:
        self.assertEqual(selected_fraction(self.mask), 0.5)


class MaskToolTests(unittest.TestCase):
    """The dialog is replaced by a mock so these tests need no display."""

    def setUp(self) -> None:
        self.document = ImageDocument()
        self.document.current = gradient()

    def _run_with_dialog_result(self, settings):
        with mock.patch("forensics_app.tools.mask.MaskDialog") as dialog:
            dialog.return_value.result = settings
            return MaskTool().run(None, self.document)

    def test_black_fill(self) -> None:
        result = self._run_with_dialog_result(MaskSettings(mask_path="/x/m.png", mask_image=row_mask(0, 255, 0, 0)))
        self.assertEqual(list(result.image.getdata()), [(0, 0, 0), (100, 100, 100), (0, 0, 0), (0, 0, 0)])
        self.assertEqual(result.details["Mask"], "m.png")
        self.assertEqual(result.details["Rest filled with"], "Black")
        self.assertNotIn("Mask resized", result.details)

    def test_mask_of_other_size_is_resized_and_reported(self) -> None:
        result = self._run_with_dialog_result(MaskSettings(mask_path="/x/m.png", mask_image=row_mask(0, 255)))
        self.assertEqual(result.image.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(result.image.getpixel((3, 0)), (255, 255, 255))
        self.assertEqual(result.details["Mask resized"], "2×1 → 4×1")

    def test_fill_image_replaces_black_areas(self) -> None:
        result = self._run_with_dialog_result(
            MaskSettings(
                mask_path="/x/m.png",
                mask_image=row_mask(0, 0, 255, 255),
                fill_path="/x/other.jpg",
                fill_image=Image.new("RGB", (2, 2), (9, 8, 7)),
            )
        )
        self.assertEqual(list(result.image.getdata()), [(9, 8, 7), (9, 8, 7), (200, 200, 200), (255, 255, 255)])
        self.assertEqual(result.details["Rest filled with"], "Image other.jpg")
        self.assertEqual(result.details["Fill image resized"], "2×2 → 4×1")

    def test_invert_is_reported(self) -> None:
        result = self._run_with_dialog_result(
            MaskSettings(mask_path="/x/m.png", mask_image=row_mask(0, 255, 0, 0), invert=True)
        )
        self.assertEqual(result.details["Mask"], "NOT m.png")

    def test_cancel_returns_none(self) -> None:
        self.assertIsNone(self._run_with_dialog_result(None))


if __name__ == "__main__":
    unittest.main()
