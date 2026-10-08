import unittest
from unittest import mock

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.threshold import ThresholdSettings, ThresholdTool, threshold_mask


def gradient() -> Image.Image:
    """4×1 grayscale image with values 0, 100, 200, 255."""
    image = Image.new("L", (4, 1))
    image.putdata([0, 100, 200, 255])
    return image


class ThresholdMaskTests(unittest.TestCase):
    def test_above_threshold(self) -> None:
        mask = threshold_mask(gradient(), ThresholdSettings(low=150, high=255))
        self.assertEqual(list(mask.getdata()), [0, 0, 255, 255])

    def test_below_threshold(self) -> None:
        mask = threshold_mask(gradient(), ThresholdSettings(low=0, high=100))
        self.assertEqual(list(mask.getdata()), [255, 255, 0, 0])

    def test_bounds_are_inclusive(self) -> None:
        mask = threshold_mask(gradient(), ThresholdSettings(low=100, high=200))
        self.assertEqual(list(mask.getdata()), [0, 255, 255, 0])

    def test_invert_selects_outside_the_range(self) -> None:
        mask = threshold_mask(gradient(), ThresholdSettings(low=100, high=200, invert=True))
        self.assertEqual(list(mask.getdata()), [255, 0, 0, 255])

    def test_single_colour_channel(self) -> None:
        image = Image.new("RGB", (2, 1))
        image.putdata([(255, 0, 0), (0, 0, 255)])
        mask = threshold_mask(image, ThresholdSettings(channel="B", low=128, high=255))
        self.assertEqual(list(mask.getdata()), [0, 255])

    def test_rejects_bad_range_and_channel(self) -> None:
        with self.assertRaises(ValueError):
            threshold_mask(gradient(), ThresholdSettings(low=200, high=100))
        with self.assertRaises(ValueError):
            threshold_mask(gradient(), ThresholdSettings(channel="A"))


class ThresholdToolTests(unittest.TestCase):
    """The dialog is replaced by a mock so these tests need no display."""

    def setUp(self) -> None:
        self.document = ImageDocument()
        self.document.current = gradient()

    def _run_with_dialog_result(self, settings):
        with mock.patch("forensics_app.tools.threshold.ThresholdDialog") as dialog:
            dialog.return_value.result = settings
            return ThresholdTool().run(None, self.document)

    def test_keeps_bright_pixels_and_blacks_out_the_rest(self) -> None:
        result = self._run_with_dialog_result(ThresholdSettings(low=150, high=255))
        self.assertEqual(list(result.image.getdata()), [(0, 0, 0), (0, 0, 0), (200, 200, 200), (255, 255, 255)])
        self.assertIn("50.00%", result.details["Kept pixels"])
        self.assertEqual(self.document.current.mode, "L")  # document untouched

    def test_invert_keeps_the_complement(self) -> None:
        result = self._run_with_dialog_result(ThresholdSettings(low=150, high=255, invert=True))
        self.assertEqual(list(result.image.getdata()), [(0, 0, 0), (100, 100, 100), (0, 0, 0), (0, 0, 0)])
        self.assertTrue(result.details["Selection"].startswith("NOT"))

    def test_cancel_returns_none(self) -> None:
        self.assertIsNone(self._run_with_dialog_result(None))


if __name__ == "__main__":
    unittest.main()
