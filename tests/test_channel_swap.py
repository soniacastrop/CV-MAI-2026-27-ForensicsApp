import unittest
from unittest import mock

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.channel_swap import ChannelSwapTool, has_identical_colour_channels, swap_channels


class SwapChannelsTests(unittest.TestCase):
    def test_swaps_red_and_blue(self) -> None:
        output = swap_channels(Image.new("RGB", (4, 3), (255, 0, 128)), ("B", "G", "R"))
        self.assertEqual(output.getpixel((0, 0)), (128, 0, 255))

    def test_identity_order_leaves_pixels_unchanged(self) -> None:
        output = swap_channels(Image.new("RGB", (4, 3), (10, 20, 30)), ("R", "G", "B"))
        self.assertEqual(output.getpixel((0, 0)), (10, 20, 30))

    def test_same_source_twice_is_allowed(self) -> None:
        output = swap_channels(Image.new("RGB", (4, 3), (10, 20, 30)), ("R", "R", "R"))
        self.assertEqual(output.getpixel((0, 0)), (10, 10, 10))

    def test_keeps_alpha(self) -> None:
        output = swap_channels(Image.new("RGBA", (4, 3), (255, 0, 128, 64)), ("G", "R", "B"))
        self.assertEqual(output.mode, "RGBA")
        self.assertEqual(output.getpixel((0, 0)), (0, 255, 128, 64))

    def test_grayscale_input_becomes_rgb(self) -> None:
        output = swap_channels(Image.new("L", (4, 3), 50), ("B", "G", "R"))
        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.getpixel((0, 0)), (50, 50, 50))

    def test_rejects_unknown_band(self) -> None:
        with self.assertRaises(ValueError):
            swap_channels(Image.new("RGB", (4, 3)), ("R", "G", "A"))

    def test_does_not_mutate_input(self) -> None:
        image = Image.new("RGB", (4, 3), (255, 0, 128))
        swap_channels(image, ("B", "G", "R"))
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 128))


class ChannelSwapToolTests(unittest.TestCase):
    """The dialog is replaced by a mock so these tests need no display."""

    def setUp(self) -> None:
        self.document = ImageDocument()
        self.document.current = Image.new("RGB", (4, 3), (255, 0, 128))

    def _run_with_dialog_result(self, order):
        with mock.patch("forensics_app.tools.channel_swap.ChannelOrderDialog") as dialog:
            dialog.return_value.result = order
            return ChannelSwapTool().run(None, self.document)

    def test_returns_swapped_image_and_details(self) -> None:
        result = self._run_with_dialog_result(("B", "G", "R"))
        self.assertEqual(result.image.getpixel((0, 0)), (128, 0, 255))
        self.assertEqual(result.details["Output R"], "B")
        self.assertEqual(result.details["Output B"], "R")
        self.assertEqual(self.document.current.getpixel((0, 0)), (255, 0, 128))

    def test_cancel_returns_none(self) -> None:
        self.assertIsNone(self._run_with_dialog_result(None))

    def test_grayscale_image_is_skipped_without_opening_dialog(self) -> None:
        for gray in (Image.new("L", (4, 3), 50), Image.new("RGB", (4, 3), (50, 50, 50))):
            with self.subTest(mode=gray.mode):
                self.document.current = gray
                with mock.patch("forensics_app.tools.channel_swap.ChannelOrderDialog") as dialog:
                    result = ChannelSwapTool().run(None, self.document)
                dialog.assert_not_called()
                self.assertIsNone(result.image)
                self.assertIn("grayscale", result.message)


class HasIdenticalColourChannelsTests(unittest.TestCase):
    def test_grayscale_mode(self) -> None:
        self.assertTrue(has_identical_colour_channels(Image.new("L", (4, 3), 50)))

    def test_rgb_that_looks_gray(self) -> None:
        self.assertTrue(has_identical_colour_channels(Image.new("RGB", (4, 3), (50, 50, 50))))

    def test_one_coloured_pixel_counts_as_colour(self) -> None:
        image = Image.new("RGB", (4, 3), (50, 50, 50))
        image.putpixel((3, 2), (51, 50, 50))
        self.assertFalse(has_identical_colour_channels(image))


if __name__ == "__main__":
    unittest.main()
