import tkinter as tk
import unittest

import matplotlib

matplotlib.use("Agg")  # the tool embeds its own Tk canvas; Agg keeps pyplot from opening windows
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.contrast_stretching import ContrastStretchingTool
from forensics_app.tools.histogram import HistogramTool


def _document(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    return document


def _ramp(low: int, high: int, size: tuple[int, int] = (64, 32)) -> Image.Image:
    width, height = size
    return Image.fromarray(np.tile(np.linspace(low, high, width), (height, 1)).astype(np.uint8), mode="L")


class HistogramToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.root = tk.Tk()
        except tk.TclError as error:  # no display available
            raise unittest.SkipTest(f"Tk unavailable: {error}")
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.root.destroy()

    def tearDown(self) -> None:
        for window in self.root.winfo_children():
            window.destroy()
        plt.close("all")

    def test_grayscale_reports_min_max_mean_without_changing_image(self) -> None:
        document = _document(_ramp(10, 200))
        result = HistogramTool().run(self.root, document)
        self.assertIsNone(result.image)
        self.assertEqual(result.details["Min Intensity"], 10)
        self.assertEqual(result.details["Max Intensity"], 200)
        self.assertEqual(result.details["Dimensions"], "64x32")
        self.assertEqual(np.asarray(document.current).max(), 200)

    def test_rgb_reports_one_mean_per_channel(self) -> None:
        result = HistogramTool().run(self.root, _document(Image.new("RGB", (5, 5), (255, 128, 0))))
        self.assertEqual(result.details["Red Mean"], "255.00")
        self.assertEqual(result.details["Green Mean"], "128.00")
        self.assertEqual(result.details["Blue Mean"], "0.00")

    def test_la_image_uses_luminance_and_ignores_alpha(self) -> None:
        image = Image.merge("LA", (_ramp(10, 200), Image.new("L", (64, 32), 255)))
        result = HistogramTool().run(self.root, _document(image))
        self.assertEqual(result.details["Min Intensity"], 10)
        self.assertEqual(result.details["Max Intensity"], 200)

    def test_palette_image_uses_real_colours_not_indices(self) -> None:
        image = Image.new("P", (2, 2), 0)
        image.putpalette([255, 0, 0] + [0] * 765)  # index 0 is pure red
        result = HistogramTool().run(self.root, _document(image))
        self.assertEqual(result.details["Red Mean"], "255.00")
        self.assertEqual(result.details["Green Mean"], "0.00")

    def test_other_modes_do_not_crash(self) -> None:
        for mode in ("1", "I", "F", "RGBA", "CMYK"):
            with self.subTest(mode=mode):
                result = HistogramTool().run(self.root, _document(Image.new(mode, (8, 8))))
                self.assertIsNone(result.image)

    def test_figures_do_not_accumulate_in_pyplot(self) -> None:
        before = len(plt.get_fignums())
        for _ in range(3):
            HistogramTool().run(self.root, _document(_ramp(0, 255)))
        self.assertEqual(len(plt.get_fignums()), before)

    def test_opens_a_popup_window(self) -> None:
        HistogramTool().run(self.root, _document(_ramp(0, 255)))
        popups = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
        self.assertEqual(len(popups), 1)
        self.assertEqual(popups[0].title(), "Intensity Histogram")


class ContrastStretchingTests(unittest.TestCase):
    def test_grayscale_is_stretched_to_full_range(self) -> None:
        document = _document(_ramp(100, 140))
        result = ContrastStretchingTool().run(None, document)
        arr = np.asarray(result.image)
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(arr.min(), 0)
        self.assertEqual(arr.max(), 255)
        self.assertEqual(np.asarray(document.current).max(), 140)  # document untouched

    def test_rgb_channels_are_stretched_independently(self) -> None:
        image = Image.merge("RGB", (_ramp(100, 140), _ramp(0, 50), _ramp(200, 255)))
        result = ContrastStretchingTool().run(None, _document(image))
        self.assertEqual(result.image.mode, "RGB")
        for channel in result.image.split():
            arr = np.asarray(channel)
            self.assertEqual((arr.min(), arr.max()), (0, 255))
        self.assertIn("R [p2, p98]", result.details)
        self.assertIn("B [p2, p98]", result.details)

    def test_percentiles_ignore_outliers(self) -> None:
        arr = np.full((10, 10), 120, dtype=np.uint8)
        arr[0, 0], arr[0, 1] = 0, 255  # 2% outliers
        arr[5:, :] = 130
        stretched, p_low, p_high = ContrastStretchingTool._stretch_channel(arr)
        self.assertGreaterEqual(p_low, 100)
        self.assertLessEqual(p_high, 130)
        self.assertEqual(stretched[9, 9], 255)

    def test_rgba_keeps_alpha_unchanged(self) -> None:
        image = Image.merge("RGB", (_ramp(100, 140),) * 3).convert("RGBA")
        image.putalpha(_ramp(0, 30))  # alpha must not be stretched to 0..255
        result = ContrastStretchingTool().run(None, _document(image))
        self.assertEqual(result.image.mode, "RGBA")
        np.testing.assert_array_equal(np.asarray(result.image.getchannel("A")), np.asarray(_ramp(0, 30)))
        self.assertEqual(np.asarray(result.image.getchannel("R")).max(), 255)

    def test_la_is_stretched_and_keeps_alpha(self) -> None:
        image = Image.merge("LA", (_ramp(100, 140), Image.new("L", (64, 32), 128)))
        result = ContrastStretchingTool().run(None, _document(image))
        self.assertEqual(result.image.mode, "LA")
        self.assertEqual(np.asarray(result.image.getchannel("L")).max(), 255)
        self.assertTrue(np.all(np.asarray(result.image.getchannel("A")) == 128))

    def test_palette_uses_real_colours_not_indices(self) -> None:
        # Index 0 is bright, index 1 is dark: stretching the indices would invert the image.
        image = Image.new("P", (2, 1))
        image.putpalette([200, 200, 200, 50, 50, 50] + [0] * 762)
        image.putpixel((0, 0), 0)
        image.putpixel((1, 0), 1)
        result = ContrastStretchingTool().run(None, _document(image))
        self.assertEqual(result.image.mode, "RGB")
        self.assertGreater(result.image.getpixel((0, 0))[0], result.image.getpixel((1, 0))[0])

    def test_other_modes_do_not_crash(self) -> None:
        for mode in ("1", "I", "F", "CMYK", "YCbCr"):
            with self.subTest(mode=mode):
                result = ContrastStretchingTool().run(None, _document(Image.new(mode, (8, 8))))
                self.assertIn(result.image.mode, ("L", "RGB"))

    def test_flat_image_is_left_unchanged(self) -> None:
        result = ContrastStretchingTool().run(None, _document(Image.new("L", (8, 8), 77)))
        self.assertTrue(np.all(np.asarray(result.image) == 77))


if __name__ == "__main__":
    unittest.main()
