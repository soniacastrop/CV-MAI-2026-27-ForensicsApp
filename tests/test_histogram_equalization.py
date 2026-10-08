import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.histogram_equalization import ClaheTool, HistogramEqualizationTool


def _low_contrast(mode: str) -> Image.Image:
    """A gradient squeezed into the narrow range [100, 140]."""
    ramp = np.tile(np.linspace(100, 140, 64), (32, 1)).astype(np.uint8)
    gray = Image.fromarray(ramp, mode="L")
    return gray if mode == "L" else Image.merge("RGB", (gray, gray, gray))


class HistogramEqualizationTests(unittest.TestCase):
    def test_global_equalization_expands_grayscale_range(self) -> None:
        document = ImageDocument()
        document.current = _low_contrast("L")
        result = HistogramEqualizationTool().run(None, document)
        arr = np.asarray(result.image)
        self.assertEqual(result.image.mode, "L")
        self.assertLess(arr.min(), 20)
        self.assertGreater(arr.max(), 235)
        self.assertEqual(np.asarray(document.current).max(), 140)  # document untouched

    def test_global_equalization_keeps_rgb_mode_and_size(self) -> None:
        document = ImageDocument()
        document.current = _low_contrast("RGB")
        result = HistogramEqualizationTool().run(None, document)
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, document.current.size)

    def test_global_equalization_preserves_colour(self) -> None:
        # A low-contrast reddish image: equalizing luma only must keep it reddish (no hue shift).
        document = ImageDocument()
        document.current = Image.merge(
            "RGB", (_low_contrast("L"), Image.new("L", (64, 32), 60), Image.new("L", (64, 32), 60))
        )
        result = HistogramEqualizationTool().run(None, document)
        _y_in, cb_in, cr_in = (np.asarray(c, float) for c in document.current.convert("YCbCr").split())
        y_out, cb_out, cr_out = (np.asarray(c, float) for c in result.image.convert("YCbCr").split())
        # Near pure black/white the RGB gamut forces colours to clip, so compare mid-tones only.
        mid = (y_out > 40) & (y_out < 215)
        self.assertTrue(mid.any())
        self.assertLess(np.abs(cr_out - cr_in)[mid].mean(), 3)
        self.assertLess(np.abs(cb_out - cb_in)[mid].mean(), 3)
        r, g, _b = (np.asarray(c, float) for c in result.image.split())
        self.assertTrue(np.all(r[mid] > g[mid]))  # still reddish
        self.assertEqual(result.details["Channel"], "Y (luma)")

    def test_flat_image_does_not_crash(self) -> None:
        for tool in (HistogramEqualizationTool(), ClaheTool()):
            with self.subTest(tool=tool.tool_id):
                document = ImageDocument()
                document.current = Image.new("L", (16, 16), 128)
                with mock.patch(
                    "forensics_app.tools.histogram_equalization.simpledialog.askfloat", return_value=0.01
                ):
                    result = tool.run(None, document)
                self.assertEqual(result.image.size, (16, 16))

    def test_clahe_increases_contrast(self) -> None:
        document = ImageDocument()
        document.current = _low_contrast("RGB")
        with mock.patch("forensics_app.tools.histogram_equalization.simpledialog.askfloat", return_value=0.03):
            result = ClaheTool().run(None, document)
        before = np.asarray(document.current.convert("L"), dtype=float).std()
        after = np.asarray(result.image.convert("L"), dtype=float).std()
        self.assertEqual(result.image.mode, "RGB")
        self.assertGreater(after, before)
        self.assertEqual(result.details["Clip limit"], 0.03)

    def test_clahe_cancel_returns_none(self) -> None:
        document = ImageDocument()
        document.current = _low_contrast("L")
        with mock.patch("forensics_app.tools.histogram_equalization.simpledialog.askfloat", return_value=None):
            self.assertIsNone(ClaheTool().run(None, document))


if __name__ == "__main__":
    unittest.main()
