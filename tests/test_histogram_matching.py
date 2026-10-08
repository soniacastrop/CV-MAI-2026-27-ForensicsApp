import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.histogram_matching import HistogramMatchingTool

MODULE = "forensics_app.tools.histogram_matching"


def _ramp(low: int, high: int) -> Image.Image:
    return Image.fromarray(np.tile(np.linspace(low, high, 64), (32, 1)).astype(np.uint8), mode="L")


class HistogramMatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        handle, self.ref_path = tempfile.mkstemp(suffix=".png")
        os.close(handle)

    def tearDown(self) -> None:
        os.remove(self.ref_path)

    def _run(self, source: Image.Image, reference: Image.Image, per_channel=None):
        reference.save(self.ref_path)
        document = ImageDocument()
        document.current = source
        with mock.patch(f"{MODULE}.filedialog.askopenfilename", return_value=self.ref_path), mock.patch(
            f"{MODULE}.messagebox.askyesnocancel", return_value=per_channel
        ):
            return HistogramMatchingTool().run(None, document)

    def test_grayscale_result_takes_reference_range(self) -> None:
        result = self._run(_ramp(100, 140), _ramp(0, 255))
        arr = np.asarray(result.image)
        self.assertEqual(result.image.mode, "L")
        self.assertLess(arr.min(), 10)
        self.assertGreater(arr.max(), 245)

    def test_per_channel_copies_reference_colour_tone(self) -> None:
        gray = _ramp(50, 200)
        source = Image.merge("RGB", (gray, gray, gray))
        reference = Image.merge("RGB", (_ramp(150, 255), _ramp(0, 60), _ramp(0, 60)))  # reddish
        result = self._run(source, reference, per_channel=True)
        r, g, _ = (np.asarray(c, dtype=float).mean() for c in result.image.split())
        self.assertGreater(r, g + 50)

    def test_luminance_only_keeps_source_neutral(self) -> None:
        gray = _ramp(50, 200)
        source = Image.merge("RGB", (gray, gray, gray))
        reference = Image.merge("RGB", (_ramp(150, 255), _ramp(0, 60), _ramp(0, 60)))
        result = self._run(source, reference, per_channel=False)
        r, g, _ = (np.asarray(c, dtype=float).mean() for c in result.image.split())
        self.assertLess(abs(r - g), 3)

    def test_reference_of_different_size_keeps_source_size(self) -> None:
        result = self._run(_ramp(100, 140), _ramp(0, 255).resize((200, 10)))
        self.assertEqual(result.image.size, (64, 32))
        self.assertEqual(result.details["Reference size"], "200 × 10")

    def test_grayscale_source_does_not_ask_about_colour(self) -> None:
        _ramp(0, 255).save(self.ref_path)
        document = ImageDocument()
        document.current = _ramp(100, 140)
        with mock.patch(f"{MODULE}.filedialog.askopenfilename", return_value=self.ref_path), mock.patch(
            f"{MODULE}.messagebox.askyesnocancel"
        ) as ask:
            HistogramMatchingTool().run(None, document)
        ask.assert_not_called()

    def test_cancelling_colour_question_returns_none(self) -> None:
        gray = _ramp(50, 200)
        self.assertIsNone(self._run(Image.merge("RGB", (gray, gray, gray)), _ramp(0, 255), per_channel=None))

    def test_does_not_mutate_document(self) -> None:
        source = _ramp(100, 140)
        _ramp(0, 255).save(self.ref_path)
        document = ImageDocument()
        document.current = source
        with mock.patch(f"{MODULE}.filedialog.askopenfilename", return_value=self.ref_path):
            HistogramMatchingTool().run(None, document)
        self.assertEqual(np.asarray(document.current).max(), 140)

    def test_cancelling_file_dialog_returns_none(self) -> None:
        document = ImageDocument()
        document.current = _ramp(0, 255)
        with mock.patch(f"{MODULE}.filedialog.askopenfilename", return_value=""):
            self.assertIsNone(HistogramMatchingTool().run(None, document))


if __name__ == "__main__":
    unittest.main()
