import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.canny import CannyTool
from forensics_app.tools.convolution import ConvolutionTool
from forensics_app.tools.skimage_filters import SkimageFiltersTool

FILTERS = "forensics_app.tools.skimage_filters.simpledialog"
CANNY = "forensics_app.tools.canny.simpledialog"


def _document(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    return document


def _square(size: int = 40) -> Image.Image:
    """A white square on a black background: edges only around the square's border."""
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 255
    return Image.fromarray(arr, mode="L")


class ConvolutionTests(unittest.TestCase):
    def test_flat_image_has_zero_gradient(self) -> None:
        result = ConvolutionTool().run(None, _document(Image.new("RGB", (10, 10), "gray")))
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.image.size, (10, 10))
        self.assertEqual(np.asarray(result.image).max(), 0)

    def test_responds_only_at_edges(self) -> None:
        arr = np.asarray(ConvolutionTool().run(None, _document(_square())).image)
        self.assertGreater(arr[20, 10], 0)  # on the left border of the square
        self.assertEqual(arr[20, 20], 0)  # inside the square
        self.assertEqual(arr[2, 2], 0)  # background

    def test_does_not_mutate_document(self) -> None:
        document = _document(_square())
        ConvolutionTool().run(None, document)
        self.assertEqual(document.current.getpixel((20, 20)), 255)


class SkimageFiltersTests(unittest.TestCase):
    def _run(self, choice, sigma=None):
        with mock.patch(f"{FILTERS}.askstring", return_value=choice), mock.patch(
            f"{FILTERS}.askfloat", return_value=sigma
        ):
            return SkimageFiltersTool().run(None, _document(_square()))

    def test_sobel_and_prewitt_accept_name_or_number(self) -> None:
        for choice, algorithm in (("sobel", "Sobel"), ("1", "Sobel"), ("prewitt", "Prewitt"), ("2", "Prewitt")):
            with self.subTest(choice=choice):
                result = self._run(choice)
                self.assertTrue(result.details["Algorithm"].startswith(algorithm))
                arr = np.asarray(result.image)
                self.assertEqual(arr.max(), 255)  # normalized to full range
                self.assertEqual(arr[20, 20], 0)  # flat interior has no edges

    def test_choice_is_case_and_whitespace_insensitive(self) -> None:
        self.assertIn("Sobel", self._run("  SoBeL ").details["Algorithm"])

    def test_gaussian_uses_sigma_and_blurs(self) -> None:
        result = self._run("gaussian", sigma=3.0)
        self.assertEqual(result.details["Configurable Sigma"], 3.0)
        arr = np.asarray(result.image)
        self.assertTrue(0 < arr[20, 10] < 255)  # hard edge became a smooth transition

    def test_cancel_at_either_dialog_returns_none(self) -> None:
        self.assertIsNone(self._run(None))
        self.assertIsNone(self._run("gaussian", sigma=None))

    def test_unknown_choice_reports_error_without_image(self) -> None:
        result = self._run("median")
        self.assertIsNone(result.image)
        self.assertEqual(result.details["Error"], "Invalid choice")


class CannyTests(unittest.TestCase):
    def _run(self, image: Image.Image, sigma):
        with mock.patch(f"{CANNY}.askfloat", return_value=sigma):
            return CannyTool().run(None, _document(image))

    def test_finds_thin_edges_around_square(self) -> None:
        result = self._run(_square(), 1.0)
        arr = np.asarray(result.image)
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(set(np.unique(arr)), {0, 255})
        self.assertEqual(arr[20, 20], 0)
        self.assertGreater(result.details["Edge Pixels"], 0)
        self.assertEqual(result.details["Configured Sigma"], 1.0)

    def test_flat_image_has_no_edges(self) -> None:
        result = self._run(Image.new("RGB", (20, 20), "white"), 1.0)
        self.assertEqual(result.details["Edge Pixels"], 0)

    def test_cancel_returns_none(self) -> None:
        self.assertIsNone(self._run(_square(), None))


if __name__ == "__main__":
    unittest.main()
