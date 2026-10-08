import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.canny import (
    DARKEN_FACTOR,
    DISPLAYS,
    CannySettings,
    CannyTool,
    detect_edges,
    label_contours,
    render_edges,
)
from forensics_app.tools.convolution import ConvolutionTool
from forensics_app.tools.skimage_filters import SkimageFiltersTool

FILTERS = "forensics_app.tools.skimage_filters.simpledialog"
CANNY_DIALOG = "forensics_app.tools.canny.CannyDialog"


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


def _gray_square_on_blue(size: int = 40) -> Image.Image:
    """A grey square on a blue background, so overlays can be told apart from the image."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :] = (0, 0, 200)
    arr[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = (160, 160, 160)
    return Image.fromarray(arr, mode="RGB")


class CannyToolTests(unittest.TestCase):
    """The dialog is replaced by a mock so these tests need no display."""

    def _run(self, image: Image.Image, settings):
        with mock.patch(CANNY_DIALOG) as dialog:
            dialog.return_value.result = settings
            return CannyTool().run(None, _document(image))

    def test_edge_map_finds_thin_edges_around_square(self) -> None:
        result = self._run(_square(), CannySettings(sigma=1.0))
        arr = np.asarray(result.image)
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(set(np.unique(arr)), {0, 255})
        self.assertEqual(arr[20, 20], 0)
        self.assertGreater(result.details["Edge Pixels"], 0)
        self.assertEqual(result.details["Configured Sigma"], 1.0)

    def test_flat_image_has_no_edges(self) -> None:
        result = self._run(Image.new("RGB", (20, 20), "white"), CannySettings(sigma=1.0))
        self.assertEqual(result.details["Edge Pixels"], 0)

    def test_cancel_returns_none(self) -> None:
        self.assertIsNone(self._run(_square(), None))

    def test_every_display_produces_an_image_of_same_size(self) -> None:
        for display in DISPLAYS:
            with self.subTest(display=display):
                result = self._run(_gray_square_on_blue(), CannySettings(sigma=1.0, display=display))
                self.assertEqual(result.image.size, (40, 40))
                self.assertEqual(result.details["Display"], DISPLAYS[display])

    def test_contours_reports_count(self) -> None:
        result = self._run(_square(), CannySettings(sigma=1.0, display="contours"))
        self.assertGreaterEqual(result.details["Contours"], 1)

    def test_does_not_mutate_document(self) -> None:
        document = _document(_gray_square_on_blue())
        with mock.patch(CANNY_DIALOG) as dialog:
            dialog.return_value.result = CannySettings(sigma=1.0, display="red_edges")
            CannyTool().run(None, document)
        self.assertEqual(document.current.getpixel((0, 0)), (0, 0, 200))


class CannyRenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = _gray_square_on_blue()
        self.edges = detect_edges(self.image, 1.0)
        self.rgb = np.asarray(self.image)
        self.assertTrue(self.edges.any())

    def _render(self, display: str) -> np.ndarray:
        return np.asarray(render_edges(self.image, self.edges, display))

    def test_red_edges_are_pure_red_and_rest_unchanged(self) -> None:
        out = self._render("red_edges")
        self.assertTrue(np.all(out[self.edges] == (255, 0, 0)))
        np.testing.assert_array_equal(out[~self.edges], self.rgb[~self.edges])

    def test_red_overlay_blends_with_image(self) -> None:
        out = self._render("red_overlay").astype(int)
        edge_out, edge_in = out[self.edges], self.rgb[self.edges].astype(int)
        self.assertTrue(np.all(edge_out[:, 0] >= edge_in[:, 0]))  # redder...
        self.assertTrue(np.any(edge_out[:, 2] > 0))  # ...but the blue background still shows through
        self.assertFalse(np.all(edge_out == (255, 0, 0)))
        np.testing.assert_array_equal(out[~self.edges], self.rgb[~self.edges])

    def test_darken_keeps_edges_and_dims_the_rest(self) -> None:
        out = self._render("darken").astype(float)
        np.testing.assert_array_equal(out[self.edges], self.rgb[self.edges])
        np.testing.assert_allclose(out[~self.edges], self.rgb[~self.edges] * DARKEN_FACTOR, atol=1)

    def test_contours_get_distinct_colours(self) -> None:
        two_squares = np.zeros((40, 80), dtype=np.uint8)
        two_squares[10:30, 5:25] = 255
        two_squares[10:30, 50:70] = 255
        image = Image.fromarray(two_squares, mode="L")
        edges = detect_edges(image, 1.0)
        labels, count = label_contours(edges)
        self.assertEqual(count, 2)
        out = np.asarray(render_edges(image, edges, "contours"))
        colour_a = out[labels == 1][0]
        colour_b = out[labels == 2][0]
        self.assertFalse(np.array_equal(colour_a, colour_b))
        self.assertTrue(np.all(out[labels == 1] == colour_a))  # one colour per contour
        source = np.asarray(image.convert("RGB"))
        np.testing.assert_array_equal(out[~edges], source[~edges])  # non-edge pixels unchanged

    def test_grayscale_and_transparent_inputs_become_rgb(self) -> None:
        for image in (_square(), _gray_square_on_blue().convert("RGBA"), _square().convert("P")):
            with self.subTest(mode=image.mode):
                edges = detect_edges(image, 1.0)
                self.assertEqual(render_edges(image, edges, "red_overlay").mode, "RGB")

    def test_rejects_unknown_display(self) -> None:
        with self.assertRaises(ValueError):
            render_edges(self.image, self.edges, "sparkles")


if __name__ == "__main__":
    unittest.main()
