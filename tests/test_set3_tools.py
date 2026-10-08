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
from forensics_app.tools.convolution import (
    ConvolutionTool,
    KernelSettings,
    convolve_image,
    max_kernel_size,
)
from forensics_app.tools.skimage_filters import SkimageFiltersTool

FILTERS = "forensics_app.tools.skimage_filters.simpledialog"
CANNY_DIALOG = "forensics_app.tools.canny.CannyDialog"
CONV = "forensics_app.tools.convolution"


def _document(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    return document


def _square(size: int = 40) -> Image.Image:
    """A white square on a black background: edges only around the square's border."""
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 255
    return Image.fromarray(arr, mode="L")


def _random(mode: str = "L", size: tuple[int, int] = (12, 9)) -> Image.Image:
    rng = np.random.default_rng(0)
    bands = 1 if mode == "L" else 3
    arr = rng.integers(0, 256, size=(size[1], size[0], bands), dtype=np.uint8)
    return Image.fromarray(arr[:, :, 0] if bands == 1 else arr, mode=mode)


class ConvolutionMathTests(unittest.TestCase):
    def test_identity_kernel_returns_the_image(self) -> None:
        for mode in ("L", "RGB"):
            with self.subTest(mode=mode):
                image = _random(mode)
                kernel = np.zeros((3, 3))
                kernel[1, 1] = 1
                output, _low, _high = convolve_image(image, KernelSettings(kernel))
                np.testing.assert_array_equal(np.asarray(output), np.asarray(image))

    def test_single_value_kernel_scales_intensities(self) -> None:
        output, _low, _high = convolve_image(Image.new("L", (5, 5), 100), KernelSettings(np.array([[0.5]])))
        self.assertTrue(np.all(np.asarray(output) == 50))

    def test_matches_scipy_convolve2d_with_symmetric_borders(self) -> None:
        from scipy.signal import convolve2d

        image = _random("L")
        kernel = np.array([[1, 2, 0], [-1, 0, 3], [0, -2, 1]], dtype=float)
        output, _low, _high = convolve_image(image, KernelSettings(kernel))
        expected = convolve2d(np.asarray(image, float), kernel, mode="same", boundary="symm")
        np.testing.assert_array_equal(np.asarray(output), np.clip(np.round(expected), 0, 255).astype(np.uint8))

    def test_kernel_is_flipped_true_convolution(self) -> None:
        image = Image.new("L", (5, 5), 0)
        image.putpixel((2, 2), 100)
        kernel = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=float)  # weight on the left
        output = np.asarray(convolve_image(image, KernelSettings(kernel))[0])
        # out[x] = sum_k K[k] * img[x - k]: the weight at offset -1 reads img[x + 1],
        # so the impulse moves left (correlation would move it right)
        self.assertEqual(output[2, 1], 100)
        self.assertEqual(output[2, 3], 0)

    def test_normalize_divides_by_sum(self) -> None:
        image = _random("L")
        output, _low, _high = convolve_image(image, KernelSettings(np.ones((3, 3)), normalize=True))
        self.assertLessEqual(np.asarray(output).max(), np.asarray(image).max())  # an average, not a 9x sum
        with self.assertRaises(ValueError):
            convolve_image(image, KernelSettings(np.array([[1.0, -1.0], [0.0, 0.0]]), normalize=True))

    def test_output_mappings_for_negative_responses(self) -> None:
        image = Image.new("L", (6, 6), 0)
        image.paste(200, (3, 0, 6, 6))  # vertical step edge
        kernel = np.array([[-1.0, 0.0, 1.0]] * 3)  # flipped: img[x-1] - img[x+1] < 0 on a rising step
        clip = np.asarray(convolve_image(image, KernelSettings(kernel, output="clip"))[0])
        absolute = np.asarray(convolve_image(image, KernelSettings(kernel, output="abs"))[0])
        rescaled, low, high = convolve_image(image, KernelSettings(kernel, output="rescale"))
        self.assertEqual(clip.max(), 0)  # negative values clipped away
        self.assertEqual(absolute.max(), 255)  # |-600| clipped to 255
        self.assertLess(low, 0)
        self.assertEqual((np.asarray(rescaled).min(), np.asarray(rescaled).max()), (0, 255))

    def test_kernel_size_limits(self) -> None:
        image = Image.new("L", (10, 6))
        self.assertEqual(max_kernel_size(image), 5)  # n - 1 with n the smaller side
        convolve_image(image, KernelSettings(np.ones((5, 5))))  # largest allowed works
        with self.assertRaises(ValueError):
            convolve_image(image, KernelSettings(np.ones((6, 6))))
        with self.assertRaises(ValueError):
            convolve_image(image, KernelSettings(np.ones((2, 3))))

    def test_even_sized_kernel_keeps_image_size(self) -> None:
        output, _low, _high = convolve_image(_random("RGB"), KernelSettings(np.ones((4, 4)), normalize=True))
        self.assertEqual(output.size, (12, 9))

    def test_alpha_is_preserved(self) -> None:
        image = _random("RGB").convert("RGBA")
        image.putalpha(77)
        output, _low, _high = convolve_image(image, KernelSettings(np.ones((3, 3)), normalize=True))
        self.assertEqual(output.mode, "RGBA")
        self.assertTrue(np.all(np.asarray(output.getchannel("A")) == 77))


class ConvolutionToolTests(unittest.TestCase):
    """Both dialogs are replaced by mocks so these tests need no display."""

    def _run(self, image: Image.Image, size, settings):
        with mock.patch(f"{CONV}.simpledialog.askinteger", return_value=size) as ask, mock.patch(
            f"{CONV}.KernelDialog"
        ) as dialog:
            dialog.return_value.result = settings
            result = ConvolutionTool().run(None, _document(image))
        return result, ask, dialog

    def test_asks_size_within_1_and_n_minus_1_then_values(self) -> None:
        kernel = np.zeros((3, 3))
        kernel[1, 1] = 1
        result, ask, dialog = self._run(_square(), 3, KernelSettings(kernel))
        self.assertEqual(ask.call_args.kwargs["minvalue"], 1)
        self.assertEqual(ask.call_args.kwargs["maxvalue"], 39)
        self.assertEqual(dialog.call_args.args[1], 3)  # the grid is built for the chosen size
        self.assertEqual(result.details["Kernel size"], "3 × 3")
        self.assertEqual(result.details["Kernel"], "0 0 0; 0 1 0; 0 0 0")

    def test_cancel_size_skips_values_dialog(self) -> None:
        result, _ask, dialog = self._run(_square(), None, None)
        self.assertIsNone(result)
        dialog.assert_not_called()

    def test_cancel_values_returns_none(self) -> None:
        result, _ask, _dialog = self._run(_square(), 3, None)
        self.assertIsNone(result)

    def test_too_small_image_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._run(Image.new("L", (1, 10)), 1, None)

    def test_does_not_mutate_document(self) -> None:
        document = _document(_square())
        with mock.patch(f"{CONV}.simpledialog.askinteger", return_value=1), mock.patch(
            f"{CONV}.KernelDialog"
        ) as dialog:
            dialog.return_value.result = KernelSettings(np.array([[0.0]]))
            ConvolutionTool().run(None, document)
        self.assertEqual(document.current.getpixel((20, 20)), 255)


class SkimageFiltersTests(unittest.TestCase):
    def _run(self, choice, sigma=None, floats=None, integer=None, image=None):
        """``floats`` answers successive askfloat prompts (defaults to ``[sigma]``)."""
        with mock.patch(f"{FILTERS}.askstring", return_value=choice), mock.patch(
            f"{FILTERS}.askfloat", side_effect=list(floats) if floats is not None else [sigma]
        ), mock.patch(f"{FILTERS}.askinteger", return_value=integer):
            return SkimageFiltersTool().run(None, _document(image or _square()))

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
        result = self._run("bilateral")
        self.assertIsNone(result.image)
        self.assertEqual(result.details["Error"], "Invalid choice")

    def test_scharr_accepts_name_or_number(self) -> None:
        for choice in ("scharr", "5"):
            with self.subTest(choice=choice):
                result = self._run(choice)
                self.assertEqual(result.details["Algorithm"], "Scharr edge detector")
                arr = np.asarray(result.image)
                self.assertEqual(arr.max(), 255)
                self.assertEqual(arr[20, 20], 0)

    def test_median_removes_salt_noise_and_keeps_scale(self) -> None:
        noisy = Image.new("L", (30, 30), 100)
        for xy in ((5, 5), (15, 20), (25, 8)):
            noisy.putpixel(xy, 255)  # isolated salt pixels
        for choice in ("median", "4"):
            with self.subTest(choice=choice):
                result = self._run(choice, integer=2, image=noisy)
                self.assertEqual(result.details["Radius"], 2)
                self.assertTrue(np.all(np.asarray(result.image) == 100))  # not stretched to 0..255

    def test_unsharp_mask_increases_edge_contrast(self) -> None:
        step = np.zeros((20, 40), dtype=np.uint8)
        step[:, :20], step[:, 20:] = 80, 170
        image = Image.fromarray(step, mode="L")
        for choice in ("unsharp_mask", "6", "unsharp"):
            with self.subTest(choice=choice):
                result = self._run(choice, floats=[2.0, 1.5], image=image)
                arr = np.asarray(result.image).astype(int)
                self.assertEqual(result.details["Radius"], 2.0)
                self.assertEqual(result.details["Amount"], 1.5)
                self.assertLess(arr[10, 18], 80)  # dark side undershoots
                self.assertGreater(arr[10, 21], 170)  # bright side overshoots
                self.assertEqual(arr[10, 2], 80)  # flat areas keep their level (no stretching)

    def test_unsharp_mask_cancel_at_either_prompt(self) -> None:
        self.assertIsNone(self._run("unsharp_mask", floats=[None]))
        self.assertIsNone(self._run("unsharp_mask", floats=[1.0, None]))

    def test_median_cancel_returns_none(self) -> None:
        self.assertIsNone(self._run("median", integer=None))

    def test_otsu_gives_binary_image_split_between_two_levels(self) -> None:
        arr = np.full((20, 40), 60, dtype=np.uint8)
        arr[:, 30:] = 200  # 25% bright
        for choice in ("threshold_otsu", "7", "otsu"):
            with self.subTest(choice=choice):
                result = self._run(choice, image=Image.fromarray(arr, mode="L"))
                out = np.asarray(result.image)
                self.assertEqual(set(np.unique(out)), {0, 255})
                self.assertTrue(np.all(out[:, 30:] == 255))
                self.assertTrue(np.all(out[:, :30] == 0))
                threshold = float(result.details["Otsu threshold"].split(" ")[0])
                self.assertTrue(60 <= threshold < 200)
                self.assertEqual(result.details["Pixels above"], "25.00%")


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
