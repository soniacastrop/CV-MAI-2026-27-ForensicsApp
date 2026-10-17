import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.hog import HogSettings, HogTool, extract_hog, render_hog
from forensics_app.tools.hog_detection import (
    BOX_COLOURS,
    DetectionSettings,
    HogDetectionTool,
    detect,
    draw_detections,
    resize_template,
)

HOG_DIALOG = "forensics_app.tools.hog.HogDialog"
DET = "forensics_app.tools.hog_detection"


def _document(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    return document


def _figure(height: int = 48, width: int = 24) -> np.ndarray:
    """A bright upright figure with a 'head' on a dark background: a toy person template."""
    arr = np.full((height, width), 30, dtype=np.uint8)
    arr[4:12, width // 2 - 4 : width // 2 + 4] = 220  # head
    arr[14:44, width // 2 - 6 : width // 2 + 6] = 200  # body
    return arr


def _scene(top: int = 40, left: int = 56) -> Image.Image:
    """A noisy dark scene with the figure pasted at (top, left)."""
    rng = np.random.default_rng(0)
    arr = rng.integers(20, 60, size=(120, 140)).astype(np.uint8)
    figure = _figure()
    arr[top : top + figure.shape[0], left : left + figure.shape[1]] = figure
    return Image.fromarray(arr, mode="L").convert("RGB")


class HogFunctionTests(unittest.TestCase):
    def test_feature_length_follows_the_settings(self) -> None:
        image = Image.fromarray(_figure(), mode="L")  # 24 wide x 48 tall
        features, hog_image = extract_hog(image, HogSettings(orientations=9, pixels_per_cell=8, cells_per_block=2))
        # 3 x 6 cells -> 2 x 5 blocks of 2 x 2 cells x 9 bins
        self.assertEqual(features.size, 2 * 5 * 2 * 2 * 9)
        self.assertEqual(hog_image.shape, (48, 24))

    def test_more_orientations_give_more_features(self) -> None:
        image = Image.fromarray(_figure(), mode="L")
        few, _ = extract_hog(image, HogSettings(orientations=2, pixels_per_cell=6, cells_per_block=3))
        many, _ = extract_hog(image, HogSettings(orientations=9, pixels_per_cell=6, cells_per_block=3))
        self.assertEqual(many.size * 2, few.size * 9)

    def test_flat_image_has_no_gradients(self) -> None:
        features, hog_image = extract_hog(Image.new("L", (32, 32), 128), HogSettings())
        self.assertFalse(features.any())
        self.assertFalse(np.asarray(render_hog(Image.new("L", (32, 32)), hog_image, "hog")).any())

    def test_too_small_image_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_hog(Image.new("RGB", (10, 10)), HogSettings(pixels_per_cell=8, cells_per_block=3))

    def test_render_modes(self) -> None:
        image = Image.fromarray(_figure(), mode="L")
        _, hog_image = extract_hog(image, HogSettings(pixels_per_cell=4, cells_per_block=2))
        hog_out = render_hog(image, hog_image, "hog")
        self.assertEqual((hog_out.mode, hog_out.size), ("L", image.size))
        self.assertEqual(np.asarray(hog_out).max(), 255)  # stretched to full range
        overlay = render_hog(image, hog_image, "overlay")
        self.assertEqual(overlay.mode, "RGB")
        r, g, _b = (np.asarray(c).astype(int) for c in overlay.split())
        self.assertTrue((r > g).any())  # gradients drawn in red
        with self.assertRaises(ValueError):
            render_hog(image, hog_image, "nope")


class HogToolTests(unittest.TestCase):
    def test_run_returns_visualization_and_details(self) -> None:
        image = Image.fromarray(_figure(), mode="L")
        with mock.patch(HOG_DIALOG) as dialog:
            dialog.return_value.result = HogSettings(orientations=4, pixels_per_cell=6, cells_per_block=2)
            result = HogTool().run(mock.Mock(), _document(image))
        self.assertEqual(result.image.size, image.size)
        self.assertEqual(result.details["Orientations"], 4)
        self.assertEqual(result.details["Cells"], "4 × 8")
        self.assertEqual(result.details["Feature length"], 3 * 7 * 2 * 2 * 4)

    def test_cancel_returns_none(self) -> None:
        with mock.patch(HOG_DIALOG) as dialog:
            dialog.return_value.result = None
            self.assertIsNone(HogTool().run(mock.Mock(), _document(Image.new("L", (32, 32)))))


class DetectionFunctionTests(unittest.TestCase):
    def test_both_measures_find_the_figure(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        settings = DetectionSettings(orientations=9, pixels_per_cell=4, cells_per_block=2, step=4, method="both")
        detections, windows = detect(_scene(top=40, left=56), template, settings)
        self.assertEqual(windows, len(range(0, 120 - 48 + 1, 4)) * len(range(0, 140 - 24 + 1, 4)))
        self.assertEqual([d.method for d in detections], ["euclidean", "dot"])
        for detection in detections:
            self.assertEqual((detection.row, detection.col), (40, 56))
            self.assertEqual((detection.height, detection.width), (48, 24))

    def test_single_method(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        detections, _ = detect(_scene(), template, DetectionSettings(pixels_per_cell=4, step=8, method="dot"))
        self.assertEqual([d.method for d in detections], ["dot"])

    def test_resize_template_to_window(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        self.assertIs(resize_template(template, DetectionSettings()), template)  # no size given: unchanged
        resized = resize_template(template, DetectionSettings(window_width=48, window_height=96))
        self.assertEqual(resized.size, (48, 96))

    def test_larger_window_finds_a_larger_figure(self) -> None:
        # The scene holds the figure at twice the template's size
        big = Image.fromarray(_figure(), mode="L").resize((48, 96), Image.Resampling.NEAREST)
        scene = Image.new("L", (160, 140), 30)
        scene.paste(big, (64, 20))
        template = Image.fromarray(_figure(), mode="L")
        settings = DetectionSettings(pixels_per_cell=8, step=4, window_width=48, window_height=96)
        detections, _ = detect(scene.convert("RGB"), template, settings)
        for detection in detections:
            self.assertEqual((detection.row, detection.col), (20, 64))
            self.assertEqual((detection.height, detection.width), (96, 48))

    def test_window_larger_than_image_is_rejected(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        with self.assertRaises(ValueError):
            detect(_scene(), template, DetectionSettings(window_width=100, window_height=200))

    def test_template_larger_than_image_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            detect(Image.new("RGB", (20, 20)), Image.new("RGB", (32, 64)), DetectionSettings())

    def test_draw_detections_outlines_box(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        detections, _ = detect(_scene(), template, DetectionSettings(pixels_per_cell=4, step=4, method="euclidean"))
        out = np.asarray(draw_detections(_scene(), detections))
        self.assertEqual(tuple(out[40, 56]), BOX_COLOURS["euclidean"])  # top-left corner
        self.assertEqual(tuple(out[40 + 47, 56 + 23]), BOX_COLOURS["euclidean"])  # bottom-right corner


class DetectionToolTests(unittest.TestCase):
    def test_run_draws_boxes_and_reports_location(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        settings = DetectionSettings(orientations=9, pixels_per_cell=4, cells_per_block=2, step=4)
        with (
            mock.patch(f"{DET}.filedialog.askopenfilename", return_value="person.png"),
            mock.patch(f"{DET}.Image.open") as opened,
            mock.patch(f"{DET}.DetectionDialog") as dialog,
        ):
            opened.return_value.__enter__.return_value = template
            dialog.return_value.result = settings
            result = HogDetectionTool().run(mock.Mock(), _document(_scene()))
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.details["Euclidean box (x, y)"], "(56, 40)")
        self.assertEqual(result.details["Dot product box (x, y)"], "(56, 40)")
        self.assertEqual(result.details["Window size"], "24 × 48")
        self.assertEqual(result.details["Scale"], "1.00 × 1.00")

    def test_cancel_template_returns_none(self) -> None:
        with mock.patch(f"{DET}.filedialog.askopenfilename", return_value=""):
            self.assertIsNone(HogDetectionTool().run(mock.Mock(), _document(_scene())))


if __name__ == "__main__":
    unittest.main()
