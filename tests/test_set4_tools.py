from dataclasses import replace
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from skimage.feature import hog

from forensics_app.tools.hog import (
    CellInspector,
    HogSettings,
    HogTool,
    cell_histograms,
    comparison_figure,
    extract_hog,
    parse_orientations,
    render_hog,
    to_rgb_array,
)
from forensics_app.tools.hog_detection import (
    BOX_COLOURS,
    DetectionSettings,
    HogDetectionTool,
    detect,
    cosine_similarity,
    draw_detections,
    hog_features,
    match_strength,
    parse_scales,
    render_heatmap,
    resize_template,
)

HOG = "forensics_app.tools.hog"
HOG_DIALOG = f"{HOG}.HogDialog"
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


def _big_scene() -> Image.Image:
    """A flat scene holding the figure at twice the template's size, top-left at (20, 64)."""
    big = Image.fromarray(_figure(), mode="L").resize((48, 96), Image.Resampling.NEAREST)
    scene = Image.new("L", (160, 140), 30)
    scene.paste(big, (64, 20))
    return scene.convert("RGB")


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


def _stripes(vertical: bool = True) -> Image.Image:
    """Stripes 4 px wide: vertical stripes only have horizontal gradients (0°), horizontal ones only vertical (90°)."""
    arr = np.zeros((32, 32), dtype=np.uint8)
    if vertical:
        arr[:, (np.arange(32) // 4) % 2 == 1] = 255
    else:
        arr[(np.arange(32) // 4) % 2 == 1, :] = 255
    return Image.fromarray(arr, mode="L")


class CellHistogramTests(unittest.TestCase):
    def test_matches_skimage_per_cell_histograms(self) -> None:
        # skimage with 1×1 blocks and L1 normalization returns each cell's histogram divided by its sum
        image = _scene()
        for orientations, cell in ((9, 8), (4, 6), (16, 5)):
            ours = cell_histograms(image, cell, orientations)
            theirs = hog(
                to_rgb_array(image),
                orientations=orientations,
                pixels_per_cell=(cell, cell),
                cells_per_block=(1, 1),
                block_norm="L1",
                feature_vector=False,
                channel_axis=2,
            )[:, :, 0, 0, :]
            np.testing.assert_allclose(ours / (ours.sum(axis=2, keepdims=True) + 1e-5), theirs, atol=1e-6)

    def test_stripe_direction_lands_in_the_expected_bin(self) -> None:
        vertical = cell_histograms(_stripes(vertical=True), 8, 4)  # bins centred at 22.5, 67.5, 112.5, 157.5
        horizontal = cell_histograms(_stripes(vertical=False), 8, 4)
        self.assertTrue((vertical[1:-1, 1:-1, 0] > 0).all())  # 0° bin: horizontal gradient
        self.assertEqual(vertical[..., 1:].sum(), 0)
        self.assertTrue((horizontal[1:-1, 1:-1, 2] > 0).all())  # 90° falls in the [90, 135) bin
        self.assertEqual(horizontal[..., [0, 1, 3]].sum(), 0)

    def test_flat_image_has_empty_histograms(self) -> None:
        self.assertFalse(cell_histograms(Image.new("RGB", (16, 16), (90, 90, 90)), 4, 9).any())


class ComparisonTests(unittest.TestCase):
    def test_parse_orientations(self) -> None:
        self.assertEqual(parse_orientations("2, 4, 9"), (2, 4, 9))
        self.assertEqual(parse_orientations("9;2, 9"), (9, 2))
        for bad in ("", "a", "0", "2, 50", "1,2,3,4,5,6,7"):
            with self.assertRaises(ValueError, msg=bad):
                parse_orientations(bad)

    def test_figure_has_original_plus_one_panel_per_count(self) -> None:
        image = Image.fromarray(_figure(), mode="L")
        settings = HogSettings(pixels_per_cell=6, cells_per_block=2, compare_orientations=(2, 4, 9))
        figure, lengths = comparison_figure(image, settings)
        titles = [ax.get_title() for ax in figure.axes]
        self.assertEqual(titles[0], "Original image")
        self.assertEqual(len(titles), 4)
        self.assertTrue(titles[3].startswith("HOG, orientations=9"))
        self.assertEqual(lengths, {2: 3 * 7 * 2 * 2 * 2, 4: 3 * 7 * 2 * 2 * 4, 9: 3 * 7 * 2 * 2 * 9})

    def test_tool_compare_opens_popup_and_keeps_image(self) -> None:
        settings = HogSettings(pixels_per_cell=6, cells_per_block=2, display="compare")
        with mock.patch(HOG_DIALOG, **{"return_value.result": settings}), mock.patch(f"{HOG}.show_figure") as show:
            result = HogTool().run(mock.Mock(), _document(Image.fromarray(_figure(), mode="L")))
        show.assert_called_once()
        self.assertIsNone(result.image)
        self.assertIn("Features (orientations=4)", result.details)


class CellInspectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = HogSettings(orientations=4, pixels_per_cell=8, cells_per_block=2)
        self.inspector = CellInspector(_stripes(vertical=True), self.settings)

    def test_starts_on_the_strongest_cell(self) -> None:
        totals = self.inspector.hist.sum(axis=2)
        self.assertEqual(totals[self.inspector.selected], totals.max())

    def test_click_selects_the_cell_and_plots_its_histogram(self) -> None:
        ax = self.inspector.ax_image
        event = mock.Mock(inaxes=ax, xdata=20.0, ydata=13.0)  # x 20 -> col 2, y 13 -> row 1
        with mock.patch.object(self.inspector.figure, "canvas"):
            self.inspector.on_click(event)
        self.assertEqual(self.inspector.selected, (1, 2))
        self.assertEqual(self.inspector.highlight.get_xy(), (15.5, 7.5))
        heights = [bar.get_height() for bar in self.inspector.ax_hist.patches]
        np.testing.assert_allclose(heights, self.inspector.hist[1, 2])
        self.assertIn("dominant bin: 22.5°", self.inspector.ax_hist.get_title())

    def test_clicks_outside_the_image_are_ignored(self) -> None:
        before = self.inspector.selected
        self.inspector.on_click(mock.Mock(inaxes=self.inspector.ax_hist, xdata=1.0, ydata=1.0))
        self.inspector.on_click(mock.Mock(inaxes=self.inspector.ax_image, xdata=None, ydata=None))
        self.assertEqual(self.inspector.selected, before)
        self.assertIsNone(self.inspector.cell_at(40.0, 3.0))  # past the last column of cells

    def test_tool_inspect_keeps_image(self) -> None:
        settings = HogSettings(orientations=4, pixels_per_cell=8, cells_per_block=2, display="inspect")
        with mock.patch(HOG_DIALOG, **{"return_value.result": settings}), mock.patch(f"{HOG}.show_figure") as show:
            result = HogTool().run(mock.Mock(), _document(_stripes()))
        self.assertIsNone(result.image)
        self.assertIsInstance(show.return_value.inspector, CellInspector)


class DetectionFunctionTests(unittest.TestCase):
    def test_all_metrics_find_the_figure(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        settings = DetectionSettings(orientations=9, pixels_per_cell=4, cells_per_block=2, step=4)
        detections, score_maps = detect(_scene(top=40, left=56), template, settings)
        self.assertEqual(len(score_maps), 1)
        self.assertEqual(score_maps[0].windows, len(range(0, 120 - 48 + 1, 4)) * len(range(0, 140 - 24 + 1, 4)))
        self.assertEqual([d.method for d in detections], ["euclidean", "dot", "cosine"])
        for detection in detections:
            self.assertEqual((detection.row, detection.col), (40, 56))
            self.assertEqual((detection.height, detection.width), (48, 24))
        self.assertAlmostEqual(detections[2].score, 1.0)  # exact copy of the template

    def test_selected_metrics_only(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        settings = DetectionSettings(pixels_per_cell=4, step=8, metrics=("dot", "cosine"))
        detections, _ = detect(_scene(), template, settings)
        self.assertEqual([d.method for d in detections], ["dot", "cosine"])

    def test_cosine_similarity_ignores_magnitude(self) -> None:
        a = np.array([1.0, 2.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, 5 * a), 1.0)
        self.assertAlmostEqual(cosine_similarity(a, np.array([0.0, 0.0, 3.0])), 0.0)
        self.assertEqual(cosine_similarity(a, np.zeros(3)), 0.0)  # flat window: no division by zero

    def test_dot_product_prefers_strong_texture_but_cosine_does_not(self) -> None:
        template = np.array([1.0, 1.0, 0.0, 0.0])
        same_shape = 0.5 * template  # same pattern, weaker gradients
        busy = np.array([1.0, 1.0, 1.0, 1.0])  # gradients everywhere
        self.assertGreater(np.dot(busy, template), np.dot(same_shape, template))
        self.assertGreater(cosine_similarity(same_shape, template), cosine_similarity(busy, template))

    def test_resize_template_to_window(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        self.assertIs(resize_template(template, DetectionSettings()), template)  # no size given: unchanged
        resized = resize_template(template, DetectionSettings(window_width=48, window_height=96))
        self.assertEqual(resized.size, (48, 96))

    def test_larger_window_finds_a_larger_figure(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        settings = DetectionSettings(pixels_per_cell=8, step=4, window_width=48, window_height=96)
        detections, _ = detect(_big_scene(), template, settings)
        for detection in detections:
            self.assertEqual((detection.row, detection.col), (20, 64))
            self.assertEqual((detection.height, detection.width), (96, 48))

    def test_multi_scale_picks_the_right_scale(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        settings = DetectionSettings(pixels_per_cell=8, step=4, scales=(1.0, 1.5, 2.0))
        detections, score_maps = detect(_big_scene(), template, settings)
        self.assertEqual([m.scale for m in score_maps], [1.0, 1.5, 2.0])
        for detection in detections:
            self.assertEqual(detection.scale, 2.0, detection.method)
            self.assertEqual((detection.row, detection.col), (20, 64))

    def test_scales_that_do_not_fit_are_skipped(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        _, score_maps = detect(_scene(), template, DetectionSettings(pixels_per_cell=4, step=8, scales=(1.0, 5.0)))
        self.assertEqual([m.scale for m in score_maps], [1.0])
        with self.assertRaises(ValueError):
            detect(_scene(), template, DetectionSettings(scales=(5.0,)))

    def test_parse_scales(self) -> None:
        self.assertEqual(parse_scales("1.25, 0.75,1, 1"), (0.75, 1.0, 1.25))
        self.assertEqual(parse_scales("2"), (2.0,))
        for bad in ("", "a, 1", "0", "50"):
            with self.assertRaises(ValueError, msg=bad):
                parse_scales(bad)

    def test_window_larger_than_image_is_rejected(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        with self.assertRaises(ValueError):
            detect(_scene(), template, DetectionSettings(window_width=100, window_height=200))

    def test_template_larger_than_image_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            detect(Image.new("RGB", (20, 20)), Image.new("RGB", (32, 64)), DetectionSettings())

    def test_draw_detections_outlines_box(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        settings = DetectionSettings(pixels_per_cell=4, step=4, metrics=("euclidean",))
        detections, _ = detect(_scene(), template, settings)
        out = np.asarray(draw_detections(_scene(), detections))
        self.assertEqual(tuple(out[40, 56]), BOX_COLOURS["euclidean"])  # top-left corner
        self.assertEqual(tuple(out[40 + 47, 56 + 23]), BOX_COLOURS["euclidean"])  # bottom-right corner


class HeatmapTests(unittest.TestCase):
    def setUp(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        self.settings = DetectionSettings(pixels_per_cell=4, step=4)
        _, (self.score_map,) = detect(_scene(), template, self.settings)

    def test_match_strength_is_one_at_the_best_window(self) -> None:
        for metric in ("euclidean", "dot", "cosine"):
            strength = match_strength(self.score_map, metric)
            self.assertAlmostEqual(strength.max(), 1.0)
            self.assertAlmostEqual(strength.min(), 0.0)
            a, b = np.unravel_index(np.argmax(strength), strength.shape)
            self.assertEqual((self.score_map.rows[a], self.score_map.cols[b]), (40, 56), metric)

    def test_heatmap_is_brightest_at_the_figure_centre(self) -> None:
        heat = np.asarray(render_heatmap(_scene(), self.score_map, "cosine", self.settings.step).convert("L"))
        self.assertEqual(heat.shape, (120, 140))
        self.assertEqual(heat[40 + 24, 56 + 12], heat.max())

    def test_tool_heatmap_display(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        settings = DetectionSettings(pixels_per_cell=4, step=4, display="heatmap", heatmap_metric="dot", metrics=())
        with (
            mock.patch(f"{DET}.filedialog.askopenfilename", return_value="person.png"),
            mock.patch(f"{DET}.Image.open") as opened,
            mock.patch(f"{DET}.VariantDialog", **{"return_value.result": settings.variant}),
            mock.patch(f"{DET}.DetectionDialog") as dialog,
        ):
            opened.return_value.__enter__.return_value = template
            dialog.return_value.result = settings
            result = HogDetectionTool().run(mock.Mock(), _document(_scene()))
        self.assertEqual(result.details["Heatmap metric"], "Dot product (scale 1)")
        self.assertNotIn("Dot product box (x, y, w, h)", result.details)  # heatmap only, no boxes


class DetectionToolTests(unittest.TestCase):
    def test_run_draws_boxes_and_reports_location(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        settings = DetectionSettings(orientations=9, pixels_per_cell=4, cells_per_block=2, step=4)
        with (
            mock.patch(f"{DET}.filedialog.askopenfilename", return_value="person.png"),
            mock.patch(f"{DET}.Image.open") as opened,
            mock.patch(f"{DET}.VariantDialog", **{"return_value.result": settings.variant}),
            mock.patch(f"{DET}.DetectionDialog") as dialog,
        ):
            opened.return_value.__enter__.return_value = template
            dialog.return_value.result = settings
            result = HogDetectionTool().run(mock.Mock(), _document(_scene()))
        self.assertEqual(result.image.mode, "RGB")
        for name in ("Euclidean", "Dot product", "Cosine"):
            self.assertEqual(result.details[f"{name} box (x, y, w, h)"], "(56, 40, 24, 48)")
        self.assertEqual(result.details["Window sizes"], "24 × 48")
        self.assertEqual(result.details["Scales searched"], "1")

    def test_cancel_variant_returns_none(self) -> None:
        with (
            mock.patch(f"{DET}.VariantDialog", **{"return_value.result": None}),
            mock.patch(f"{DET}.filedialog.askopenfilename") as ask,
        ):
            self.assertIsNone(HogDetectionTool().run(mock.Mock(), _document(_scene())))
        ask.assert_not_called()  # the variant is chosen before the template

    def test_cancel_template_returns_none(self) -> None:
        with (
            mock.patch(f"{DET}.VariantDialog", **{"return_value.result": "exact"}),
            mock.patch(f"{DET}.filedialog.askopenfilename", return_value=""),
        ):
            self.assertIsNone(HogDetectionTool().run(mock.Mock(), _document(_scene())))

    def test_chosen_variant_is_passed_to_the_options_dialog(self) -> None:
        with (
            mock.patch(f"{DET}.VariantDialog", **{"return_value.result": "fast"}),
            mock.patch(f"{DET}.filedialog.askopenfilename", return_value="person.png"),
            mock.patch(f"{DET}.Image.open"),
            mock.patch(f"{DET}.DetectionDialog", **{"return_value.result": None}) as dialog,
        ):
            self.assertIsNone(HogDetectionTool().run(mock.Mock(), _document(_scene())))
        self.assertEqual(dialog.call_args.args[3], "fast")


class FastVariantTests(unittest.TestCase):
    def test_fast_finds_the_same_figure(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        settings = DetectionSettings(pixels_per_cell=4, step=4, variant="fast")
        detections, (score_map,) = detect(_scene(top=40, left=56), template, settings)
        for detection in detections:
            self.assertEqual((detection.row, detection.col), (40, 56), detection.method)
        exact, (exact_map,) = detect(_scene(top=40, left=56), template, replace(settings, variant="exact"))
        self.assertEqual(score_map.windows, exact_map.windows)
        # Only the border gradients differ, so the scores are close but not identical
        diff = np.abs(score_map.scores["cosine"] - exact_map.scores["cosine"])
        self.assertLess(diff.mean(), 0.05)

    def test_inner_blocks_match_the_exact_hog(self) -> None:
        image = _scene()
        settings = DetectionSettings(orientations=9, pixels_per_cell=4, cells_per_block=2)
        rgb = to_rgb_array(image)
        blocks = hog_features(rgb, settings, feature_vector=False)
        window = hog_features(rgb[40:88, 56:80], settings, feature_vector=False)  # 12 x 6 cells -> 11 x 5 blocks
        sliced = blocks[10:21, 14:19]
        # Blocks that do not touch the window border are identical...
        np.testing.assert_allclose(window[1:-1, 1:-1], sliced[1:-1, 1:-1])
        # ...border blocks differ (skimage zeroes the gradient on the crop's outer pixels)
        self.assertFalse(np.allclose(window, sliced))

    def test_fast_rejects_step_off_the_cell_grid(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        with self.assertRaises(ValueError):
            detect(_scene(), template, DetectionSettings(pixels_per_cell=8, step=4, variant="fast"))

    def test_fast_multi_scale_computes_the_image_hog_once(self) -> None:
        template = Image.fromarray(_figure(), mode="L")
        settings = DetectionSettings(pixels_per_cell=8, step=8, scales=(1.0, 1.5, 2.0), variant="fast")
        with mock.patch(f"{DET}.hog_features", wraps=hog_features) as spy:
            detections, score_maps = detect(_big_scene(), template, settings)
        image_calls = [c for c in spy.call_args_list if c.kwargs.get("feature_vector") is False]
        self.assertEqual(len(image_calls), 1)
        self.assertEqual(len(score_maps), 3)
        for detection in detections:
            self.assertEqual(detection.scale, 2.0, detection.method)
            # y = 20 is off the 8-px grid: the best the fast variant can do is 16 or 24 (both 4 px away)
            self.assertIn(detection.row, (16, 24), detection.method)
            self.assertEqual(detection.col, 64)


if __name__ == "__main__":
    unittest.main()
