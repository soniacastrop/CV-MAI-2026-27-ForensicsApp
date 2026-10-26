from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from forensics_app.core import ImageDocument
from forensics_app.tools.corner_detection import (
    CORNER_COLOUR,
    CORNER_DISPLAYS,
    CornerDetectionSettings,
    CornerDetectionTool,
    compute_corner_response,
    corner_comparison_figure,
    detect_corner_peaks,
    detect_corners,
    render_corners,
    response_to_image,
)
from forensics_app.tools.descriptors import (
    DESCRIPTOR_DISPLAYS,
    DescriptorExtractorTool,
    DescriptorSettings,
    descriptor_comparison_figure,
    extract_descriptors,
    render_keypoints,
)
from forensics_app.tools.feature_matching import (
    FeatureMatchingTool,
    MatchingSettings,
    draw_matches,
    extract_features,
    match_features,
    prepare_gray,
    ransac_inliers,
)
from forensics_app.tools.image_lookup import (
    ImageLookupTool,
    LookupSettings,
    list_images,
    rank_images,
    score_matches,
)

CD = "forensics_app.tools.corner_detection"
DESC = "forensics_app.tools.descriptors"
FM = "forensics_app.tools.feature_matching"
LOOKUP = "forensics_app.tools.image_lookup"
SHIFT = (12, 20)  # (rows, cols) the second image is shifted by


def _texture(seed: int = 0, size: int = 220) -> np.ndarray:
    """Blocky random texture: plenty of corners for ORB and SIFT."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(size // 10, size // 10)).astype(np.float64)
    return np.clip(zoom(blocks, 10, order=0), 0, 255).astype(np.uint8)


def _corner_scene(seed: int = 0, size: int = 160) -> Image.Image:
    """Fine blocky texture with small features suitable for both FAST (n=12) and Harris."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(size // 4, size // 4)).astype(np.float64)
    arr = np.clip(zoom(blocks, 4, order=0), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _pair() -> tuple[Image.Image, Image.Image]:
    """Two overlapping crops of one texture: content at (r, c) in the first is at (r - 12, c - 20) in the second."""
    texture = _texture()
    first = texture[:180, :180]
    second = texture[SHIFT[0] : SHIFT[0] + 180, SHIFT[1] : SHIFT[1] + 180]
    return Image.fromarray(first, mode="L"), Image.fromarray(second, mode="L").convert("RGB")


def _document(image: Image.Image, path: Path | None = None) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    document.path = path
    return document


def _parent() -> mock.Mock:
    parent = mock.Mock()
    parent.winfo_toplevel.return_value.title.return_value = "Forensics"
    return parent


class CornerDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = _corner_scene()

    def test_fast_and_harris_detect_corners_on_textured_image(self) -> None:
        for detector in ("fast", "harris"):
            with self.subTest(detector=detector):
                settings = CornerDetectionSettings(detector=detector, threshold_rel=0.1)
                response, corners = detect_corners(self.image, settings)
                self.assertEqual(response.shape, (160, 160))
                self.assertGreater(len(corners), 5)
                self.assertEqual(corners.shape[1], 2)

    def test_higher_peak_threshold_reduces_detected_corners(self) -> None:
        for detector in ("fast", "harris"):
            with self.subTest(detector=detector):
                response = compute_corner_response(self.image, CornerDetectionSettings(detector=detector))
                low = detect_corner_peaks(response, CornerDetectionSettings(detector=detector, threshold_rel=0.05))
                high = detect_corner_peaks(response, CornerDetectionSettings(detector=detector, threshold_rel=0.5))
                self.assertGreater(len(low), len(high))

    def test_flat_image_has_no_corners(self) -> None:
        flat = Image.new("L", (80, 80), 120)
        for detector in ("fast", "harris"):
            with self.subTest(detector=detector):
                response, corners = detect_corners(flat, CornerDetectionSettings(detector=detector))
                self.assertEqual(len(corners), 0)
                self.assertEqual(np.asarray(response_to_image(response)).max(), 0)

    def test_harris_has_no_artificial_corners_on_the_image_frame(self) -> None:
        # A bright image with one dark square: the only true corners are the square's four corners
        arr = np.full((60, 60), 200, dtype=np.uint8)
        arr[20:40, 20:40] = 30
        _response, corners = detect_corners(
            Image.fromarray(arr, mode="L"), CornerDetectionSettings(detector="harris", threshold_rel=0.1)
        )
        self.assertEqual(len(corners), 4)
        self.assertTrue(((corners >= 17) & (corners <= 42)).all(), corners)

    def test_invalid_parameters_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            compute_corner_response(self.image, CornerDetectionSettings(detector="moravec"))
        with self.assertRaises(ValueError):
            compute_corner_response(self.image, CornerDetectionSettings(detector="fast", fast_n=20))
        with self.assertRaises(ValueError):
            compute_corner_response(self.image, CornerDetectionSettings(detector="harris", harris_k=0.5))
        response = np.zeros((20, 20))
        with self.assertRaises(ValueError):
            detect_corner_peaks(response, CornerDetectionSettings(threshold_rel=1.5))
        with self.assertRaises(ValueError):
            detect_corner_peaks(response, CornerDetectionSettings(min_distance=0))

    def test_render_modes_and_comparison_figure(self) -> None:
        settings = CornerDetectionSettings(detector="harris", threshold_rel=0.2)
        response, corners = detect_corners(self.image, settings)
        for display in CORNER_DISPLAYS:
            if display == "popup":
                continue
            rendered = render_corners(self.image, response, corners, display)
            self.assertEqual(rendered.size, self.image.size)
        peaks_img = np.asarray(render_corners(self.image, response, corners, "peaks_gray"))
        self.assertTrue((peaks_img[..., 0] > peaks_img[..., 1]).any())  # red circles drawn
        with self.assertRaises(ValueError):
            render_corners(self.image, response, corners, "unknown")

        fig = corner_comparison_figure(self.image, response, corners, settings)
        self.assertEqual(len(fig.axes), 2)
        self.assertIn("Harris", fig.axes[0].get_title())
        self.assertIn("threshold_rel=0.2", fig.axes[1].get_title())

    def test_corner_tool_runs_and_handles_popup_and_cancel(self) -> None:
        settings = CornerDetectionSettings(detector="fast", threshold_rel=0.15, display="peaks_gray")
        with mock.patch(f"{CD}.CornerDetectionDialog", **{"return_value.result": settings}):
            result = CornerDetectionTool().run(mock.Mock(), _document(self.image))
        self.assertIsNotNone(result.image)
        self.assertEqual(result.image.size, self.image.size)
        self.assertGreater(result.details["Detected corners"], 0)
        self.assertIn("corner_fast", result.details["Library"])

        popup_settings = CornerDetectionSettings(detector="harris", threshold_rel=0.2, display="popup")
        with (
            mock.patch(f"{CD}.CornerDetectionDialog", **{"return_value.result": popup_settings}),
            mock.patch(f"{CD}.show_figure") as show,
        ):
            popup_res = CornerDetectionTool().run(mock.Mock(), _document(self.image))
        show.assert_called_once()
        self.assertIsNone(popup_res.image)
        self.assertIn("corner_harris", popup_res.details["Library"])

        with mock.patch(f"{CD}.CornerDetectionDialog", **{"return_value.result": None}):
            self.assertIsNone(CornerDetectionTool().run(mock.Mock(), _document(self.image)))


class DescriptorExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = _pair()[1]  # RGB image

    def test_orb_and_sift_configurable_extraction(self) -> None:
        orb_settings = DescriptorSettings(descriptor="orb", n_keypoints=40, fast_threshold=0.1)
        orb_features = extract_descriptors(self.image, orb_settings)
        self.assertGreater(len(orb_features.keypoints), 0)
        self.assertLessEqual(len(orb_features.keypoints), 40)
        self.assertEqual(orb_features.descriptors.shape, (len(orb_features.keypoints), 256))

        sift_settings = DescriptorSettings(descriptor="sift", n_octaves=3, n_scales=3, sigma_min=1.6)
        sift_features = extract_descriptors(self.image, sift_settings)
        self.assertGreater(len(sift_features.keypoints), 0)
        self.assertEqual(sift_features.descriptors.shape[1], 128)

    def test_render_keypoints_and_figure(self) -> None:
        settings = DescriptorSettings(descriptor="orb", n_keypoints=30)
        features = extract_descriptors(self.image, settings)
        for display in DESCRIPTOR_DISPLAYS:
            if display == "popup":
                continue
            rendered = render_keypoints(self.image, features, display)
            self.assertEqual(rendered.size, self.image.size)
            self.assertEqual(rendered.mode, "RGB")
        arr = np.asarray(render_keypoints(self.image, features, "keypoints_gray"))
        self.assertTrue((arr[..., 0] > arr[..., 1]).any())  # red circles drawn
        with self.assertRaises(ValueError):
            render_keypoints(self.image, features, "bad_mode")

        fig = descriptor_comparison_figure(features, settings)
        self.assertEqual(len(fig.axes), 2)
        self.assertIn("ORB keypoints", fig.axes[0].get_title())
        self.assertIn("ORB descriptors", fig.axes[1].get_title())

    def test_descriptor_tool_runs_and_handles_popup_and_cancel(self) -> None:
        settings = DescriptorSettings(descriptor="sift", n_octaves=3, n_scales=3, sigma_min=1.6)
        with mock.patch(f"{DESC}.DescriptorDialog", **{"return_value.result": settings}):
            result = DescriptorExtractorTool().run(mock.Mock(), _document(self.image))
        self.assertIsNotNone(result.image)
        self.assertEqual(result.image.size, self.image.size)
        self.assertEqual(result.details["Descriptor"], "SIFT")
        self.assertEqual(result.details["Library"], "skimage.feature.SIFT")
        self.assertGreater(result.details["Keypoints detected"], 0)

        popup_settings = DescriptorSettings(descriptor="orb", n_keypoints=25, display="popup")
        with (
            mock.patch(f"{DESC}.DescriptorDialog", **{"return_value.result": popup_settings}),
            mock.patch(f"{DESC}.show_figure") as show,
        ):
            popup_res = DescriptorExtractorTool().run(mock.Mock(), _document(self.image))
        show.assert_called_once()
        self.assertIsNone(popup_res.image)
        self.assertEqual(popup_res.details["Descriptor"], "ORB")

        with mock.patch(f"{DESC}.DescriptorDialog", **{"return_value.result": None}):
            self.assertIsNone(DescriptorExtractorTool().run(mock.Mock(), _document(self.image)))


class ExtractionTests(unittest.TestCase):
    def test_orb_respects_n_keypoints_and_gives_binary_descriptors(self) -> None:
        features = extract_features(_pair()[0], MatchingSettings(descriptor="orb", n_keypoints=50))
        self.assertGreater(len(features.keypoints), 0)
        self.assertLessEqual(len(features.keypoints), 50)
        self.assertEqual(features.descriptors.shape, (len(features.keypoints), 256))
        self.assertEqual(features.descriptors.dtype, bool)

    def test_sift_gives_128_dimensional_descriptors(self) -> None:
        features = extract_features(_pair()[0], MatchingSettings(descriptor="sift", n_octaves=3))
        self.assertGreater(len(features.keypoints), 0)
        self.assertEqual(features.descriptors.shape[1], 128)

    def test_flat_image_has_no_features_and_no_matches(self) -> None:
        flat = Image.new("L", (120, 120), 128)
        for descriptor in ("orb", "sift"):
            with self.subTest(descriptor=descriptor):
                settings = MatchingSettings(descriptor=descriptor)
                features = extract_features(flat, settings)
                self.assertEqual(len(features.keypoints), 0)
                other = extract_features(_pair()[0], settings)
                self.assertEqual(match_features(features, other, settings).shape, (0, 2))

    def test_prepare_gray_downscales_the_longest_side(self) -> None:
        gray, scale = prepare_gray(Image.new("RGBA", (400, 200), (255, 0, 0, 0)), max_side=100)
        self.assertEqual(gray.shape, (50, 100))
        self.assertAlmostEqual(scale, 0.25)
        self.assertLessEqual(gray.max(), 1.0)
        _gray, scale = prepare_gray(Image.new("L", (80, 60)), max_side=100)
        self.assertEqual(scale, 1.0)  # never upscaled

    def test_unknown_descriptor_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_features(_pair()[0], MatchingSettings(descriptor="surf"))


class MatchingTests(unittest.TestCase):
    def test_matches_follow_the_shift(self) -> None:
        first, second = _pair()
        for descriptor in ("orb", "sift"):
            with self.subTest(descriptor=descriptor):
                settings = MatchingSettings(descriptor=descriptor, use_ransac=True)
                f1, f2 = extract_features(first, settings), extract_features(second, settings)
                matches = match_features(f1, f2, settings)
                self.assertGreaterEqual(len(matches), 10)
                offsets = f1.keypoints[matches[:, 0]] - f2.keypoints[matches[:, 1]]
                np.testing.assert_allclose(np.median(offsets, axis=0), SHIFT, atol=1.5)

    def test_lower_ratio_keeps_fewer_matches(self) -> None:
        first, second = _pair()
        loose, strict = MatchingSettings(max_ratio=1.0), MatchingSettings(max_ratio=0.5)
        f1, f2 = extract_features(first, loose), extract_features(second, loose)
        self.assertLessEqual(len(match_features(f1, f2, strict)), len(match_features(f1, f2, loose)))

    def test_ransac_rejects_outliers(self) -> None:
        rng = np.random.default_rng(1)
        keypoints1 = rng.uniform(0, 100, size=(30, 2))
        keypoints2 = keypoints1 + (5.0, -3.0)
        keypoints2[:6] = rng.uniform(0, 100, size=(6, 2))  # wrong matches
        matches = np.column_stack([np.arange(30), np.arange(30)])
        inliers = ransac_inliers(keypoints1, keypoints2, matches, residual_threshold=1.0)
        self.assertFalse(inliers[:6].any())
        self.assertTrue(inliers[6:].all())

    def test_ransac_needs_more_than_three_matches(self) -> None:
        keypoints = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        matches = np.column_stack([np.arange(3), np.arange(3)])
        self.assertFalse(ransac_inliers(keypoints, keypoints, matches, 1.0).any())

    def test_draw_matches_places_images_side_by_side(self) -> None:
        first, second = _pair()
        settings = MatchingSettings()
        f1 = extract_features(first, settings)
        f2 = extract_features(second.crop((0, 0, 100, 150)), settings)
        canvas = draw_matches(f1, f2, np.array([[0, 0]]), show_keypoints=False)
        self.assertEqual(canvas.size, (180 + 100, 180))
        self.assertEqual(canvas.mode, "RGB")
        # A match line starts at the first keypoint (drawn in the first line colour)
        row, col = np.round(f1.keypoints[0]).astype(int)
        pixels = np.asarray(canvas)[row - 3 : row + 4, col - 3 : col + 4].reshape(-1, 3)
        self.assertTrue(any(tuple(pixel) in {(31, 119, 180), CORNER_COLOUR} for pixel in pixels))


class FeatureMatchingToolTests(unittest.TestCase):
    def test_run_draws_matches_and_reports_counts(self) -> None:
        first, second = _pair()
        with (
            mock.patch(f"{FM}.filedialog.askopenfilename", return_value="second.png"),
            mock.patch(f"{FM}.Image.open") as opened,
            mock.patch(f"{FM}.MatchingDialog", **{"return_value.result": MatchingSettings(use_ransac=True)}),
        ):
            opened.return_value.__enter__.return_value = second
            result = FeatureMatchingTool().run(mock.Mock(), _document(first))
        self.assertEqual(result.image.size, (360, 180))
        self.assertGreaterEqual(result.details["Matches"], 10)
        self.assertEqual(result.details["Metric"], "hamming")
        self.assertIn("ransac", result.details["Library"])
        self.assertIn("second.png", result.message)

    def test_cancel_file_or_dialog_returns_none(self) -> None:
        with mock.patch(f"{FM}.filedialog.askopenfilename", return_value=""):
            self.assertIsNone(FeatureMatchingTool().run(mock.Mock(), _document(_pair()[0])))
        with (
            mock.patch(f"{FM}.filedialog.askopenfilename", return_value="second.png"),
            mock.patch(f"{FM}.Image.open"),
            mock.patch(f"{FM}.MatchingDialog", **{"return_value.result": None}),
        ):
            self.assertIsNone(FeatureMatchingTool().run(mock.Mock(), _document(_pair()[0])))


class LookupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        first, second = _pair()
        self.reference_path = self.folder / "reference.png"
        first.save(self.reference_path)
        (self.folder / "same_place").mkdir()
        second.save(self.folder / "same_place" / "shifted.jpg", quality=95)
        Image.fromarray(_texture(seed=7)[:180, :180], mode="L").save(self.folder / "other.png")
        Image.new("L", (100, 100), 90).save(self.folder / "flat.bmp")
        (self.folder / "notes.txt").write_text("not an image")
        (self.folder / ".DS_Store").write_bytes(b"\0")
        (self.folder / "__MACOSX").mkdir()
        (self.folder / "__MACOSX" / "._other.png").write_bytes(b"\0")
        (self.folder / "broken.png").write_bytes(b"not a png")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_list_images_skips_hidden_metadata_and_excluded_files(self) -> None:
        names = [p.relative_to(self.folder).as_posix() for p in list_images(self.folder, exclude=self.reference_path)]
        self.assertEqual(names, ["broken.png", "flat.bmp", "other.png", "same_place/shifted.jpg"])
        flat = [p.name for p in list_images(self.folder, recursive=False)]
        self.assertEqual(flat, ["broken.png", "flat.bmp", "other.png", "reference.png"])

    def test_rank_images_puts_the_same_scene_first(self) -> None:
        for score in ("matches", "normalized"):
            for use_ransac in (False, True):
                with self.subTest(score=score, ransac=use_ransac):
                    settings = LookupSettings(MatchingSettings(use_ransac=use_ransac), score)
                    reference = extract_features(_pair()[0], settings.matching)
                    progress = mock.Mock()
                    paths = list_images(self.folder, exclude=self.reference_path)
                    candidates, failed = rank_images(reference, paths, settings, progress)
                    self.assertEqual(candidates[0].path.name, "shifted.jpg")
                    self.assertEqual([p.name for p in failed], ["broken.png"])
                    self.assertEqual(progress.call_count, len(paths))
                    scores = [c.score for c in candidates]
                    self.assertEqual(scores, sorted(scores, reverse=True))

    def test_normalized_score(self) -> None:
        settings = MatchingSettings()
        many = extract_features(_pair()[0], settings)
        none = extract_features(Image.new("L", (64, 64)), settings)
        matches = np.zeros((5, 2), dtype=np.intp)
        self.assertEqual(score_matches("matches", matches, many, many), 5.0)
        self.assertAlmostEqual(score_matches("normalized", matches, many, many), 5 / len(many.keypoints))
        self.assertEqual(score_matches("normalized", matches, many, none), 0.0)
        with self.assertRaises(ValueError):
            score_matches("best", matches, many, many)

    def test_tool_finds_best_image_and_skips_the_loaded_file(self) -> None:
        parent = _parent()
        with (
            mock.patch(f"{LOOKUP}.filedialog.askdirectory", return_value=str(self.folder)),
            mock.patch(f"{LOOKUP}.LookupDialog", **{"return_value.result": LookupSettings()}),
        ):
            result = ImageLookupTool().run(parent, _document(_pair()[0], self.reference_path))
        self.assertEqual(result.details["Best match"], "same_place/shifted.jpg")
        self.assertEqual(result.details["Images searched"], 3)  # reference and broken file left out
        self.assertEqual(result.details["Unreadable files"], "broken.png")
        self.assertIn("#1 same_place/shifted.jpg", result.details)
        self.assertEqual(result.image.size, (360, 180))
        parent.winfo_toplevel.return_value.title.assert_called_with("Forensics")  # title restored

    def test_tool_empty_folder_and_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            with (
                mock.patch(f"{LOOKUP}.filedialog.askdirectory", return_value=empty),
                mock.patch(f"{LOOKUP}.LookupDialog", **{"return_value.result": LookupSettings()}),
            ):
                result = ImageLookupTool().run(_parent(), _document(_pair()[0]))
        self.assertIsNone(result.image)
        self.assertIn("No images", result.message)
        with mock.patch(f"{LOOKUP}.filedialog.askdirectory", return_value=""):
            self.assertIsNone(ImageLookupTool().run(_parent(), _document(_pair()[0])))


if __name__ == "__main__":
    unittest.main()
