from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from forensics_app.core import ImageDocument
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

FM = "forensics_app.tools.feature_matching"
LOOKUP = "forensics_app.tools.image_lookup"
SHIFT = (12, 20)  # (rows, cols) the second image is shifted by


def _texture(seed: int = 0, size: int = 220) -> np.ndarray:
    """Blocky random texture: plenty of corners for ORB and SIFT."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(size // 10, size // 10)).astype(np.float64)
    return np.clip(zoom(blocks, 10, order=0), 0, 255).astype(np.uint8)


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
        self.assertTrue(any(tuple(pixel) in {(31, 119, 180), (255, 0, 0)} for pixel in pixels))


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
