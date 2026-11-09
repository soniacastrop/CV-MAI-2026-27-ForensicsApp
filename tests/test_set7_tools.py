from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matplotlib
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from forensics_app.core import ImageDocument
from forensics_app.tools import bovw
from forensics_app.tools.bovw import (
    BOVW_MODEL_FILENAME,
    PATCH_SIZE,
    BoVWClassificationTool,
    ClassificationSettings,
    CodebookBuilderTool,
    CodebookSettings,
    HistogramSettings,
    VisualWordHistogramTool,
    assign_visual_words,
    build_codebook,
    build_histogram,
    classification_result_figure,
    crop_patch,
    discover_splits,
    find_bovw_model,
    fit_classifier,
    histogram_rows_figure,
    load_bovw_model,
    predict_image_probabilities,
    sample_keypoints_figure,
    save_bovw_model,
    select_test_images_from_model,
    visual_words_figure,
)

CLASSES = ("butterfly", "kangaroo", "dalmatian")


def _class_image(cls_index: int, instance: int, size: int = 96) -> Image.Image:
    """Synthetic textured image with a distinct spatial/frequency pattern per class so SIFT/ORB can separate them."""
    rng = np.random.default_rng(1000 * cls_index + instance)
    arr = np.full((size, size), 120, dtype=np.float64)
    if cls_index == 0:
        # Fine checkerboard blocks (high-frequency corners)
        blocks = rng.integers(20, 235, size=(size // 6, size // 6)).astype(np.float64)
        arr = zoom(blocks, 6, order=0)
    elif cls_index == 1:
        # Coarse blocks + strong diagonal ramp
        blocks = rng.integers(30, 220, size=(size // 12, size // 12)).astype(np.float64)
        arr = zoom(blocks, 12, order=1)
    else:
        # Dalmatian-like dark spots on bright background
        arr = np.full((size, size), 230.0)
        for _ in range(18):
            r = int(rng.integers(8, size - 8))
            c = int(rng.integers(8, size - 8))
            rad = int(rng.integers(3, 6))
            yy, xx = np.ogrid[:size, :size]
            arr[(yy - r) ** 2 + (xx - c) ** 2 <= rad**2] = 20.0
    arr = np.clip(arr + rng.normal(0, 3.0, size=(size, size)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def _make_split_dataset(root: Path, n_train: int = 4, n_test: int = 2) -> None:
    for split, count, offset in (("train", n_train, 0), ("test", n_test, 100)):
        for cls_idx, cls_name in enumerate(CLASSES):
            folder = root / split / cls_name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                _class_image(cls_idx, offset + i).save(folder / f"{cls_name}_{i:02d}.png")


def _parent() -> mock.Mock:
    parent = mock.Mock()
    parent.winfo_toplevel.return_value.title.return_value = "ForensicsApp"
    return parent


def _document(image: Image.Image | None = None, path: Path | None = None) -> ImageDocument:
    doc = ImageDocument()
    doc.current = image
    doc.path = path
    return doc


class Set7BoVWTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _make_split_dataset(self.root)
        self.settings = CodebookSettings(
            folder=str(self.root),
            descriptor="orb",
            words_per_class=10,
            n_keypoints=60,
            max_side=120,
            show_keypoints_popup=False,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_discover_splits_handles_split_and_unsplit_folders(self) -> None:
        train_items, test_items = discover_splits(self.root)
        self.assertEqual(len(train_items), 3 * 4)
        self.assertEqual(len(test_items), 3 * 2)

        with tempfile.TemporaryDirectory() as unsplit_tmp:
            unsplit = Path(unsplit_tmp)
            for idx, cls_name in enumerate(CLASSES):
                folder = unsplit / cls_name
                folder.mkdir()
                for i in range(4):
                    _class_image(idx, i).save(folder / f"{cls_name}_{i}.png")
            u_train, u_test = discover_splits(unsplit)
            self.assertEqual(len(u_train) + len(u_test), 12)
            self.assertEqual(len(u_test), 3)  # 1 held-out test image per class

    def test_codebook_construction_sets_k_to_10_times_num_classes(self) -> None:
        model, samples, failed = build_codebook(self.root, self.settings)
        self.assertEqual(failed, [])
        self.assertEqual(model.classes, sorted(CLASSES))
        # K = 10 * num_classes = 30
        self.assertEqual(model.n_words, 10 * len(CLASSES))
        self.assertEqual(model.centroids.shape, (30, 256))
        self.assertEqual(model.word_patches.shape, (30, PATCH_SIZE, PATCH_SIZE))
        self.assertEqual(model.train_histograms.shape, (12, 30))
        np.testing.assert_allclose(model.train_histograms.sum(axis=1), np.ones(12), atol=1e-6)
        self.assertGreater(len(samples), 0)

        # Save and reload round-trip
        model_path = self.root / BOVW_MODEL_FILENAME
        save_bovw_model(model, model_path)
        loaded = load_bovw_model(model_path)
        self.assertEqual(loaded.n_words, 30)
        np.testing.assert_allclose(loaded.centroids, model.centroids)
        np.testing.assert_array_equal(loaded.word_patches, model.word_patches)
        self.assertEqual(find_bovw_model(self.root / "test" / "kangaroo" / "kangaroo_00.png"), model_path)

    def test_assign_words_and_build_histogram(self) -> None:
        centroids = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        descriptors = np.array([[0.1, -0.1], [9.8, 0.2], [10.1, -0.2], [0.2, 9.9]])
        words = assign_visual_words(descriptors, centroids)
        np.testing.assert_array_equal(words, [0, 1, 1, 2])
        hist = build_histogram(descriptors, centroids, normalize=True)
        np.testing.assert_allclose(hist, [0.25, 0.50, 0.25])
        empty_hist = build_histogram(np.zeros((0, 2)), centroids, normalize=True)
        np.testing.assert_array_equal(empty_hist, [0.0, 0.0, 0.0])

    def test_figures_render_without_error(self) -> None:
        model, samples, _ = build_codebook(self.root, self.settings)
        vw_fig = visual_words_figure(model)
        self.assertEqual(len(vw_fig.axes), 30)
        self.assertEqual(vw_fig.axes[0].get_title(), "vw 0")
        self.assertEqual(vw_fig.axes[29].get_title(), "vw 29")

        kp_fig = sample_keypoints_figure(samples[:4])
        self.assertEqual(len(kp_fig.axes), 4)

        test_items = select_test_images_from_model(model, max_images=3)
        self.assertEqual(len(test_items), 3)
        hist_fig = histogram_rows_figure(test_items, model, connect_keypoints=True)
        self.assertEqual(len(hist_fig.axes), 6)  # 3 rows x 2 columns

        cls_fig = classification_result_figure(_class_image(1, 0), "kangaroo", 0.64)
        self.assertEqual(cls_fig.axes[0].get_title(), "p(kangaroo) = 0.64")

    def test_classifiers_and_probabilities_sum_to_one(self) -> None:
        model, _, _ = build_codebook(self.root, self.settings)
        query = _class_image(2, 99)
        for kind in ("svm_chi2", "svm_rbf", "logistic", "random_forest", "knn"):
            with self.subTest(classifier=kind):
                pred, best_prob, prob_map, hist = predict_image_probabilities(query, model, kind)
                self.assertIn(pred, CLASSES)
                self.assertAlmostEqual(sum(prob_map.values()), 1.0, places=5)
                self.assertAlmostEqual(best_prob, max(prob_map.values()), places=6)
                self.assertEqual(len(hist), 30)

    def test_tools_run_end_to_end(self) -> None:
        # 7.1 CodebookBuilderTool
        with mock.patch.object(bovw, "CodebookDialog", **{"return_value.result": self.settings}):
            res_71 = CodebookBuilderTool().run(_parent(), _document())
        self.assertIsNotNone(res_71.image)
        self.assertIn("30", res_71.details["Visual words (K)"])
        model_path = self.root / BOVW_MODEL_FILENAME
        self.assertTrue(model_path.is_file())

        # 7.2 VisualWordHistogramTool (test_set mode)
        hist_settings = HistogramSettings(model_path=str(model_path), mode="test_set", max_images=3)
        with mock.patch.object(bovw, "HistogramDialog", **{"return_value.result": hist_settings}):
            res_72 = VisualWordHistogramTool().run(_parent(), _document())
        self.assertIsNotNone(res_72.image)
        self.assertEqual(res_72.details["Images visualized"], 3)

        # 7.3 BoVWClassificationTool (classify mode & evaluate mode)
        query_img = _class_image(2, 50)
        cls_settings = ClassificationSettings(model_path=str(model_path), classifier="svm_rbf", mode="classify")
        with mock.patch.object(bovw, "ClassificationDialog", **{"return_value.result": cls_settings}):
            res_73 = BoVWClassificationTool().run(_parent(), _document(query_img))
        self.assertIsNotNone(res_73.image)
        self.assertTrue(res_73.message.startswith("p("))
        self.assertIn("Top probability", res_73.details)

        eval_settings = ClassificationSettings(model_path=str(model_path), classifier="logistic", mode="evaluate")
        with mock.patch.object(bovw, "ClassificationDialog", **{"return_value.result": eval_settings}):
            res_eval = BoVWClassificationTool().run(_parent(), _document(query_img))
        self.assertIsNotNone(res_eval.image)
        self.assertIn("Accuracy", res_eval.details)


    def test_wrapper_folder_and_macosx_are_skipped(self) -> None:
        # The class zip unpacks to Caltech_101_subset/{Caltech_101_subset/<class>, __MACOSX}
        with tempfile.TemporaryDirectory() as tmp:
            outer = Path(tmp)
            inner = outer / "Caltech_101_subset"
            for idx, cls_name in enumerate(CLASSES):
                (inner / cls_name).mkdir(parents=True)
                for i in range(4):
                    _class_image(idx, i).save(inner / cls_name / f"{cls_name}_{i}.png")
            (outer / "__MACOSX" / "Caltech_101_subset").mkdir(parents=True)
            (outer / "__MACOSX" / "Caltech_101_subset" / "._butterfly.png").write_bytes(b"junk")
            self.assertEqual(bovw.dataset_root(outer), inner)
            self.assertEqual(bovw.dataset_root(self.root), self.root)  # train/ is not a wrapper
            model, _, failed = build_codebook(outer, replace(self.settings, folder=str(outer)))
            self.assertEqual(model.root, inner)
            self.assertEqual(model.classes, sorted(CLASSES))
            self.assertEqual(failed, [])

    def test_keypoint_samples_mix_classes_and_descriptor_count_is_kept(self) -> None:
        model, samples, _ = build_codebook(self.root, self.settings)
        self.assertEqual([label for _, _, label in samples[:3]], sorted(CLASSES))
        self.assertEqual(len(samples), 10)
        self.assertGreater(model.n_descriptors, model.n_words)
        model_path = self.root / BOVW_MODEL_FILENAME
        save_bovw_model(model, model_path)
        self.assertEqual(load_bovw_model(model_path).n_descriptors, model.n_descriptors)

    def test_histogram_figure_links_keypoints_and_shows_word_patches(self) -> None:
        from matplotlib.offsetbox import AnnotationBbox
        from matplotlib.patches import ConnectionPatch

        model, _, _ = build_codebook(self.root, self.settings)
        image = _class_image(2, 100)
        features = bovw.extract_features(image, model.matching_settings())
        hist = build_histogram(features.descriptors, model.centroids)
        figure = histogram_rows_figure([(image, "dalmatian")], model, connect_keypoints=True)
        lines = [a for a in figure.artists if isinstance(a, ConnectionPatch)]
        self.assertEqual(len(histogram_rows_figure([(image, "dalmatian")], model).artists), 0)  # lines off by default

        # The 5 most frequent words are coloured (bar and keypoints alike); every other bar is grey
        colours = bovw.highlight_colours(hist, 5)
        top = set(int(w) for w in np.argsort(-hist, kind="stable")[:5])
        self.assertEqual(set(colours), top)
        self.assertEqual(len(set(colours.values())), 5)
        bars = figure.axes[1].patches
        for word_id, bar in enumerate(bars):
            expected = colours.get(word_id, matplotlib.colors.to_rgba(bovw.OTHER_WORDS_COLOUR))
            np.testing.assert_allclose(bar.get_facecolor(), expected)
        words = assign_visual_words(features.descriptors, model.centroids)
        coloured_dots = figure.axes[0].collections[-1].get_facecolors()
        first = int(np.nonzero(np.isin(words, list(colours)))[0][0])
        np.testing.assert_allclose(coloured_dots[0], colours[int(words[first])])
        self.assertEqual(len(lines), int(np.isin(words, list(colours)).sum()))  # lines only for coloured keypoints
        plain = histogram_rows_figure([(image, "dalmatian")], model, highlight_words=0)
        self.assertTrue(all(np.allclose(b.get_facecolor(), matplotlib.colors.to_rgba("lightgray")) for b in plain.axes[1].patches))

    def test_interactive_mode_predicts_only_after_classify(self) -> None:
        model, _, _ = build_codebook(self.root, self.settings)
        model_path = self.root / BOVW_MODEL_FILENAME
        save_bovw_model(model, model_path)
        settings = ClassificationSettings(model_path=str(model_path), mode="interactive")

        class FakePopup:
            def __init__(self, _parent, image, title, model, kind, tfidf=False) -> None:
                self.figure = histogram_rows_figure([(image, title)], model)

        with mock.patch.object(bovw, "ClassificationDialog", **{"return_value.result": settings}), mock.patch.object(
            bovw, "InteractiveClassifierPopup", FakePopup
        ), mock.patch.object(bovw, "predict_image_probabilities") as predict:
            result = BoVWClassificationTool().run(_parent(), _document(_class_image(1, 7)))
        predict.assert_not_called()
        self.assertNotIn("Predicted class", result.details)
        self.assertIn("Classify", result.message)

    def test_chi2_svm_is_default_and_uses_raw_histograms(self) -> None:
        self.assertEqual(ClassificationSettings().classifier, "svm_chi2")
        self.assertEqual(next(iter(bovw.CLASSIFIERS)), "svm_chi2")
        model, _, _ = build_codebook(self.root, self.settings)
        clf, preprocessing, classes = fit_classifier(model, "svm_chi2")
        np.testing.assert_array_equal(preprocessing.transform(model.train_histograms), model.train_histograms)
        self.assertEqual(sorted(classes), sorted(CLASSES))
        # γ = 1 / mean χ² distance between distinct training histograms
        h = model.train_histograms
        pairs = [np.sum((h[i] - h[j]) ** 2 / np.maximum(h[i] + h[j], 1e-300) * ((h[i] + h[j]) > 0))
                 for i in range(len(h)) for j in range(i + 1, len(h))]
        self.assertAlmostEqual(bovw.chi2_gamma(h), 1.0 / np.mean(pairs), places=6)
        self.assertEqual(bovw.chi2_gamma(h[:1]), 1.0)

    def test_tfidf_weighting(self) -> None:
        histograms = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [1.0, 0.0, 0.0]])
        weighting = bovw.IdfWeighting().fit(histograms)
        # word 0 is in all 3 images, words 1 and 2 in one: idf = ln(4 / (1 + n)) + 1
        np.testing.assert_allclose(weighting.idf_, [1.0, np.log(2) + 1, np.log(2) + 1])
        weighted = weighting.transform(histograms)
        np.testing.assert_allclose(weighted.sum(axis=1), 1.0)
        self.assertLess(weighted[0, 0], weighted[0, 1])  # the word every image has now counts less
        self.assertTrue((weighted >= 0).all())
        np.testing.assert_array_equal(weighting.transform(np.zeros((1, 3))), np.zeros((1, 3)))

        model, _, _ = build_codebook(self.root, self.settings)
        query = _class_image(1, 42)
        for kind in bovw.CLASSIFIERS:
            with self.subTest(classifier=kind):
                clf, preprocessing, classes = fit_classifier(model, kind, tfidf=True)
                self.assertIsInstance(preprocessing.steps[0][1], bovw.IdfWeighting)
                pred, best, probs, _ = predict_image_probabilities(query, model, kind, tfidf=True)
                self.assertIn(pred, CLASSES)
                self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)
        self.assertFalse(ClassificationSettings().tfidf)

        model_path = self.root / BOVW_MODEL_FILENAME
        save_bovw_model(model, model_path)
        settings = ClassificationSettings(model_path=str(model_path), mode="evaluate", tfidf=True)
        with mock.patch.object(bovw, "ClassificationDialog", **{"return_value.result": settings}):
            result = BoVWClassificationTool().run(_parent(), _document(query))
        self.assertTrue(result.details["TF-IDF weighting"].startswith("yes"))

    def test_trained_classifier_is_reused(self) -> None:
        model, _, _ = build_codebook(self.root, self.settings)
        trained = fit_classifier(model, "logistic")
        with mock.patch.object(bovw, "fit_classifier") as refit:
            pred, *_ = predict_image_probabilities(_class_image(0, 9), model, "logistic", trained)
        refit.assert_not_called()
        self.assertIn(pred, CLASSES)

if __name__ == "__main__":
    unittest.main()
