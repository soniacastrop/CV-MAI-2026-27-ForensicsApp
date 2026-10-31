import csv
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import eigenfaces as ef
from forensics_app.tools import face_detection as fd
from forensics_app.tools import face_recognition as fr
from forensics_app.tools.eigenfaces import (
    MODEL_FILENAME,
    EigenfaceBuilderTool,
    EigenfaceSettings,
    FaceSet,
    components_for,
    eigenface_gallery,
    export_lfw,
    face_vector,
    fit_eigenfaces,
    label_for,
    load_faces,
    load_model,
    save_model,
    split_folders,
    variance_figure,
)
from forensics_app.tools.face_detection import (
    Face,
    FaceDetectionSettings,
    FaceDetectionTool,
    WebcamSession,
    create_detector,
    crop_face,
    detect_faces,
    draw_faces,
    to_bgr,
)
from forensics_app.tools.face_recognition import (
    EigenfaceExplorer,
    FaceRecognitionTool,
    RecognitionSettings,
    find_model,
    nearest_faces,
    predict,
    reconstruction_errors,
    short_name,
    training_face,
    true_label,
)

SHAPE = (20, 16)  # (h, w) of the synthetic faces
PEOPLE = ("Ada_Lovelace", "Alan_Turing", "Grace_Hopper")
SET6 = Path(__file__).resolve().parents[1] / "images" / "Set6_materials"
YUNET = fd.default_model_path()
LENA = SET6 / "lena.png"


def _person_images(person: int, count: int, seed: int) -> list[np.ndarray]:
    """``count`` noisy uint8 versions of one smooth random 'face' per person."""
    base = np.random.default_rng(100 + person).uniform(40, 215, size=(5, 4))
    face = np.kron(base, np.ones((4, 4)))
    rng = np.random.default_rng(seed)
    return [np.clip(face + rng.normal(0, 12, SHAPE), 0, 255).astype(np.uint8) for _ in range(count)]


def _make_dataset(root: Path, n_train: int = 6, n_test: int = 3) -> None:
    """``root/{train,test}/<person>/*.png`` with distinct people, plus files the loader must skip."""
    for split, count, seed in (("train", n_train, 1), ("test", n_test, 2)):
        for index, person in enumerate(PEOPLE):
            folder = root / split / person
            folder.mkdir(parents=True)
            for number, image in enumerate(_person_images(index, count, seed * 10 + index)):
                Image.fromarray(image, mode="L").save(folder / f"{person}_{number:04d}.png")
    (root / "train" / PEOPLE[0] / "broken.png").write_bytes(b"not a png")
    (root / "train" / ".DS_Store").write_bytes(b"\0")
    (root / "train" / "__MACOSX").mkdir()
    (root / "train" / "__MACOSX" / "._x.png").write_bytes(b"\0")


def _parent() -> mock.Mock:
    parent = mock.Mock()
    parent.winfo_toplevel.return_value.title.return_value = "Forensics"
    return parent


def _document(image: Image.Image, path: Path | None = None) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    document.path = path
    return document


class _DatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _make_dataset(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _model(self, n_components: int = 10) -> ef.EigenfaceModel:
        faces, _failed = load_faces(self.root / "train")
        return fit_eigenfaces(faces, n_components, self.root)


# ------------------------------------------------------------------ 6.2 Eigenfaces


class FaceLoadingTests(_DatasetTestCase):
    def test_labels_come_from_folders_and_bad_files_are_skipped(self) -> None:
        faces, failed = load_faces(self.root / "train")
        self.assertEqual(faces.data.shape, (18, SHAPE[0] * SHAPE[1]))
        self.assertEqual(faces.shape, SHAPE)
        self.assertEqual(sorted(set(faces.labels)), ["Ada Lovelace", "Alan Turing", "Grace Hopper"])
        self.assertEqual([p.name for p in failed], ["broken.png"])
        self.assertTrue(0 <= faces.data.min() and faces.data.max() <= 1)

    def test_other_sizes_and_colour_are_resized_to_the_model_shape(self) -> None:
        big = Image.new("RGBA", (64, 80), (200, 10, 10, 128))
        vector = face_vector(big, SHAPE)
        self.assertEqual(vector.shape, (SHAPE[0] * SHAPE[1],))
        faces, _ = load_faces(self.root / "test", shape=(10, 8))
        self.assertEqual(faces.data.shape[1], 80)

    def test_label_for_loose_files_and_split_folders(self) -> None:
        self.assertEqual(label_for(self.root / "Bob_Smith" / "x.png", self.root), "Bob Smith")
        self.assertEqual(label_for(self.root / "cmu_0012.png", self.root), "cmu")
        self.assertEqual(split_folders(self.root), (self.root / "train", self.root / "test"))
        flat = self.root / "train" / PEOPLE[1]
        self.assertEqual(split_folders(flat), (flat, None))


class PcaTests(_DatasetTestCase):
    def test_pca_follows_the_slides(self) -> None:
        faces, _ = load_faces(self.root / "train")
        model = fit_eigenfaces(faces, 5)
        np.testing.assert_allclose(model.mean, faces.data.mean(axis=0))
        # Orthonormal eigenfaces, eigenvalues sorted descending
        np.testing.assert_allclose(model.eigenfaces @ model.eigenfaces.T, np.eye(5), atol=1e-10)
        self.assertTrue(np.all(np.diff(model.eigenvalues) <= 0))
        # Eigenvectors of the covariance C = A A^T / (n - 1): C u = lambda u
        centred = faces.data - model.mean
        covariance = centred.T @ centred / (len(centred) - 1)
        for u, lam in zip(model.eigenfaces, model.eigenvalues):
            np.testing.assert_allclose(covariance @ u, lam * u, atol=1e-10)
        # w_i = U_K^T (x_i - mu)
        np.testing.assert_allclose(model.coefficients, centred @ model.eigenfaces.T)
        self.assertAlmostEqual(model.total_variance, np.trace(covariance))

    def test_matches_sklearn_pca_up_to_sign(self) -> None:
        from sklearn.decomposition import PCA

        faces, _ = load_faces(self.root / "train")
        model = fit_eigenfaces(faces, 6)
        reference = PCA(n_components=6, svd_solver="full").fit(faces.data)
        np.testing.assert_allclose(np.abs(model.eigenfaces), np.abs(reference.components_), atol=1e-8)
        np.testing.assert_allclose(model.pov, np.cumsum(reference.explained_variance_ratio_), atol=1e-10)

    def test_components_are_capped_and_full_basis_reconstructs_exactly(self) -> None:
        faces, _ = load_faces(self.root / "train")
        model = fit_eigenfaces(faces, 500)
        self.assertEqual(model.n_components, len(faces.data) - 1)
        np.testing.assert_allclose(model.reconstruct(model.coefficients), faces.data, atol=1e-10)
        self.assertAlmostEqual(model.pov[-1], 1.0)
        fewer = fit_eigenfaces(faces, 2)
        error = np.linalg.norm(faces.data - fewer.reconstruct(fewer.coefficients))
        self.assertGreater(error, 0.1)

    def test_too_few_images(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            fit_eigenfaces(FaceSet(np.zeros((1, 4)), ["a"], [Path("a.png")], (2, 2)), 3)

    def test_save_and_load_round_trip(self) -> None:
        model = self._model()
        path = self.root / MODEL_FILENAME
        save_model(model, path)
        loaded = load_model(path)
        for name in ("mean", "eigenfaces", "eigenvalues", "coefficients"):
            np.testing.assert_allclose(getattr(loaded, name), getattr(model, name))
        self.assertEqual(list(loaded.labels), list(model.labels))
        self.assertTrue(loaded.paths[0].startswith("train/"))
        self.assertEqual(loaded.shape, SHAPE)
        self.assertEqual(loaded.root, self.root)
        (self.root / "junk.npz").write_bytes(b"junk")
        with self.assertRaisesRegex(ValueError, "not an eigenface model"):
            load_model(self.root / "junk.npz")

    def test_gallery_and_variance_plot(self) -> None:
        model = self._model()
        self.assertEqual(eigenface_gallery(model, 12).mode, "RGB")
        self.assertGreater(eigenface_gallery(model, 12, include_mean=True).height, eigenface_gallery(model, 4).height)
        figure = variance_figure(model, target=0.5)
        self.assertEqual([ax.get_title() for ax in figure.axes][1], "(b) Proportion of variance explained")
        needed = components_for(model, 0.5)
        self.assertGreaterEqual(model.pov[needed - 1], 0.5)
        self.assertTrue(needed == 1 or model.pov[needed - 2] < 0.5)
        self.assertIsNone(components_for(self._model(1), 0.9999))


class ExportLfwTests(unittest.TestCase):
    def test_export_writes_split_folders_and_labels(self) -> None:
        images = np.random.default_rng(0).uniform(0, 1, size=(20, 5, 4)).astype(np.float32)
        bunch = mock.Mock(images=images, target=np.repeat([0, 1], 10), target_names=np.array(["Ann Lee", "Bo Wu"]))
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "sklearn.datasets.fetch_lfw_people", return_value=bunch
        ) as fetch:
            folder = Path(tmp)
            n_train, n_test, names = export_lfw(folder, test_size=0.3)
            fetch.assert_called_once_with(min_faces_per_person=70, resize=0.4)
            self.assertEqual((n_train, n_test, names), (14, 6, ["Ann Lee", "Bo Wu"]))
            for split in ("train", "test"):  # stratified: both people in both splits
                self.assertEqual(sorted(p.name for p in (folder / split).iterdir()), ["Ann_Lee", "Bo_Wu"])
            with open(folder / "labels.csv", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 20)
            first = rows[0]
            self.assertTrue((folder / first["path"]).is_file())
            self.assertIn(first["name"].replace(" ", "_"), first["path"])
            saved = np.asarray(Image.open(folder / first["path"]))
            self.assertEqual(saved.shape, (5, 4))
            self.assertGreater(saved.max(), 1)  # [0, 1] input was scaled to [0, 255]


class EigenfaceToolTests(_DatasetTestCase):
    def test_run_saves_model_and_shows_eigenfaces(self) -> None:
        settings = EigenfaceSettings(folder=str(self.root), n_components=8, n_display=6)
        parent = _parent()
        with mock.patch(f"{ef.__name__}.EigenfaceDialog", **{"return_value.result": settings}), mock.patch(
            f"{ef.__name__}.show_figure"
        ) as show:
            result = EigenfaceBuilderTool().run(parent, ImageDocument())
        self.assertFalse(EigenfaceBuilderTool.requires_image)
        show.assert_called_once()
        self.assertTrue((self.root / MODEL_FILENAME).is_file())
        self.assertEqual(load_model(self.root / MODEL_FILENAME).n_components, 8)
        self.assertEqual(result.details["Training images"], 18)
        self.assertEqual(result.details["People"], 3)
        self.assertEqual(result.details["Unreadable files"], "broken.png")
        self.assertIn("Test folder", result.details)
        self.assertEqual(result.image.mode, "RGB")
        parent.winfo_toplevel.return_value.title.assert_called_with("Forensics")  # title restored

    def test_cancel_and_empty_folder(self) -> None:
        with mock.patch(f"{ef.__name__}.EigenfaceDialog", **{"return_value.result": None}):
            self.assertIsNone(EigenfaceBuilderTool().run(_parent(), ImageDocument()))
        with tempfile.TemporaryDirectory() as empty, mock.patch(
            f"{ef.__name__}.EigenfaceDialog", **{"return_value.result": EigenfaceSettings(folder=empty)}
        ):
            result = EigenfaceBuilderTool().run(_parent(), ImageDocument())
        self.assertIsNone(result.image)
        self.assertIn("at least 2", result.message)


# ------------------------------------------------------------------ 6.3 Recognition


class RecognitionFunctionTests(_DatasetTestCase):
    def test_training_face_is_its_own_nearest_neighbour(self) -> None:
        model = self._model()
        neighbours = nearest_faces(model, model.coefficients[4], 3)
        self.assertEqual(neighbours[0].index, 4)
        self.assertAlmostEqual(neighbours[0].distance, 0.0)
        self.assertEqual([n.label for n in neighbours], [model.labels[4]] * 3)
        distances = [n.distance for n in neighbours]
        self.assertEqual(distances, sorted(distances))

    def test_distance_is_euclidean_on_the_weights(self) -> None:
        model = self._model()
        weights = model.coefficients[0] + 1.0
        neighbour = nearest_faces(model, weights, 1, k=4)[0]
        expected = np.linalg.norm(model.coefficients[:, :4] - weights[:4], axis=1)
        self.assertAlmostEqual(neighbour.distance, expected.min())
        whitened = nearest_faces(model, weights, 1, k=4, whiten=True)[0]
        self.assertAlmostEqual(
            whitened.distance,
            np.linalg.norm((model.coefficients[:, :4] - weights[:4]) / np.sqrt(model.eigenvalues[:4]), axis=1).min(),
        )

    def test_test_faces_are_recognised(self) -> None:
        model = self._model()
        test, _ = load_faces(self.root / "test", model.shape)
        for whiten in (False, True):
            with self.subTest(whiten=whiten):
                self.assertEqual(predict(model, test.data, whiten=whiten), test.labels)

    def test_true_label_short_name_and_model_search(self) -> None:
        model = self._model()
        path = self.root / "test" / PEOPLE[2] / "x.png"
        self.assertEqual(true_label(path, model), "Grace Hopper")
        self.assertIsNone(true_label(self.root / "elsewhere" / "x.png", model))
        self.assertIsNone(true_label(None, model))
        self.assertEqual(short_name("George W Bush"), "Bush")
        save_model(model, self.root / MODEL_FILENAME)
        self.assertEqual(find_model(path), self.root / MODEL_FILENAME)

    def test_moved_training_image_falls_back_to_its_reconstruction(self) -> None:
        model = self._model()
        face, found = training_face(model, 0)
        self.assertTrue(found)
        np.testing.assert_allclose(face.ravel(), load_faces(self.root / "train")[0].data[0])
        (self.root / model.paths[0]).unlink()
        face, found = training_face(model, 0)
        self.assertFalse(found)
        self.assertEqual(face.shape, SHAPE)


class RecognitionToolTests(_DatasetTestCase):
    def setUp(self) -> None:
        super().setUp()
        save_model(self._model(), self.root / MODEL_FILENAME)
        self.test_path = self.root / "test" / PEOPLE[1] / f"{PEOPLE[1]}_0000.png"
        self.document = ImageDocument()
        self.document.load(self.test_path)

    def _run(self, settings: RecognitionSettings | None) -> object:
        with mock.patch(f"{fr.__name__}.RecognitionDialog", **{"return_value.result": settings}) as dialog:
            result = FaceRecognitionTool().run(_parent(), self.document)
        self.assertEqual(dialog.call_args[0][1], self.root / MODEL_FILENAME)  # model found automatically
        return result

    def test_query_shows_closest_faces_with_names(self) -> None:
        result = self._run(RecognitionSettings(model_path=str(self.root / MODEL_FILENAME), n_neighbours=3))
        self.assertEqual(result.details["Predicted"], "Alan Turing")
        self.assertEqual(result.details["True"], "Alan Turing")
        self.assertEqual(result.details["Correct"], "yes")
        self.assertEqual(sum(key.startswith("#") for key in result.details), 3)
        self.assertIn("✓", result.message)
        self.assertEqual(result.image.mode, "RGB")
        self.assertGreater(result.image.width, result.image.height)  # one row of tiles

    def test_evaluate_reports_accuracy_and_gallery(self) -> None:
        settings = RecognitionSettings(model_path=str(self.root / MODEL_FILENAME), mode="evaluate", n_gallery=6)
        result = self._run(settings)
        self.assertEqual(result.details["Test faces"], 9)
        self.assertTrue(result.details["Accuracy"].startswith("100.0%"))
        self.assertEqual(result.details["  Grace Hopper"], "3/3 (100%)")
        self.assertIn("accuracy 100.0%", result.message)

    def test_fewer_eigenfaces_and_invalid_k(self) -> None:
        result = self._run(RecognitionSettings(model_path=str(self.root / MODEL_FILENAME), n_components=2))
        self.assertEqual(result.details["Eigenfaces used (k)"], "2 of 10")
        with self.assertRaisesRegex(ValueError, "only 10 eigenfaces"):
            self._run(RecognitionSettings(model_path=str(self.root / MODEL_FILENAME), n_components=11))

    def test_cancel(self) -> None:
        self.assertIsNone(self._run(None))

    def test_explore_opens_the_explorer_popup(self) -> None:
        settings = RecognitionSettings(model_path=str(self.root / MODEL_FILENAME), mode="explore")
        with mock.patch(f"{fr.__name__}.show_figure") as show:
            result = self._run(settings)
        show.assert_called_once()
        self.assertIsInstance(show.return_value.explorer, EigenfaceExplorer)  # kept alive for the slider
        self.assertIsNone(result.image)
        self.assertIn("Reconstruction error (k = 1 / K)", result.details)


class EigenfaceExplorerTests(_DatasetTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.model = self._model()
        test, _ = load_faces(self.root / "test", self.model.shape)
        self.query = test.data[0]
        self.explorer = EigenfaceExplorer(self.model, self.query, 3, truth=test.labels[0])

    def test_error_curve_is_the_real_reconstruction_error_and_decreases(self) -> None:
        errors = reconstruction_errors(self.model, self.query)
        for k in (1, 4, 10):
            direct = np.linalg.norm(self.query - self.model.reconstruct(self.model.project(self.query, k)))
            self.assertAlmostEqual(errors[k - 1], direct)
        self.assertTrue(np.all(np.diff(errors) <= 1e-12))

    def test_k_updates_reconstruction_slider_and_bars(self) -> None:
        e = self.explorer
        self.assertEqual((e.k, e.slider.val, e.selected), (3, 3, 0))
        e.set_k(7)
        self.assertEqual(e.slider.val, 7)
        expected = np.clip(self.model.reconstruct(e.weights[:7]).reshape(SHAPE), 0, 1)
        np.testing.assert_allclose(e.recon_image.get_array(), expected)
        self.assertIn("k = 7", e.ax_recon.get_title())
        self.assertEqual(e.nearest, nearest_faces(self.model, e.weights, 1, 7)[0])
        colours = [bar.get_facecolor() for bar in e.bars]
        self.assertEqual(colours[6], colours[1])  # used
        self.assertNotEqual(colours[7], colours[1])  # not used yet
        e.slider.set_val(2)  # dragging the slider goes through the same path
        self.assertEqual(e.k, 2)
        e.set_k(999)
        self.assertEqual(e.k, self.model.n_components)

    def test_click_and_keys_select_eigenfaces(self) -> None:
        e = self.explorer
        with mock.patch.object(e.figure, "canvas"):
            e.on_click(mock.Mock(inaxes=e.ax_weights, xdata=4.3))
            self.assertEqual(e.selected, 4)
            np.testing.assert_allclose(e.eigen_image.get_array(), self.model.eigenfaces[4].reshape(SHAPE))
            self.assertIn("eigenface 4", e.ax_eigen.get_title())
            e.on_click(mock.Mock(inaxes=e.ax_error, xdata=2.0))  # other axes: ignored
            e.on_click(mock.Mock(inaxes=e.ax_weights, xdata=None))
            self.assertEqual(e.selected, 4)
            e.on_key(mock.Mock(key="up"))
            e.on_key(mock.Mock(key="right"))
            e.on_key(mock.Mock(key="q"))
        self.assertEqual((e.selected, e.k), (5, 4))


# ------------------------------------------------------------------ 6.1 Detection


def _yunet_row(x: float, y: float, w: float, h: float, score: float) -> list[float]:
    landmarks = [x + 0.3 * w, y + 0.4 * h, x + 0.7 * w, y + 0.4 * h, x + 0.5 * w, y + 0.6 * h,
                 x + 0.35 * w, y + 0.8 * h, x + 0.65 * w, y + 0.8 * h]
    return [x, y, w, h, *landmarks, score]


class _FakeDetector:
    """Stands in for ``cv2.FaceDetectorYN`` (same ``detect`` output layout: 15 values per face)."""

    def __init__(self, rows: list[list[float]] | None) -> None:
        self.rows = rows
        self.sizes: list[tuple[int, int]] = []

    def setInputSize(self, size: tuple[int, int]) -> None:
        self.sizes.append(size)

    def detect(self, image: np.ndarray) -> tuple[int, np.ndarray | None]:
        return 1, None if self.rows is None else np.array(self.rows, dtype=np.float32)


class _FakeCapture:
    def __init__(self, frames: int = 3) -> None:
        self.frames = frames
        self.released = False

    def isOpened(self) -> bool:
        return not self.released

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.frames == 0:
            return False, None
        self.frames -= 1
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :8] = (255, 0, 0)  # blue stripe on the left (BGR)
        return True, frame

    def release(self) -> None:
        self.released = True


class DetectionFunctionTests(unittest.TestCase):
    def test_detect_faces_parses_and_sorts_by_score(self) -> None:
        detector = _FakeDetector([_yunet_row(5, 5, 20, 24, 0.7), _yunet_row(40, 10, 16, 20, 0.95)])
        faces = detect_faces(detector, np.zeros((60, 80, 3), dtype=np.uint8))
        self.assertEqual(detector.sizes, [(80, 60)])  # input size follows the frame (width, height)
        self.assertEqual([f.score for f in faces], sorted([f.score for f in faces], reverse=True))
        self.assertEqual(faces[0].box, (40.0, 10.0, 16.0, 20.0))
        self.assertEqual(faces[0].landmarks.shape, (5, 2))
        self.assertEqual(detect_faces(_FakeDetector(None), np.zeros((8, 8, 3), dtype=np.uint8)), [])

    def test_draw_and_crop(self) -> None:
        image = Image.new("L", (100, 80), 30)
        face = Face((20, 20, 40, 30), np.array([[30, 30], [50, 30], [40, 38], [32, 44], [48, 44]], float), 0.9)
        drawn = draw_faces(image, [face])
        self.assertEqual(drawn.mode, "RGB")
        self.assertEqual(drawn.getpixel((40, 20)), fd.BOX_COLOUR)  # top edge of the box
        self.assertEqual(drawn.getpixel((40, 38)), fd.LANDMARK_COLOUR)  # nose
        self.assertNotEqual(draw_faces(image, [face], show_landmarks=False).getpixel((40, 38)), fd.LANDMARK_COLOUR)
        self.assertEqual(crop_face(image, face, 0.0).size, (40, 30))
        self.assertEqual(crop_face(image, face, 0.5).size, (80, 60))
        self.assertEqual(crop_face(image, Face((-10, -10, 50, 50), face.landmarks, 0.9), 0.5).size, (65, 65))

    def test_to_bgr_handles_gray_and_alpha(self) -> None:
        self.assertEqual(to_bgr(Image.new("L", (4, 3), 9)).shape, (3, 4, 3))
        bgr = to_bgr(Image.new("RGBA", (4, 3), (255, 0, 0, 10)))
        self.assertEqual(tuple(bgr[0, 0]), (0, 0, 255))
        self.assertTrue(bgr.flags["C_CONTIGUOUS"])

    def test_missing_model_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "not found"):
            create_detector(FaceDetectionSettings(model_path="nowhere.onnx"))


class WebcamSessionTests(unittest.TestCase):
    def test_frames_are_mirrored_detected_and_released(self) -> None:
        capture = _FakeCapture(frames=2)
        detector = _FakeDetector([_yunet_row(10, 10, 20, 20, 0.8)])
        with mock.patch(f"{fd.__name__}.create_detector", return_value=detector):
            session = WebcamSession(FaceDetectionSettings(source="webcam"), capture)
        first = session.next_frame()
        self.assertEqual(first.size, (64, 48))
        self.assertEqual(session.last_frame.getpixel((63, 30)), (0, 0, 255))  # mirrored blue stripe, now RGB
        self.assertEqual(len(session.last_faces), 1)
        session.next_frame()
        self.assertEqual(session.frames, 2)
        self.assertIsNone(session.next_frame())  # camera stopped delivering
        session.close()
        self.assertTrue(capture.released)

    def test_closed_camera_raises(self) -> None:
        capture = _FakeCapture()
        capture.released = True
        with self.assertRaisesRegex(RuntimeError, "Could not open webcam"):
            WebcamSession(FaceDetectionSettings(), capture)


class FaceDetectionToolTests(unittest.TestCase):
    def _run(self, settings: FaceDetectionSettings, document: ImageDocument, rows: list[list[float]] | None):
        with (
            mock.patch(f"{fd.__name__}.FaceDetectionDialog", **{"return_value.result": settings}),
            mock.patch(f"{fd.__name__}.create_detector", return_value=_FakeDetector(rows)),
        ):
            return FaceDetectionTool().run(_parent(), document)

    def test_annotate_and_crop_current_image(self) -> None:
        document = _document(Image.new("RGB", (120, 90), (90, 90, 90)))
        rows = [_yunet_row(30, 20, 40, 50, 0.92)]
        result = self._run(FaceDetectionSettings(output="annotate"), document, rows)
        self.assertEqual(result.details["Faces found"], 1)
        self.assertIn("score 0.920", result.details["Face 1"])
        self.assertIn("  Nose", result.details)
        self.assertEqual(result.image.size, (120, 90))
        crop = self._run(FaceDetectionSettings(output="crop", crop_margin=0.0), document, rows)
        self.assertEqual(crop.image.size, (40, 50))
        none = self._run(FaceDetectionSettings(output="crop"), document, None)
        self.assertIsNone(none.image)
        self.assertIn("No face", none.message)

    def test_side_by_side_is_the_default_like_the_slide(self) -> None:
        document = _document(Image.new("RGB", (120, 90), (90, 90, 90)))
        result = self._run(FaceDetectionSettings(), document, [_yunet_row(30, 20, 40, 50, 0.92)])
        self.assertEqual(FaceDetectionSettings().output, "side")
        self.assertEqual(result.image.mode, "RGB")
        self.assertGreater(result.image.width, 1.5 * result.image.height)  # two panels next to each other
        self.assertEqual(result.details["Faces found"], 1)

    def test_image_source_needs_an_image_and_cancel(self) -> None:
        self.assertFalse(FaceDetectionTool.requires_image)
        with self.assertRaisesRegex(ValueError, "Open an image"):
            self._run(FaceDetectionSettings(), ImageDocument(), [])
        self.assertIsNone(self._run(None, ImageDocument(), []))

    def test_webcam_capture_becomes_the_result(self) -> None:
        frame = Image.new("RGB", (64, 48), (10, 20, 30))
        session = mock.Mock(frames=12, fps=24.5, last_faces=[])
        window = mock.Mock(captured=frame)
        settings = FaceDetectionSettings(source="webcam")
        with (
            mock.patch(f"{fd.__name__}.FaceDetectionDialog", **{"return_value.result": settings}),
            mock.patch(f"{fd.__name__}.WebcamSession", return_value=session),
            mock.patch(f"{fd.__name__}.WebcamWindow", return_value=window),
        ):
            parent = _parent()
            result = FaceDetectionTool().run(parent, ImageDocument())
            parent.wait_window.assert_called_once_with(window.window)
            self.assertGreater(result.image.width, result.image.height)
            self.assertEqual(result.details["Frames processed"], 12)
            self.assertTrue(result.message.startswith("Captured webcam frame"))
            window.captured = None
            closed = FaceDetectionTool().run(_parent(), ImageDocument())
        self.assertIsNone(closed.image)
        self.assertIn("Webcam closed after 12 frame(s)", closed.message)


@unittest.skipUnless(YUNET is not None and LENA.is_file(), "YuNet model / lena.png from Set6_materials not present")
class YuNetOnLenaTests(unittest.TestCase):
    """The real model reproduces the notebook: one face, score 0.91, right eye at (272, 270)."""

    def test_detects_lena_like_the_notebook(self) -> None:
        settings = FaceDetectionSettings(model_path=str(YUNET))
        image = Image.open(LENA)
        faces = detect_faces(create_detector(settings, image.size), to_bgr(image))
        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(faces[0].score, 0.91, places=2)
        np.testing.assert_allclose(faces[0].landmarks[0], (272, 270), atol=2)

    def test_higher_threshold_drops_the_face(self) -> None:
        image = Image.open(LENA)
        settings = FaceDetectionSettings(model_path=str(YUNET), score_threshold=0.95)
        self.assertEqual(detect_faces(create_detector(settings, image.size), to_bgr(image)), [])


if __name__ == "__main__":
    unittest.main()
