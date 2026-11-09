"""Set 7: Bag of Visual Words (BoVW) — codebook construction (7.1), histogram construction (7.2), and classification (7.3)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import ConnectionPatch
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialog_fields import read_number, spin_row
from .eigenfaces import TRAIN_DIR, TEST_DIR, figure_to_image, grid_shape, label_for
from .feature_matching import (
    DESCRIPTORS,
    FAST_THRESHOLD_RANGE,
    MAX_SIDE_RANGE,
    N_KEYPOINTS_RANGE,
    N_OCTAVES_RANGE,
    N_SCALES_RANGE,
    SIGMA_RANGE,
    MatchingSettings,
    extract_features,
    prepare_gray,
)
from .hog import show_figure
from .image_lookup import IMAGE_SUFFIXES, SKIPPED_DIRS, list_images
from .progress import ProgressPopup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOVW_DATASET = PROJECT_ROOT / "datasets" / "bovw_dataset"
# The class material: Caltech-101 subset (butterfly, dalmatian, kangaroo) unzipped into images/
CLASS_BOVW_DATASET = PROJECT_ROOT / "images" / "Caltech_101_subset" / "Caltech_101_subset"


def default_bovw_dataset() -> Path:
    """``datasets/bovw_dataset`` if it exists, else the class's Caltech-101 subset."""
    return DEFAULT_BOVW_DATASET if DEFAULT_BOVW_DATASET.is_dir() or not CLASS_BOVW_DATASET.is_dir() else CLASS_BOVW_DATASET


def dataset_root(folder: Path) -> Path:
    """Step into a lone wrapper folder (``Caltech_101_subset/Caltech_101_subset/<class>``) so classes are found.

    Unzipped archives often add one extra level, plus ``__MACOSX`` metadata that is ignored.
    """
    root = Path(folder)
    while True:
        if any(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in root.iterdir()):
            return root
        subfolders = [
            path
            for path in root.iterdir()
            if path.is_dir() and path.name not in SKIPPED_DIRS and not path.name.startswith(".")
        ]
        if len(subfolders) != 1 or subfolders[0].name in (TRAIN_DIR, TEST_DIR):
            return root
        root = subfolders[0]
BOVW_MODEL_FILENAME = "bovw_model.npz"
MODEL_TYPES = [("BoVW model", "*.npz"), ("All files", "*.*")]

WORDS_PER_CLASS_DEFAULT = 10  # Slide 7.1: "Cluster descriptors using K-Means (K = 10*num_classes)"
WORDS_PER_CLASS_RANGE = (1, 100)
CLUSTERS_OVERRIDE_RANGE = (0, 2000)  # 0 = use words_per_class * num_classes
PATCH_SIZE = 28  # side length (px) of each visual word thumbnail (vw 0 .. vw K-1)
MAX_DISPLAY_WORDS = 100
MAX_TEST_HISTOGRAMS = 6
MAX_CONNECTED_KEYPOINTS = 150  # lines drawn per image in the 7.2 figure
N_KEYPOINT_SAMPLES = 10  # training images with keypoints shown in 7.1 (2 rows of 5, as on the slide)
MAX_SAMPLE_KEYPOINTS = 60  # red dots drawn per sample image in 7.1
MODEL_SEARCH_DEPTH = 4
AUTO_TEST_FRACTION = 0.25
RANDOM_SEED = 0

CLASSIFIERS = {
    "svm_rbf": "SVM (RBF kernel, probability=True)",
    "logistic": "Logistic Regression (multinomial)",
    "random_forest": "Random Forest (100 trees)",
    "knn": "k-Nearest Neighbours (distance-weighted)",
}

HISTOGRAM_MODES = {
    "test_set": "Test images from the dataset (one per class, as in slide 7.2)",
    "current": "Current canvas image",
}

CLASSIFY_MODES = {
    "interactive": "Visualize 7.2 histogram first, then click 'Classify' button (as in slide 7.3)",
    "classify": "Run trained classifier immediately and show top probability on canvas",
    "evaluate": "Evaluate the whole test set (predicted probability gallery)",
}

# Remember the most recently saved or loaded codebook path so 7.2 and 7.3 find it automatically
_LAST_MODEL_PATH: Path | None = None


@dataclass(frozen=True)
class CodebookSettings:
    folder: str = str(default_bovw_dataset())
    descriptor: str = "sift"
    words_per_class: int = WORDS_PER_CLASS_DEFAULT
    n_clusters_override: int = 0  # 0 -> K = words_per_class * num_classes
    n_keypoints: int = 200  # ORB
    fast_threshold: float = 0.08  # ORB
    n_octaves: int = 5  # SIFT
    n_scales: int = 3  # SIFT
    sigma_min: float = 1.6  # SIFT
    max_side: int = 300
    show_keypoints_popup: bool = True

    def to_matching_settings(self) -> MatchingSettings:
        return MatchingSettings(
            descriptor=self.descriptor,
            n_keypoints=self.n_keypoints,
            fast_threshold=self.fast_threshold,
            n_octaves=self.n_octaves,
            n_scales=self.n_scales,
            sigma_min=self.sigma_min,
            max_side=self.max_side,
        )


@dataclass(frozen=True)
class HistogramSettings:
    model_path: str = ""
    mode: str = "test_set"
    max_images: int = 3  # Slide 7.2 shows 3 test images (butterfly, kangaroo, dalmatian)
    connect_keypoints: bool = True


@dataclass(frozen=True)
class ClassificationSettings:
    model_path: str = ""
    classifier: str = "svm_rbf"
    mode: str = "interactive"
    n_gallery: int = 9


@dataclass(frozen=True)
class BoVWModel:
    """Saved Bag-of-Visual-Words codebook, visual-word patches, and training histograms."""

    centroids: np.ndarray  # (K, D) float64 cluster centres (visual words)
    word_patches: np.ndarray  # (K, PATCH_SIZE, PATCH_SIZE) uint8 grayscale patch for each visual word
    train_histograms: np.ndarray  # (N_train, K) float64 normalized frequency histograms
    train_labels: np.ndarray  # (N_train,) class names
    train_paths: np.ndarray  # (N_train,) relative paths from root
    test_paths: np.ndarray  # (N_test,) relative paths from root
    test_labels: np.ndarray  # (N_test,) class names
    descriptor: str  # "sift" or "orb"
    n_keypoints: int
    fast_threshold: float
    n_octaves: int
    n_scales: int
    sigma_min: float
    max_side: int
    root: Path | None = None
    n_descriptors: int = 0  # rows of the stacked descriptor matrix K-Means was fitted on

    @property
    def n_words(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def classes(self) -> list[str]:
        return sorted(set(str(label) for label in self.train_labels))

    def matching_settings(self) -> MatchingSettings:
        return MatchingSettings(
            descriptor=self.descriptor,
            n_keypoints=int(self.n_keypoints),
            fast_threshold=float(self.fast_threshold),
            n_octaves=int(self.n_octaves),
            n_scales=int(self.n_scales),
            sigma_min=float(self.sigma_min),
            max_side=int(self.max_side),
        )


# ---------------------------------------------------------------- dataset discovery


def discover_splits(
    root: Path,
    seed: int = RANDOM_SEED,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """Return ``(train_items, test_items)`` as ``[(path, class_label), ...]``.

    Supports both ``root/{train,test}/<class>/*.jpg`` and unsplit ``root/<class>/*.jpg``
    (automatically holding out ~25% of each class as test images when ``test/`` is absent).
    """
    train_dir = root / TRAIN_DIR
    test_dir = root / TEST_DIR
    if train_dir.is_dir():
        train_items = [(p, label_for(p, train_dir)) for p in list_images(train_dir)]
        test_items = [(p, label_for(p, test_dir)) for p in list_images(test_dir)] if test_dir.is_dir() else []
        return train_items, test_items

    all_items = [(p, label_for(p, root)) for p in list_images(root)]
    by_class: dict[str, list[Path]] = {}
    for path, label in all_items:
        by_class.setdefault(label, []).append(path)

    rng = np.random.default_rng(seed)
    train_items: list[tuple[Path, str]] = []
    test_items: list[tuple[Path, str]] = []
    for label in sorted(by_class):
        paths = sorted(by_class[label])
        if len(paths) < 2:
            train_items.extend((p, label) for p in paths)
            continue
        n_test = max(1, int(round(len(paths) * AUTO_TEST_FRACTION)))
        n_test = min(n_test, len(paths) - 1)
        perm = rng.permutation(len(paths))
        test_idx = set(perm[:n_test].tolist())
        for idx, p in enumerate(paths):
            (test_items if idx in test_idx else train_items).append((p, label))
    return train_items, test_items


# ---------------------------------------------------------------- 7.1 Codebook construction


def crop_patch(gray: np.ndarray, row: float, col: float, patch_size: int = PATCH_SIZE) -> np.ndarray:
    """Extract a ``patch_size × patch_size`` uint8 patch centred at ``(row, col)`` from ``gray`` (in [0, 1])."""
    half = patch_size // 2
    padded = np.pad(gray, half, mode="reflect")
    r = int(round(row)) + half
    c = int(round(col)) + half
    patch = padded[r - half : r - half + patch_size, c - half : c - half + patch_size]
    if patch.shape != (patch_size, patch_size):
        patch = np.resize(patch, (patch_size, patch_size))
    low, high = float(patch.min()), float(patch.max())
    if high > low:
        patch = (patch - low) / (high - low)
    return np.clip(np.round(patch * 255.0), 0, 255).astype(np.uint8)


def assign_visual_words(descriptors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Map each descriptor row to the index ``0 .. K-1`` of its closest cluster centroid."""
    if len(descriptors) == 0:
        return np.zeros(0, dtype=np.intp)
    desc = np.asarray(descriptors, dtype=np.float64)
    # ||d - c||^2 = ||d||^2 + ||c||^2 - 2 d . c^T
    d_norm = np.sum(desc**2, axis=1, keepdims=True)
    c_norm = np.sum(centroids**2, axis=1, keepdims=True).T
    sq_dists = d_norm + c_norm - 2.0 * (desc @ centroids.T)
    return np.argmin(sq_dists, axis=1).astype(np.intp)


def build_histogram(descriptors: np.ndarray, centroids: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Build a ``K``-dimensional visual word frequency histogram for ``descriptors``."""
    k = len(centroids)
    if len(descriptors) == 0:
        return np.zeros(k, dtype=np.float64)
    words = assign_visual_words(descriptors, centroids)
    counts = np.bincount(words, minlength=k).astype(np.float64)
    if normalize and counts.sum() > 0:
        counts /= counts.sum()
    return counts


def build_codebook(
    root: Path,
    settings: CodebookSettings,
    progress: Callable[[int, int, Path], None] | None = None,
) -> tuple[BoVWModel, list[tuple[np.ndarray, np.ndarray, str]], list[Path]]:
    """Extract descriptors from training images, cluster with MiniBatchKMeans, and build the BoVW model.

    Returns ``(model, sample_keypoints_visuals, unreadable_paths)`` where ``sample_keypoints_visuals``
    holds up to 10 ``(gray_image, keypoints, label)`` tuples for the keypoint preview popup on Slide 7.1.
    """
    root = dataset_root(root)
    train_items, test_items = discover_splits(root)
    if not train_items:
        raise ValueError(f"No training images found in {root}.")

    match_settings = settings.to_matching_settings()
    feature_descriptors: list[np.ndarray] = []
    per_image_descriptors: list[np.ndarray] = []
    kept_train_paths: list[Path] = []
    kept_train_labels: list[str] = []
    failed: list[Path] = []

    # Track the source image and keypoint of every descriptor so each centroid gets its closest visual patch
    all_patch_sources: list[tuple[np.ndarray, float, float]] = []
    samples_by_class: dict[str, list[tuple[np.ndarray, np.ndarray, str]]] = {}

    for index, (path, label) in enumerate(train_items):
        if progress is not None:
            progress(index, len(train_items), path)
        try:
            with Image.open(path) as opened:
                image = opened.copy()
        except (OSError, UnidentifiedImageError):
            failed.append(path)
            continue

        features = extract_features(image, match_settings)
        desc = np.asarray(features.descriptors, dtype=np.float64)
        per_image_descriptors.append(desc)
        kept_train_paths.append(path)
        kept_train_labels.append(label)
        class_samples = samples_by_class.setdefault(label, [])
        if len(class_samples) < N_KEYPOINT_SAMPLES:
            class_samples.append((features.gray, features.keypoints, label))
        if len(desc) > 0:
            feature_descriptors.append(desc)
            for row, col in features.keypoints:
                all_patch_sources.append((features.gray, float(row), float(col)))

    if not feature_descriptors:
        raise ValueError("No local descriptors could be extracted from the training images.")

    # Slide 7.1 shows training images of every class mixed: take them round-robin over the classes
    sample_visuals: list[tuple[np.ndarray, np.ndarray, str]] = []
    for round_index in range(N_KEYPOINT_SAMPLES):
        for label in sorted(samples_by_class):
            if round_index < len(samples_by_class[label]) and len(sample_visuals) < N_KEYPOINT_SAMPLES:
                sample_visuals.append(samples_by_class[label][round_index])

    classes = sorted(set(kept_train_labels))
    if len(classes) < 2:
        raise ValueError(
            f"Need at least 2 classes (subfolders) to build a classification codebook; found {len(classes)}."
        )

    # Slide 7.1: stack descriptors with np.vstack and fit MiniBatchKMeans(n_clusters=ncl, random_state=0)
    descriptors = feature_descriptors[0]
    for descriptor in feature_descriptors[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    ncl = (
        settings.n_clusters_override
        if settings.n_clusters_override > 0
        else settings.words_per_class * len(classes)
    )
    ncl = min(ncl, len(descriptors))
    if ncl < 2:
        raise ValueError(f"Need at least 2 descriptors to cluster; found {len(descriptors)}.")

    kmeans = MiniBatchKMeans(n_clusters=ncl, random_state=RANDOM_SEED, n_init="auto").fit(descriptors)
    centroids = np.asarray(kmeans.cluster_centers_, dtype=np.float64)

    # For each cluster centre (visual word), extract the image patch of the closest training descriptor
    word_patches = np.zeros((ncl, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    d_norm = np.sum(descriptors**2, axis=1, keepdims=True)
    c_norm = np.sum(centroids**2, axis=1, keepdims=True).T
    sq_dists = d_norm + c_norm - 2.0 * (descriptors @ centroids.T)
    closest_indices = np.argmin(sq_dists, axis=0)
    for word_idx, desc_idx in enumerate(closest_indices):
        gray_img, r, c = all_patch_sources[int(desc_idx)]
        word_patches[word_idx] = crop_patch(gray_img, r, c, PATCH_SIZE)

    # Build normalized frequency histograms for all training images
    train_histograms = np.vstack([build_histogram(d, centroids, normalize=True) for d in per_image_descriptors])

    valid_test_paths: list[str] = []
    valid_test_labels: list[str] = []
    for path, label in test_items:
        if label in set(classes):
            valid_test_paths.append(path.relative_to(root).as_posix())
            valid_test_labels.append(label)

    model = BoVWModel(
        centroids=centroids,
        word_patches=word_patches,
        train_histograms=train_histograms,
        train_labels=np.array(kept_train_labels, dtype=str),
        train_paths=np.array([p.relative_to(root).as_posix() for p in kept_train_paths], dtype=str),
        test_paths=np.array(valid_test_paths, dtype=str),
        test_labels=np.array(valid_test_labels, dtype=str),
        descriptor=settings.descriptor,
        n_keypoints=settings.n_keypoints,
        fast_threshold=settings.fast_threshold,
        n_octaves=settings.n_octaves,
        n_scales=settings.n_scales,
        sigma_min=settings.sigma_min,
        max_side=settings.max_side,
        root=root,
        n_descriptors=len(descriptors),
    )
    return model, sample_visuals, failed


def save_bovw_model(model: BoVWModel, path: Path) -> None:
    """Save the BoVW codebook, visual-word patches, and training histograms to a compressed ``.npz`` file."""
    global _LAST_MODEL_PATH
    np.savez_compressed(
        path,
        centroids=model.centroids,
        word_patches=model.word_patches,
        train_histograms=model.train_histograms,
        train_labels=model.train_labels,
        train_paths=model.train_paths,
        test_paths=model.test_paths,
        test_labels=model.test_labels,
        descriptor=np.array(model.descriptor),
        n_keypoints=np.array(model.n_keypoints),
        fast_threshold=np.array(model.fast_threshold),
        n_octaves=np.array(model.n_octaves),
        n_scales=np.array(model.n_scales),
        sigma_min=np.array(model.sigma_min),
        max_side=np.array(model.max_side),
        n_descriptors=np.array(model.n_descriptors),
    )
    _LAST_MODEL_PATH = Path(path)


def load_bovw_model(path: Path) -> BoVWModel:
    """Load a BoVW model written by :func:`save_bovw_model`."""
    global _LAST_MODEL_PATH
    try:
        with np.load(path, allow_pickle=False) as data:
            model = BoVWModel(
                centroids=np.asarray(data["centroids"], dtype=np.float64),
                word_patches=np.asarray(data["word_patches"], dtype=np.uint8),
                train_histograms=np.asarray(data["train_histograms"], dtype=np.float64),
                train_labels=np.asarray(data["train_labels"], dtype=str),
                train_paths=np.asarray(data["train_paths"], dtype=str),
                test_paths=np.asarray(data["test_paths"], dtype=str),
                test_labels=np.asarray(data["test_labels"], dtype=str),
                descriptor=str(data["descriptor"]),
                n_keypoints=int(data["n_keypoints"]),
                fast_threshold=float(data["fast_threshold"]),
                n_octaves=int(data["n_octaves"]),
                n_scales=int(data["n_scales"]),
                sigma_min=float(data["sigma_min"]),
                max_side=int(data["max_side"]),
                root=Path(path).parent,
                n_descriptors=int(data["n_descriptors"]) if "n_descriptors" in data.files else 0,
            )
            _LAST_MODEL_PATH = Path(path)
            return model
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"{Path(path).name} is not a valid BoVW model: {error}") from None


def find_bovw_model(image_path: Path | None = None) -> Path | None:
    """Find a saved ``bovw_model.npz`` near ``image_path``, in the last used folder, or in ``DEFAULT_BOVW_DATASET``."""
    candidates: list[Path] = []
    if image_path is not None:
        candidates += [parent / BOVW_MODEL_FILENAME for parent in list(Path(image_path).parents)[:MODEL_SEARCH_DEPTH]]
    if _LAST_MODEL_PATH is not None:
        candidates.append(_LAST_MODEL_PATH)
    candidates += [DEFAULT_BOVW_DATASET / BOVW_MODEL_FILENAME, CLASS_BOVW_DATASET / BOVW_MODEL_FILENAME]
    return next((path for path in candidates if path.is_file()), None)


# ---------------------------------------------------------------- 7.3 Classifier training & prediction


def fit_classifier(model: BoVWModel, kind: str = "svm_rbf") -> tuple[Any, StandardScaler, list[str]]:
    """Fit a probabilistic classifier on ``model.train_histograms`` and ``model.train_labels``."""
    if kind not in CLASSIFIERS:
        raise ValueError(f"Classifier must be one of {', '.join(CLASSIFIERS)}.")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(model.train_histograms)
    y_train = model.train_labels

    if kind == "svm_rbf":
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            clf: Any = SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)
            clf.fit(x_train, y_train)
        return clf, scaler, [str(c) for c in clf.classes_]
    if kind == "logistic":
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    elif kind == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    else:
        n_neighbors = min(5, len(x_train))
        clf = KNeighborsClassifier(n_neighbors=max(1, n_neighbors), weights="distance")

    clf.fit(x_train, y_train)
    classes = [str(c) for c in clf.classes_]
    return clf, scaler, classes


def predict_image_probabilities(
    image: Image.Image,
    model: BoVWModel,
    kind: str = "svm_rbf",
    trained: tuple[Any, StandardScaler, list[str]] | None = None,
) -> tuple[str, float, dict[str, float], np.ndarray]:
    """Extract features of ``image``, build its visual-word histogram, and return ``(best_class, best_prob, probs_by_class, histogram)``.

    ``trained`` is a classifier already returned by :func:`fit_classifier`; otherwise one is trained here.
    """
    features = extract_features(image, model.matching_settings())
    histogram = build_histogram(features.descriptors, model.centroids, normalize=True)
    clf, scaler, classes = trained if trained is not None else fit_classifier(model, kind)
    probs = clf.predict_proba(scaler.transform(histogram.reshape(1, -1)))[0]
    prob_map = {cls: float(prob) for cls, prob in zip(classes, probs)}
    best_idx = int(np.argmax(probs))
    return classes[best_idx], float(probs[best_idx]), prob_map, histogram


# ---------------------------------------------------------------- Figures (7.1, 7.2, 7.3)


def visual_words_figure(model: BoVWModel, max_words: int = MAX_DISPLAY_WORDS, columns: int = 5) -> Figure:
    """Grid of visual word patches ``vw 0 .. vw K-1`` (as in Slide 7.1)."""
    count = min(model.n_words, max_words)
    rows, cols = grid_shape(count, columns=columns)
    figure = Figure(figsize=(1.5 * cols, 1.65 * rows), layout="constrained")
    for idx in range(count):
        ax = figure.add_subplot(rows, cols, idx + 1)
        ax.imshow(model.word_patches[idx], cmap="gray")
        ax.set_title(f"vw {idx}", size=9)
        ax.set_xticks(())
        ax.set_yticks(())
    return figure


def sample_keypoints_figure(samples: list[tuple[np.ndarray, np.ndarray, str]]) -> Figure:
    """Grid of training images with detected keypoints in red (top-right of Slide 7.1)."""
    count = len(samples)
    rows, cols = grid_shape(count, columns=5)
    figure = Figure(figsize=(2.2 * cols, 2.2 * rows), layout="constrained")
    for idx, (gray, keypoints, label) in enumerate(samples):
        ax = figure.add_subplot(rows, cols, idx + 1)
        ax.imshow(gray, cmap="gray")
        if len(keypoints) > MAX_SAMPLE_KEYPOINTS:  # SIFT finds hundreds; draw a readable subset (K-Means used all)
            keypoints = keypoints[np.random.default_rng(RANDOM_SEED).choice(len(keypoints), MAX_SAMPLE_KEYPOINTS, replace=False)]
        if len(keypoints):
            ax.scatter(keypoints[:, 1], keypoints[:, 0], s=10, color="red")  # filled red dots, as on the slide
        ax.set_title(label, size=9)
        ax.set_xticks(())
        ax.set_yticks(())
    return figure


def word_colour(word_id: int, n_words: int) -> tuple[float, float, float, float]:
    """A distinct colour per visual word, so keypoints of the same word share a colour."""
    return colormaps["hsv"](word_id / max(n_words, 1))


def draw_word_thumbnails(ax: Any, model: BoVWModel, hist: np.ndarray) -> None:
    """Small visual-word patch under every non-empty bar (the thumbnails at the foot of the bars in Slides 7.1/7.2)."""
    zoom = min(0.5, 9.0 / max(model.n_words, 1))
    for word_id in np.nonzero(hist)[0]:
        thumbnail = OffsetImage(model.word_patches[int(word_id)], cmap="gray", zoom=zoom)
        ax.add_artist(
            AnnotationBbox(
                thumbnail,
                (float(word_id), 0.0),
                xybox=(0, 6),
                boxcoords="offset points",
                frameon=False,
                box_alignment=(0.5, 0.0),
            )
        )


def histogram_rows_figure(
    items: list[tuple[Image.Image, str]],
    model: BoVWModel,
    connect_keypoints: bool = True,
) -> Figure:
    """Build the Slide 7.2 figure: each row shows a test image on the left and its visual-word histogram on the right."""
    n_rows = max(1, len(items))
    k = model.n_words
    figure = Figure(figsize=(11.0, 3.2 * n_rows), layout="constrained")
    axes = np.atleast_2d(figure.subplots(n_rows, 2, gridspec_kw={"width_ratios": [1.0, 1.45]}))

    match_settings = model.matching_settings()

    for row_idx, (image, title) in enumerate(items):
        ax_img, ax_hist = axes[row_idx, 0], axes[row_idx, 1]
        features = extract_features(image, match_settings)
        words = assign_visual_words(features.descriptors, model.centroids)
        hist = build_histogram(features.descriptors, model.centroids, normalize=True)

        ax_img.imshow(features.gray, cmap="gray")
        ax_img.set_title(title, size=10)
        ax_img.set_xticks(())
        ax_img.set_yticks(())

        ax_hist.bar(
            np.arange(k),
            hist,
            width=0.8,
            color="lightgray",
            edgecolor="dimgray",
            linewidth=0.8,
        )
        ax_hist.set_title("Histogram of visual words", size=10)
        ax_hist.set_xlabel("Visual word index", size=9)
        ax_hist.set_ylabel("Normalized frequency", size=9)
        ax_hist.set_xlim(-0.8, k - 0.2)
        upper = max(float(hist.max()) * 1.15, 0.05)
        ax_hist.set_ylim(0.0, upper)
        ax_hist.grid(True, axis="y", linestyle=":", alpha=0.6)
        draw_word_thumbnails(ax_hist, model, hist)

        # Slide 7.2: every keypoint, coloured by its visual word, linked by a thin line to the top of that word's bar
        if connect_keypoints and len(words) > 0:
            shown = np.arange(len(words))
            if len(shown) > MAX_CONNECTED_KEYPOINTS:  # keep the figure readable for SIFT's hundreds of keypoints
                shown = np.random.default_rng(RANDOM_SEED).choice(shown, MAX_CONNECTED_KEYPOINTS, replace=False)
            for kp_index in shown:
                word_id = int(words[kp_index])
                kp_row, kp_col = features.keypoints[int(kp_index)]
                ax_img.plot(kp_col, kp_row, marker="o", markersize=3.5, color=word_colour(word_id, k), zorder=3)
                figure.add_artist(
                    ConnectionPatch(
                        xyA=(kp_col, kp_row),
                        coordsA=ax_img.transData,
                        xyB=(float(word_id), float(hist[word_id])),
                        coordsB=ax_hist.transData,
                        color="gray",
                        alpha=0.3,
                        linewidth=0.5,
                    )
                )
        elif len(features.keypoints) > 0:
            ax_img.scatter(
                features.keypoints[:, 1],
                features.keypoints[:, 0],
                s=12,
                facecolors="none",
                edgecolors="r",
                linewidths=0.8,
            )

    return figure


def classification_result_figure(
    image: Image.Image,
    predicted_class: str,
    probability: float,
) -> Figure:
    """Build the Slide 7.3 result figure: the image titled with ``p(predicted_class) = probability``."""
    figure = Figure(figsize=(5.5, 4.5), layout="constrained")
    ax = figure.add_subplot(1, 1, 1)
    ax.imshow(np.asarray(image.convert("RGB")))
    ax.set_title(f"p({predicted_class}) = {probability:.2f}", size=14, pad=10)
    ax.set_xticks(())
    ax.set_yticks(())
    return figure


def select_test_images_from_model(model: BoVWModel, max_images: int = 3) -> list[tuple[Image.Image, str]]:
    """Load one test image per class (up to ``max_images``) from ``model.root`` for Slide 7.2 visualization."""
    if model.root is None:
        return []
    paths = list(model.test_paths) if len(model.test_paths) > 0 else list(model.train_paths)
    labels = list(model.test_labels) if len(model.test_labels) > 0 else list(model.train_labels)

    items: list[tuple[Image.Image, str]] = []
    seen_classes: set[str] = set()
    # First pass: one image per distinct class (like butterfly, kangaroo, dalmatian on Slide 7.2)
    for rel_path, label in zip(paths, labels):
        if label in seen_classes:
            continue
        full_path = model.root / str(rel_path)
        try:
            with Image.open(full_path) as opened:
                items.append((opened.copy(), str(label)))
                seen_classes.add(str(label))
        except (OSError, UnidentifiedImageError):
            continue
        if len(items) >= max_images:
            return items

    # Second pass if max_images > number of classes
    for rel_path, label in zip(paths, labels):
        if len(items) >= max_images:
            break
        full_path = model.root / str(rel_path)
        try:
            with Image.open(full_path) as opened:
                cand = (opened.copy(), str(label))
        except (OSError, UnidentifiedImageError):
            continue
        items.append(cand)
    return items


# ---------------------------------------------------------------- Dialogs & Tools


class CodebookDialog(simpledialog.Dialog):
    """Dialog for Set 7.1: Codebook construction (K-Means on SIFT/ORB descriptors)."""

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        defaults: CodebookSettings = CodebookSettings(),
    ) -> None:
        self.defaults = defaults
        super().__init__(parent, title=title)

    def body(self, master: tk.Frame) -> tk.Widget:
        d = self.defaults
        self.folder = tk.StringVar(value=d.folder)
        self.descriptor = tk.StringVar(value=d.descriptor)
        self.words_per_class = tk.StringVar(value=str(d.words_per_class))
        self.n_clusters_override = tk.StringVar(value=str(d.n_clusters_override))
        self.n_keypoints = tk.StringVar(value=str(d.n_keypoints))
        self.fast_threshold = tk.StringVar(value=str(d.fast_threshold))
        self.n_octaves = tk.StringVar(value=str(d.n_octaves))
        self.n_scales = tk.StringVar(value=str(d.n_scales))
        self.sigma_min = tk.StringVar(value=str(d.sigma_min))
        self.max_side = tk.StringVar(value=str(d.max_side))
        self.show_keypoints_popup = tk.BooleanVar(value=d.show_keypoints_popup)

        data = ttk.LabelFrame(master, text="Dataset folder (one subfolder per class, e.g. butterfly, kangaroo, dalmatian)")
        data.grid(row=0, column=0, columnspan=2, sticky="we")
        ttk.Entry(data, textvariable=self.folder, width=52).grid(row=0, column=0, padx=6, pady=2)
        ttk.Button(data, text="Browse…", command=self._browse).grid(row=0, column=1, padx=(0, 6), pady=2)

        km = ttk.LabelFrame(master, text="K-Means Codebook (MiniBatchKMeans)")
        km.grid(row=1, column=0, columnspan=2, sticky="we", pady=(6, 0))
        first = spin_row(km, 0, "Visual words per class (K = N × num_classes):", self.words_per_class, WORDS_PER_CLASS_RANGE, 5)
        spin_row(km, 1, "Exact K override (0 = use 10 × num_classes):", self.n_clusters_override, CLUSTERS_OVERRIDE_RANGE, 10)
        ttk.Checkbutton(
            km,
            text="Also show sample training images with keypoints (popup)",
            variable=self.show_keypoints_popup,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=6)

        desc_frame = ttk.LabelFrame(master, text="Descriptor extractor")
        desc_frame.grid(row=2, column=0, columnspan=2, sticky="we", pady=(6, 0))
        for r_offset, (key, (text, _metric)) in enumerate(DESCRIPTORS.items()):
            ttk.Radiobutton(
                desc_frame, text=text, value=key, variable=self.descriptor, command=self._toggle
            ).grid(row=r_offset, column=0, columnspan=2, sticky="w", padx=6)

        self.orb_frame = ttk.LabelFrame(master, text="ORB parameters")
        self.sift_frame = ttk.LabelFrame(master, text="SIFT parameters")
        self.orb_frame.grid(row=3, column=0, columnspan=2, sticky="we", pady=(6, 0))
        self.sift_frame.grid(row=3, column=0, columnspan=2, sticky="we", pady=(6, 0))
        spin_row(self.orb_frame, 0, "Number of keypoints:", self.n_keypoints, N_KEYPOINTS_RANGE, 50)
        spin_row(self.orb_frame, 1, "FAST threshold:", self.fast_threshold, FAST_THRESHOLD_RANGE, 0.01)
        spin_row(self.sift_frame, 0, "Number of octaves:", self.n_octaves, N_OCTAVES_RANGE, 1)
        spin_row(self.sift_frame, 1, "Scales per octave:", self.n_scales, N_SCALES_RANGE, 1)
        spin_row(self.sift_frame, 2, "Sigma min:", self.sigma_min, SIGMA_RANGE, 0.1)

        prep = ttk.LabelFrame(master, text="Preprocessing")
        prep.grid(row=4, column=0, columnspan=2, sticky="we", pady=(6, 0))
        spin_row(prep, 0, "Downscale to max side (px):", self.max_side, MAX_SIDE_RANGE, 50)

        ttk.Label(master, text=f"The codebook & training histograms are saved as {BOVW_MODEL_FILENAME} in the dataset folder.").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        self._toggle()
        return first

    def _browse(self) -> None:
        directory = filedialog.askdirectory(title="Select dataset folder", parent=self, mustexist=True)
        if directory:
            self.folder.set(directory)

    def _toggle(self) -> None:
        orb = self.descriptor.get() == "orb"
        (self.sift_frame if orb else self.orb_frame).grid_remove()
        (self.orb_frame if orb else self.sift_frame).grid()

    def validate(self) -> bool:
        try:
            folder = self.folder.get().strip()
            if not folder or not Path(folder).is_dir():
                raise ValueError("Choose an existing dataset folder with class subfolders.")
            self._settings = CodebookSettings(
                folder=folder,
                descriptor=self.descriptor.get(),
                words_per_class=read_number(self.words_per_class, "Visual words per class", WORDS_PER_CLASS_RANGE, int),
                n_clusters_override=read_number(self.n_clusters_override, "Exact K override", CLUSTERS_OVERRIDE_RANGE, int),
                n_keypoints=read_number(self.n_keypoints, "Number of keypoints", N_KEYPOINTS_RANGE, int),
                fast_threshold=read_number(self.fast_threshold, "FAST threshold", FAST_THRESHOLD_RANGE),
                n_octaves=read_number(self.n_octaves, "Number of octaves", N_OCTAVES_RANGE, int),
                n_scales=read_number(self.n_scales, "Scales per octave", N_SCALES_RANGE, int),
                sigma_min=read_number(self.sigma_min, "Sigma min", SIGMA_RANGE),
                max_side=read_number(self.max_side, "Max side", MAX_SIDE_RANGE, int),
                show_keypoints_popup=self.show_keypoints_popup.get(),
            )
        except ValueError as error:
            messagebox.showerror(self.title(), str(error), parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self._settings


class CodebookBuilderTool(ForensicsTool):
    tool_id = "bovw_codebook"
    title = "Codebook construction (BoVW)"
    category = "Set 7: Bag of Visual Words"
    description = (
        "Extract SIFT/ORB descriptors across a training dataset, cluster them with MiniBatchKMeans "
        "(K = 10 × num_classes), save the codebook model, and visualize the visual words (vw 0 … vw K-1)."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        default_folder = str(_LAST_MODEL_PATH.parent) if _LAST_MODEL_PATH else str(default_bovw_dataset())
        settings = CodebookDialog(
            parent,
            title="Codebook construction (Set 7.1)",
            defaults=CodebookSettings(folder=default_folder),
        ).result
        if settings is None:
            return None

        window = parent.winfo_toplevel()
        previous_title = window.title()
        window.configure(cursor="watch")
        try:
            with ProgressPopup(parent, "Codebook construction", "Extracting descriptors…") as popup:

                def progress(index: int, total: int, path: Path) -> None:
                    window.title(f"Extracting descriptors {index + 1}/{total}: {path.name}")
                    popup.step(index, total, f"Extracting {settings.descriptor.upper()} descriptors: {path.parent.name}/{path.name}")

                model, sample_visuals, failed = build_codebook(Path(settings.folder), settings, progress=progress)
                assert model.root is not None
                root = model.root  # the folder that holds the classes (a lone wrapper folder is skipped)
                model_path = root / BOVW_MODEL_FILENAME
                popup.run(f"Saving {BOVW_MODEL_FILENAME}…", lambda: save_bovw_model(model, model_path))
        finally:
            window.configure(cursor="")
            window.title(previous_title)

        class_counts = Counter(str(lbl) for lbl in model.train_labels)
        details: dict[str, Any] = {
            "Operation": "BoVW Codebook construction",
            "Library": f"sklearn.cluster.MiniBatchKMeans, skimage.feature.{model.descriptor.upper()}",
            "Dataset folder": str(root),
            "Classes": f"{len(model.classes)} ({', '.join(model.classes)})",
            "Training images": len(model.train_labels),
            "Test images": len(model.test_labels),
            "Descriptor": model.descriptor.upper(),
            "Visual words (K)": f"{model.n_words} (10 × {len(model.classes)} classes)"
            if settings.n_clusters_override == 0 and settings.words_per_class == 10
            else str(model.n_words),
            "Stacked descriptors (np.vstack)": f"{model.n_descriptors} × {model.centroids.shape[1]}",
            "K-Means": f"MiniBatchKMeans(n_clusters={model.n_words}, random_state={RANDOM_SEED})",
            "Centroid shape": f"{model.centroids.shape[0]} × {model.centroids.shape[1]}",
            "Model saved to": str(model_path),
        }
        for cls_name, count in class_counts.most_common():
            details[f"  Class '{cls_name}'"] = f"{count} train image(s)"
        if failed:
            details["Unreadable files"] = ", ".join(p.name for p in failed)

        if settings.show_keypoints_popup and sample_visuals:
            show_figure(parent, "Training images & detected keypoints", sample_keypoints_figure(sample_visuals))

        gallery_img = figure_to_image(visual_words_figure(model))
        return ToolResult(
            image=gallery_img,
            message=(
                f"Built codebook of K={model.n_words} visual words across {len(model.classes)} classes "
                f"({len(model.train_labels)} training images); saved {BOVW_MODEL_FILENAME}."
            ),
            details=details,
        )


class HistogramDialog(simpledialog.Dialog):
    """Dialog for Set 7.2: Visual-word histogram construction and visualization."""

    def __init__(self, parent: tk.Misc, model_path: Path | None, has_image: bool) -> None:
        self.found = model_path
        self.has_image = has_image
        super().__init__(parent, title="Histogram construction (Set 7.2)")

    def body(self, master: tk.Frame) -> tk.Widget:
        d = HistogramSettings(mode="test_set" if not self.has_image else "test_set")
        self.model_path = tk.StringVar(value=str(self.found) if self.found else "")
        self.mode = tk.StringVar(value=d.mode)
        self.max_images = tk.StringVar(value=str(d.max_images))
        self.connect_keypoints = tk.BooleanVar(value=d.connect_keypoints)

        model_frame = ttk.LabelFrame(master, text=f"Saved codebook model ({BOVW_MODEL_FILENAME} from Set 7.1)")
        model_frame.grid(row=0, column=0, columnspan=2, sticky="we")
        entry = ttk.Entry(model_frame, textvariable=self.model_path, width=50)
        entry.grid(row=0, column=0, padx=6, pady=2)
        ttk.Button(model_frame, text="Browse…", command=self._browse).grid(row=0, column=1, padx=(0, 6))

        mode_frame = ttk.LabelFrame(master, text="Visualize histogram for")
        mode_frame.grid(row=1, column=0, columnspan=2, sticky="we", pady=(6, 0))
        for row, (key, text) in enumerate(HISTOGRAM_MODES.items()):
            btn = ttk.Radiobutton(mode_frame, text=text, value=key, variable=self.mode)
            btn.grid(row=row, column=0, columnspan=2, sticky="w", padx=6)
            if key == "current" and not self.has_image:
                btn.configure(state="disabled")

        opts = ttk.LabelFrame(master, text="Display options")
        opts.grid(row=2, column=0, columnspan=2, sticky="we", pady=(6, 0))
        spin_row(opts, 0, "Test images to show (1–6):", self.max_images, (1, MAX_TEST_HISTOGRAMS), 1)
        ttk.Checkbutton(
            opts,
            text="Draw lines connecting keypoints to their visual word bar (as in slide 7.2)",
            variable=self.connect_keypoints,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=6)
        return entry

    def _browse(self) -> None:
        filename = filedialog.askopenfilename(title="Select BoVW model", filetypes=MODEL_TYPES, parent=self)
        if filename:
            self.model_path.set(filename)

    def validate(self) -> bool:
        try:
            path = self.model_path.get().strip()
            if not path or not Path(path).is_file():
                raise ValueError("Choose a saved BoVW model file (run Set 7.1 Codebook construction first).")
            self._settings = HistogramSettings(
                model_path=path,
                mode=self.mode.get(),
                max_images=read_number(self.max_images, "Test images to show", (1, MAX_TEST_HISTOGRAMS), int),
                connect_keypoints=self.connect_keypoints.get(),
            )
        except ValueError as error:
            messagebox.showerror(self.title(), str(error), parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self._settings


class VisualWordHistogramTool(ForensicsTool):
    tool_id = "bovw_histogram"
    title = "Histogram construction (BoVW)"
    category = "Set 7: Bag of Visual Words"
    description = (
        "Map image descriptors to their closest visual words, build a K-dimensional normalized "
        "frequency histogram, and visualize the histogram for test images."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        settings = HistogramDialog(
            parent,
            find_bovw_model(document.path),
            has_image=document.current is not None,
        ).result
        if settings is None:
            return None

        model = load_bovw_model(Path(settings.model_path))
        if settings.mode == "current" and document.current is not None:
            label = document.path.stem if document.path else "current image"
            items = [(document.current, label)]
        else:
            items = select_test_images_from_model(model, settings.max_images)
            if not items and document.current is not None:
                items = [(document.current, document.path.stem if document.path else "current image")]
            if not items:
                return ToolResult(message="No test images found for this model.")

        figure = histogram_rows_figure(items, model, connect_keypoints=settings.connect_keypoints)
        out_image = figure_to_image(figure)

        details: dict[str, Any] = {
            "Operation": "Visual word histogram construction (Set 7.2)",
            "Model": settings.model_path,
            "Visual words (K)": model.n_words,
            "Descriptor": model.descriptor.upper(),
            "Mode": HISTOGRAM_MODES[settings.mode],
            "Images visualized": len(items),
        }
        for idx, (img, title) in enumerate(items, start=1):
            feats = extract_features(img, model.matching_settings())
            hist = build_histogram(feats.descriptors, model.centroids, normalize=True)
            top_word = int(np.argmax(hist)) if len(hist) else 0
            details[f"Image #{idx} ({title})"] = (
                f"{len(feats.keypoints)} keypoints, peak visual word = vw {top_word} ({hist[top_word]:.2f})"
            )

        return ToolResult(
            image=out_image,
            message=(
                f"Built and visualized K={model.n_words} visual-word histogram(s) for "
                f"{len(items)} image(s) ({', '.join(t for _, t in items)})."
            ),
            details=details,
        )


class InteractiveClassifierPopup:
    """Interactive Set 7.3 window: first shows the 7.2 histogram, then runs the classifier when 'Classify' is clicked."""

    def __init__(
        self,
        parent: tk.Misc,
        image: Image.Image,
        image_title: str,
        model: BoVWModel,
        classifier_kind: str,
    ) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.image = image
        self.image_title = image_title
        self.model = model
        self.classifier_kind = classifier_kind
        self.classified = False
        self.predicted_class = ""
        self.top_probability = 0.0
        self.probabilities: dict[str, float] = {}
        self.trained = fit_classifier(model, classifier_kind)  # trained once on the 7.1 training histograms

        self.window = tk.Toplevel(parent)
        self.window.title("Set 7.3 Classification — Visualize histogram & Classify")
        self.window.geometry("980x480")

        top_bar = ttk.Frame(self.window, padding=(10, 8))
        top_bar.pack(fill="x")
        self.status_var = tk.StringVar(
            value="Step 1: Visual-word histogram (from 7.2). Click 'Classify' to run the trained classifier."
        )
        ttk.Label(top_bar, textvariable=self.status_var, font=("TkDefaultFont", 10, "bold")).pack(side="left")
        self.classify_btn = ttk.Button(top_bar, text="Classify", command=self.run_classify)
        self.classify_btn.pack(side="right", padx=(8, 0))

        self.figure = histogram_rows_figure([(image, image_title)], model, connect_keypoints=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def run_classify(self) -> None:
        pred_cls, best_prob, prob_map, _hist = predict_image_probabilities(
            self.image, self.model, self.classifier_kind, self.trained
        )
        self.classified = True
        self.predicted_class = pred_cls
        self.top_probability = best_prob
        self.probabilities = prob_map

        # Update the left subplot to show the colour image with title "p(class) = 0.64" as in Slide 7.3
        ax_img = self.figure.axes[0]
        ax_img.clear()
        ax_img.imshow(np.asarray(self.image.convert("RGB")))
        ax_img.set_title(f"p({pred_cls}) = {best_prob:.2f}", size=13, fontweight="bold")
        ax_img.set_xticks(())
        ax_img.set_yticks(())
        self.canvas.draw_idle()

        probs_text = ", ".join(f"p({c})={p:.2f}" for c, p in sorted(prob_map.items(), key=lambda x: x[1], reverse=True))
        self.status_var.set(f"Classified → p({pred_cls}) = {best_prob:.2f}   [{probs_text}]")


class ClassificationDialog(simpledialog.Dialog):
    """Dialog for Set 7.3: BoVW classification."""

    def __init__(self, parent: tk.Misc, model_path: Path | None) -> None:
        self.found = model_path
        super().__init__(parent, title="Image classification (Set 7.3)")

    def body(self, master: tk.Frame) -> tk.Widget:
        d = ClassificationSettings()
        self.model_path = tk.StringVar(value=str(self.found) if self.found else "")
        self.classifier = tk.StringVar(value=d.classifier)
        self.mode = tk.StringVar(value=d.mode)
        self.n_gallery = tk.StringVar(value=str(d.n_gallery))

        model_frame = ttk.LabelFrame(master, text=f"Saved codebook model ({BOVW_MODEL_FILENAME} from Set 7.1)")
        model_frame.grid(row=0, column=0, columnspan=2, sticky="we")
        entry = ttk.Entry(model_frame, textvariable=self.model_path, width=50)
        entry.grid(row=0, column=0, padx=6, pady=2)
        ttk.Button(model_frame, text="Browse…", command=self._browse).grid(row=0, column=1, padx=(0, 6))

        clf_frame = ttk.LabelFrame(master, text="Classifier")
        clf_frame.grid(row=1, column=0, columnspan=2, sticky="we", pady=(6, 0))
        for row, (key, label) in enumerate(CLASSIFIERS.items()):
            ttk.Radiobutton(clf_frame, text=label, value=key, variable=self.classifier).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=6
            )

        mode_frame = ttk.LabelFrame(master, text="Mode")
        mode_frame.grid(row=2, column=0, columnspan=2, sticky="we", pady=(6, 0))
        for row, (key, label) in enumerate(CLASSIFY_MODES.items()):
            ttk.Radiobutton(mode_frame, text=label, value=key, variable=self.mode).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=6
            )

        opts = ttk.LabelFrame(master, text="Evaluation options")
        opts.grid(row=3, column=0, columnspan=2, sticky="we", pady=(6, 0))
        spin_row(opts, 0, "Test evaluation gallery size:", self.n_gallery, (1, 36), 3)
        return entry

    def _browse(self) -> None:
        filename = filedialog.askopenfilename(title="Select BoVW model", filetypes=MODEL_TYPES, parent=self)
        if filename:
            self.model_path.set(filename)

    def validate(self) -> bool:
        try:
            path = self.model_path.get().strip()
            if not path or not Path(path).is_file():
                raise ValueError("Choose a saved BoVW model file (run Set 7.1 Codebook construction first).")
            self._settings = ClassificationSettings(
                model_path=path,
                classifier=self.classifier.get(),
                mode=self.mode.get(),
                n_gallery=read_number(self.n_gallery, "Gallery size", (1, 36), int),
            )
        except ValueError as error:
            messagebox.showerror(self.title(), str(error), parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self._settings


class BoVWClassificationTool(ForensicsTool):
    tool_id = "bovw_classification"
    title = "Image classification (BoVW)"
    category = "Set 7: Bag of Visual Words"
    description = (
        "Given a new image, visualize its visual-word histogram (from 7.2) and run the trained "
        "BoVW classifier to show the top class probability, e.g. p(kangaroo) = 0.64."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by main window

        settings = ClassificationDialog(parent, find_bovw_model(document.path)).result
        if settings is None:
            return None

        model = load_bovw_model(Path(settings.model_path))
        if settings.mode == "evaluate":
            return self._evaluate_test_set(model, settings)

        img_title = document.path.stem if document.path else "query"
        details: dict[str, Any] = {
            "Operation": "BoVW Image Classification (Set 7.3)",
            "Model": settings.model_path,
            "Classifier": CLASSIFIERS[settings.classifier],
            "Visual words (K)": model.n_words,
        }

        if settings.mode == "interactive":
            # Slide 7.3: show the 7.2 histogram first; the prediction only appears after clicking "Classify"
            popup = InteractiveClassifierPopup(parent, document.current, img_title, model, settings.classifier)
            parent.winfo_toplevel()._bovw_popup = popup  # keep reference alive
            return ToolResult(
                image=figure_to_image(popup.figure),
                message="Histogram of visual words shown — click 'Classify' in the popup to run the classifier.",
                details=details,
            )

        pred_cls, best_prob, prob_map, _hist = predict_image_probabilities(
            document.current, model, settings.classifier
        )
        details |= {
            "Predicted class": pred_cls,
            "Top probability": f"p({pred_cls}) = {best_prob:.2f}",
        }
        for cls_name, prob in sorted(prob_map.items(), key=lambda item: item[1], reverse=True):
            details[f"  p({cls_name})"] = f"{prob:.4f}"

        canvas_img = figure_to_image(classification_result_figure(document.current, pred_cls, best_prob))
        return ToolResult(
            image=canvas_img,
            message=f"p({pred_cls}) = {best_prob:.2f}",
            details=details,
        )

    @staticmethod
    def _evaluate_test_set(model: BoVWModel, settings: ClassificationSettings) -> ToolResult:
        if model.root is None or len(model.test_paths) == 0:
            raise ValueError("This model has no test split to evaluate.")

        clf, scaler, classes = fit_classifier(model, settings.classifier)
        images: list[np.ndarray] = []
        truths: list[str] = []
        preds: list[str] = []
        top_probs: list[float] = []

        for rel_path, truth in zip(model.test_paths, model.test_labels):
            full_path = model.root / str(rel_path)
            try:
                with Image.open(full_path) as opened:
                    img = opened.copy()
            except (OSError, UnidentifiedImageError):
                continue
            feats = extract_features(img, model.matching_settings())
            hist = build_histogram(feats.descriptors, model.centroids, normalize=True)
            probs = clf.predict_proba(scaler.transform(hist.reshape(1, -1)))[0]
            best_idx = int(np.argmax(probs))
            images.append(np.asarray(img.convert("RGB").resize((120, 120), Image.Resampling.LANCZOS)))
            truths.append(str(truth))
            preds.append(classes[best_idx])
            top_probs.append(float(probs[best_idx]))

        if not images:
            return ToolResult(message="Could not open any test images for evaluation.")

        correct = [p == t for p, t in zip(preds, truths)]
        accuracy = float(np.mean(correct))

        count = min(settings.n_gallery, len(images))
        rows, cols = grid_shape(count, columns=3)
        figure = Figure(figsize=(2.8 * cols, 2.8 * rows), layout="constrained")
        for i in range(count):
            ax = figure.add_subplot(rows, cols, i + 1)
            ax.imshow(images[i])
            colour = "tab:green" if correct[i] else "tab:red"
            ax.set_title(f"p({preds[i]}) = {top_probs[i]:.2f}\ntrue: {truths[i]}", size=9, color=colour)
            ax.set_xticks(())
            ax.set_yticks(())

        details: dict[str, Any] = {
            "Operation": "BoVW Test Set Evaluation",
            "Classifier": CLASSIFIERS[settings.classifier],
            "Visual words (K)": model.n_words,
            "Test images": len(images),
            "Accuracy": f"{accuracy:.1%} ({sum(correct)}/{len(images)})",
        }
        return ToolResult(
            image=figure_to_image(figure),
            message=f"BoVW test accuracy: {accuracy:.1%} ({sum(correct)}/{len(images)}).",
            details=details,
        )


__all__ = [
    "BOVW_MODEL_FILENAME",
    "BoVWClassificationTool",
    "BoVWModel",
    "CLASSIFIERS",
    "CLASSIFY_MODES",
    "ClassificationDialog",
    "ClassificationSettings",
    "CodebookBuilderTool",
    "CodebookDialog",
    "CodebookSettings",
    "DEFAULT_BOVW_DATASET",
    "HISTOGRAM_MODES",
    "HistogramDialog",
    "HistogramSettings",
    "InteractiveClassifierPopup",
    "PATCH_SIZE",
    "VisualWordHistogramTool",
    "WORDS_PER_CLASS_DEFAULT",
    "assign_visual_words",
    "build_codebook",
    "build_histogram",
    "classification_result_figure",
    "crop_patch",
    "discover_splits",
    "find_bovw_model",
    "fit_classifier",
    "histogram_rows_figure",
    "load_bovw_model",
    "predict_image_probabilities",
    "sample_keypoints_figure",
    "save_bovw_model",
    "select_test_images_from_model",
    "visual_words_figure",
]
