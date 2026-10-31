"""Set 6.2: Eigenface builder — PCA on a folder of face images, saved as a model and shown as eigenfaces."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import csv
from dataclasses import dataclass, replace
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, UnidentifiedImageError

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialog_fields import read_number, spin_row
from .hog import show_figure
from .image_lookup import list_images
from .modes import split_alpha

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "lfw_faces"  # datasets/ is git-ignored
MODEL_FILENAME = "eigenfaces_model.npz"
LABELS_FILENAME = "labels.csv"
TRAIN_DIR, TEST_DIR = "train", "test"

COMPONENTS_RANGE = (1, 1000)
DISPLAY_RANGE = (1, 64)
POV_TARGET = 0.9  # slides: "typically, stop at PoV > 0.9"

# fetch_lfw_people / train_test_split arguments, as in the tutorial notebook
LFW_MIN_FACES = 70
LFW_RESIZE = 0.4
LFW_TEST_SIZE = 0.25
LFW_SEED = 42


@dataclass(frozen=True)
class FaceSet:
    """Gray faces flattened to rows of ``data`` (values in [0, 1]) with their person names."""

    data: np.ndarray  # (N, h*w)
    labels: list[str]
    paths: list[Path]
    shape: tuple[int, int]  # (h, w)


@dataclass(frozen=True)
class EigenfaceModel:
    """PCA basis ``U_K`` (rows of ``eigenfaces``), the mean face and every training face's weights."""

    mean: np.ndarray  # (D,)
    eigenfaces: np.ndarray  # (K, D), orthonormal rows u_1 … u_K
    eigenvalues: np.ndarray  # (K,) variance along each eigenface, descending
    total_variance: float  # sum of all eigenvalues (for the PoV)
    coefficients: np.ndarray  # (N, K) w_i = U_K^T (x_i - mu)
    labels: np.ndarray  # (N,) person names
    paths: np.ndarray  # (N,) training image paths, relative to ``root``
    shape: tuple[int, int]
    root: Path | None = None  # dataset folder (holds train/ and test/)

    @property
    def n_components(self) -> int:
        return len(self.eigenfaces)

    @property
    def pov(self) -> np.ndarray:
        """Proportion of variance explained by the first 1, 2, … K eigenfaces."""
        return np.cumsum(self.eigenvalues) / self.total_variance if self.total_variance > 0 else np.zeros(0)

    def project(self, data: np.ndarray, k: int | None = None) -> np.ndarray:
        """``w = U_k^T (x - mu)`` for each row of ``data``."""
        return (np.atleast_2d(data) - self.mean) @ self.eigenfaces[: k or self.n_components].T

    def reconstruct(self, weights: np.ndarray) -> np.ndarray:
        """``x_hat = mu + sum_i w_i u_i`` with as many eigenfaces as ``weights`` has columns."""
        weights = np.atleast_2d(weights)
        return self.mean + weights @ self.eigenfaces[: weights.shape[1]]


# ---------------------------------------------------------------- dataset


def person_name(label: str) -> str:
    """Folder name -> display name (``George_W_Bush`` -> ``George W Bush``)."""
    return label.replace("_", " ")


def export_lfw(
    folder: Path,
    min_faces_per_person: int = LFW_MIN_FACES,
    resize: float = LFW_RESIZE,
    test_size: float = LFW_TEST_SIZE,
    seed: int = LFW_SEED,
) -> tuple[int, int, list[str]]:
    """Download LFW with ``fetch_lfw_people`` and save it as ``folder/{train,test}/<Person>/*.png``.

    The person is the image's folder name and ``labels.csv`` lists every image <-> name pair.
    Returns the train and test counts and the people.
    """
    from sklearn.datasets import fetch_lfw_people  # imported here: slow, and only needed for the download
    from sklearn.model_selection import train_test_split

    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize)
    images = lfw.images
    if images.max() <= 1.0:  # some scikit-learn versions return [0, 1], others [0, 255]
        images = images * 255.0
    images = np.clip(np.round(images), 0, 255).astype(np.uint8)
    names = [str(name) for name in lfw.target_names]
    indices = np.arange(len(images))
    # Stratified so every person appears in both splits
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, stratify=lfw.target)

    rows: list[tuple[str, str, str]] = []
    for split, chosen in ((TRAIN_DIR, train_idx), (TEST_DIR, test_idx)):
        counters: Counter[str] = Counter()
        for index in sorted(chosen):
            name = names[lfw.target[index]]
            label = name.replace(" ", "_")
            counters[label] += 1
            relative = Path(split) / label / f"{label}_{counters[label]:04d}.png"
            (folder / relative).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(images[index], mode="L").save(folder / relative)
            rows.append((split, relative.as_posix(), name))
    with open(folder / LABELS_FILENAME, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "path", "name"))
        writer.writerows(rows)
    return len(train_idx), len(test_idx), names


def split_folders(folder: Path) -> tuple[Path, Path | None]:
    """``(train, test)`` folders: ``folder/train`` and ``folder/test`` if present, else ``folder`` itself."""
    train = folder / TRAIN_DIR
    if train.is_dir():
        test = folder / TEST_DIR
        return train, test if test.is_dir() else None
    return folder, None


def label_for(path: Path, root: Path) -> str:
    """The person is the image's folder below ``root``; a loose file is labelled by its own name."""
    relative = path.relative_to(root)
    if len(relative.parts) > 1:
        return person_name(relative.parts[0])
    return person_name(path.stem.rstrip("0123456789_- ") or path.stem)


def face_vector(image: Image.Image, shape: tuple[int, int]) -> np.ndarray:
    """Gray, resized to ``shape`` (h, w), flattened, in [0, 1]."""
    base, _alpha = split_alpha(image)
    gray = base.convert("L")
    if gray.size != (shape[1], shape[0]):
        gray = gray.resize((shape[1], shape[0]), Image.Resampling.LANCZOS)
    return np.asarray(gray, dtype=np.float64).ravel() / 255.0


def load_faces(
    folder: Path,
    shape: tuple[int, int] | None = None,
    progress: Callable[[int, int, Path], None] | None = None,
) -> tuple[FaceSet, list[Path]]:
    """Every image under ``folder`` resized to ``shape`` (default: the first image's size).

    Returns the faces and the files that could not be read.
    """
    paths = list_images(folder)
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    kept: list[Path] = []
    failed: list[Path] = []
    for index, path in enumerate(paths):
        if progress is not None:
            progress(index, len(paths), path)
        try:
            with Image.open(path) as opened:
                image = opened.copy()
        except (OSError, UnidentifiedImageError):
            failed.append(path)
            continue
        if shape is None:
            shape = (image.height, image.width)
        vectors.append(face_vector(image, shape))
        labels.append(label_for(path, folder))
        kept.append(path)
    data = np.vstack(vectors) if vectors else np.zeros((0, 0))
    return FaceSet(data, labels, kept, shape or (0, 0)), failed


# ---------------------------------------------------------------- PCA


def fit_eigenfaces(faces: FaceSet, n_components: int, root: Path | None = None) -> EigenfaceModel:
    """PCA as in the slides: mean, centre, eigenvectors of the covariance sorted by eigenvalue, keep K.

    The covariance ``A A^T`` is d × d (1850² for LFW at 0.4), so instead of forming it we take the SVD
    of the centred n × d data ``A^T = U S V^T``: the rows of ``V^T`` are its eigenvectors and
    ``S² / (n - 1)`` the eigenvalues — the same eigenfaces, cheaper and numerically safer.
    """
    n_samples = len(faces.data)
    if n_samples < 2:
        raise ValueError("Eigenfaces need at least 2 training images.")
    mean = faces.data.mean(axis=0)
    centred = faces.data - mean
    _u, singular, vt = np.linalg.svd(centred, full_matrices=False)
    eigenvalues = singular**2 / (n_samples - 1)
    # n images span at most n - 1 directions once the mean is removed
    k = min(n_components, n_samples - 1, centred.shape[1])
    eigenfaces = vt[:k]
    # SVD signs are arbitrary: make each eigenface's largest pixel positive so results are reproducible
    signs = np.sign(eigenfaces[np.arange(k), np.abs(eigenfaces).argmax(axis=1)])
    eigenfaces = eigenfaces * np.where(signs == 0, 1, signs)[:, None]
    relative = [path.relative_to(root).as_posix() if root else path.as_posix() for path in faces.paths]
    model = EigenfaceModel(
        mean=mean,
        eigenfaces=eigenfaces,
        eigenvalues=eigenvalues[:k],
        total_variance=float(eigenvalues.sum()),
        coefficients=np.zeros((n_samples, k)),
        labels=np.array(faces.labels, dtype=str),
        paths=np.array(relative, dtype=str),
        shape=faces.shape,
        root=root,
    )
    return replace(model, coefficients=model.project(faces.data))


def save_model(model: EigenfaceModel, path: Path) -> None:
    """mean, eigenfaces, PCA coefficients, labels … in one ``.npz`` (a zip of ``.npy`` arrays)."""
    np.savez_compressed(
        path,
        mean=model.mean,
        eigenfaces=model.eigenfaces,
        eigenvalues=model.eigenvalues,
        total_variance=np.array(model.total_variance),
        coefficients=model.coefficients,
        labels=model.labels,
        paths=model.paths,
        shape=np.array(model.shape),
    )


def load_model(path: Path) -> EigenfaceModel:
    """Read a model written by :func:`save_model`; its dataset folder is the file's folder."""
    try:
        with np.load(path, allow_pickle=False) as data:
            return EigenfaceModel(
                mean=data["mean"],
                eigenfaces=data["eigenfaces"],
                eigenvalues=data["eigenvalues"],
                total_variance=float(data["total_variance"]),
                coefficients=data["coefficients"],
                labels=data["labels"],
                paths=data["paths"],
                shape=(int(data["shape"][0]), int(data["shape"][1])),
                root=Path(path).parent,
            )
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"{Path(path).name} is not an eigenface model: {error}") from None


# ---------------------------------------------------------------- figures


def figure_to_image(figure: Figure) -> Image.Image:
    """Render a matplotlib figure to an RGB PIL image."""
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())


def grid_shape(count: int, columns: int = 4) -> tuple[int, int]:
    columns = min(columns, count)
    return -(-count // columns), columns


def gallery_figure(
    images: list[np.ndarray], titles: list[str], colours: list[str] | None = None, columns: int = 4
) -> Figure:
    """Gray portraits in a grid with a title above each one (the notebook's ``plot_gallery``)."""
    rows, columns = grid_shape(len(images), columns)
    figure = Figure(figsize=(1.8 * columns, 2.4 * rows))
    figure.subplots_adjust(bottom=0.02, left=0.01, right=0.99, top=0.90, hspace=0.45)
    for index, (image, title) in enumerate(zip(images, titles)):
        ax = figure.add_subplot(rows, columns, index + 1)
        ax.imshow(image, cmap="gray")
        ax.set_title(title, size=9, color=colours[index] if colours else "black")
        ax.set_xticks(())
        ax.set_yticks(())
    return figure


def eigenface_gallery(model: EigenfaceModel, count: int, include_mean: bool = False) -> Image.Image:
    """The first ``count`` eigenfaces (each auto-contrasted by imshow), optionally after the mean face."""
    count = min(count, model.n_components)
    images = [face.reshape(model.shape) for face in model.eigenfaces[:count]]
    titles = [f"eigenface {index}" for index in range(count)]
    if include_mean:
        images.insert(0, model.mean.reshape(model.shape))
        titles.insert(0, "mean face μ")
    return figure_to_image(gallery_figure(images, titles))


def variance_figure(model: EigenfaceModel, target: float = POV_TARGET) -> Figure:
    """Scree graph and proportion of variance (PoV) explained, as in the slides."""
    k = np.arange(1, model.n_components + 1)
    figure = Figure(figsize=(8, 6), layout="constrained")
    scree, pov = figure.subplots(2, 1)
    scree.plot(k, model.eigenvalues, marker="+")
    scree.set_title("(a) Scree graph")
    scree.set_xlabel("Eigenfaces")
    scree.set_ylabel("Eigenvalues")
    scree.grid(True, linestyle=":")
    pov.plot(k, model.pov, marker="+")
    pov.set_title("(b) Proportion of variance explained")
    pov.set_xlabel("Eigenfaces")
    pov.set_ylabel("PoV")
    pov.set_ylim(0, 1.02)
    pov.grid(True, linestyle=":")
    needed = components_for(model, target)
    if needed is not None:
        pov.axhline(target, color="tab:blue", linewidth=1)
        pov.axvline(needed, color="tab:red", linewidth=1.5)
        left = needed > model.n_components / 2  # keep the label inside the axes
        pov.annotate(
            f"k = {needed} → PoV {model.pov[needed - 1]:.2f}",
            (needed, target),
            xytext=(-8 if left else 8, -18),
            textcoords="offset points",
            ha="right" if left else "left",
            color="tab:red",
        )
    return figure


def components_for(model: EigenfaceModel, target: float) -> int | None:
    """Smallest k whose PoV reaches ``target`` (``None`` if the kept eigenfaces never do)."""
    reached = np.nonzero(model.pov >= target)[0]
    return int(reached[0]) + 1 if len(reached) else None


# ---------------------------------------------------------------- tool


@dataclass(frozen=True)
class EigenfaceSettings:
    folder: str = str(DEFAULT_DATASET)
    n_components: int = 100  # as in the notebook
    n_display: int = 12  # the slide shows eigenfaces 0–11
    include_mean: bool = False
    show_variance: bool = True


class EigenfaceDialog(simpledialog.Dialog):
    """Dataset folder (with an LFW download button), K, and what to show."""

    def body(self, master: tk.Frame) -> tk.Widget:
        d = EigenfaceSettings()
        self.folder = tk.StringVar(value=d.folder)
        self.n_components = tk.StringVar(value=str(d.n_components))
        self.n_display = tk.StringVar(value=str(d.n_display))
        self.include_mean = tk.BooleanVar(value=d.include_mean)
        self.show_variance = tk.BooleanVar(value=d.show_variance)

        data = ttk.LabelFrame(master, text="Face folder (train/ and test/ subfolders, one folder per person)")
        data.grid(row=0, column=0, columnspan=2, sticky="we")
        ttk.Entry(data, textvariable=self.folder, width=52).grid(row=0, column=0, columnspan=2, padx=6, pady=2)
        ttk.Button(data, text="Browse…", command=self._browse).grid(row=1, column=0, sticky="w", padx=6, pady=2)
        ttk.Button(data, text="Download LFW into this folder…", command=self._download).grid(
            row=1, column=1, sticky="e", padx=6, pady=2
        )

        pca = ttk.LabelFrame(master, text="PCA")
        pca.grid(row=1, column=0, columnspan=2, sticky="we", pady=(6, 0))
        first = spin_row(pca, 0, "Components to keep (K):", self.n_components, COMPONENTS_RANGE, 10)
        spin_row(pca, 1, "Eigenfaces to show:", self.n_display, DISPLAY_RANGE, 4)
        ttk.Checkbutton(pca, text="Show the mean face first", variable=self.include_mean).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=6
        )
        ttk.Checkbutton(pca, text="Scree / PoV plot to choose K (popup)", variable=self.show_variance).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=6
        )
        ttk.Label(master, text=f"The model is saved as {MODEL_FILENAME} in the face folder.").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        return first

    def _browse(self) -> None:
        directory = filedialog.askdirectory(title="Face folder", parent=self, mustexist=False)
        if directory:
            self.folder.set(directory)

    def _download(self) -> None:
        folder = Path(self.folder.get().strip() or DEFAULT_DATASET)
        if (folder / TRAIN_DIR).exists() and not messagebox.askyesno(
            self.title(), f"{folder} already has a train folder. Export LFW again?", parent=self
        ):
            return
        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            n_train, n_test, names = export_lfw(folder)
        except Exception as error:  # network, disk, scikit-learn missing…
            messagebox.showerror(self.title(), f"Could not export LFW: {error}", parent=self)
            return
        finally:
            self.configure(cursor="")
        self.folder.set(str(folder))
        messagebox.showinfo(
            self.title(),
            f"Saved {n_train} train and {n_test} test faces of {len(names)} people to {folder}.",
            parent=self,
        )

    def validate(self) -> bool:
        try:
            folder = self.folder.get().strip()
            if not folder or not Path(folder).is_dir():
                raise ValueError("Choose an existing face folder (or download LFW into one).")
            self._settings = EigenfaceSettings(
                folder=folder,
                n_components=read_number(self.n_components, "Components", COMPONENTS_RANGE, int),
                n_display=read_number(self.n_display, "Eigenfaces to show", DISPLAY_RANGE, int),
                include_mean=self.include_mean.get(),
                show_variance=self.show_variance.get(),
            )
        except ValueError as error:
            messagebox.showerror(self.title(), str(error), parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self._settings


class EigenfaceBuilderTool(ForensicsTool):
    tool_id = "eigenface_builder"
    title = "Eigenface builder (PCA)"
    category = "Set 6: Faces"
    description = (
        "Compute eigenfaces (PCA) from a folder of training faces, save mean, eigenfaces and coefficients "
        "to a model file and show the first K eigenfaces."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        settings = EigenfaceDialog(parent, title="Eigenface builder").result
        if settings is None:
            return None
        root = Path(settings.folder)
        train, test = split_folders(root)

        window = parent.winfo_toplevel()
        previous_title = window.title()

        def progress(index: int, total: int, path: Path) -> None:
            if index % 50 == 0 or index == total - 1:
                window.title(f"Loading faces {index + 1}/{total}")
                window.update_idletasks()

        window.configure(cursor="watch")
        try:
            faces, failed = load_faces(train, progress=progress)
            if len(faces.data) < 2:
                return ToolResult(message=f"Need at least 2 face images in {train}; found {len(faces.data)}.")
            window.title("Computing eigenfaces…")
            window.update_idletasks()
            model = fit_eigenfaces(faces, settings.n_components, root)
            model_path = root / MODEL_FILENAME
            save_model(model, model_path)
        finally:
            window.configure(cursor="")
            window.title(previous_title)

        people = Counter(faces.labels)
        needed = components_for(model, POV_TARGET)
        details: dict[str, Any] = {
            "Operation": "Eigenfaces (PCA)",
            "Library": "numpy.linalg.svd",
            "Training folder": str(train),
            "Test folder": str(test) if test else "—",
            "Training images": len(faces.data),
            "People": len(people),
            "Image size (h × w)": f"{faces.shape[0]} × {faces.shape[1]} = {faces.data.shape[1]} pixels",
            "Components kept (K)": model.n_components,
            f"PoV with K={model.n_components}": f"{model.pov[-1]:.1%}",
            f"K for PoV ≥ {POV_TARGET:g}": needed if needed is not None else f"> {model.n_components}",
            "Model saved to": str(model_path),
            "Saved arrays": "mean, eigenfaces, eigenvalues, coefficients, labels, paths",
        }
        for name, count in people.most_common():
            details[f"  {name}"] = f"{count} image(s)"
        if failed:
            details["Unreadable files"] = ", ".join(path.name for path in failed)
        if settings.show_variance:
            show_figure(parent, "Eigenfaces: scree graph and PoV", variance_figure(model))

        return ToolResult(
            image=eigenface_gallery(model, settings.n_display, settings.include_mean),
            message=(
                f"Built {model.n_components} eigenfaces from {len(faces.data)} faces of {len(people)} people "
                f"(PoV {model.pov[-1]:.1%}); saved {MODEL_FILENAME}."
            ),
            details=details,
        )


__all__ = [
    "EigenfaceBuilderTool",
    "EigenfaceDialog",
    "EigenfaceModel",
    "EigenfaceSettings",
    "FaceSet",
    "MODEL_FILENAME",
    "components_for",
    "eigenface_gallery",
    "export_lfw",
    "face_vector",
    "figure_to_image",
    "fit_eigenfaces",
    "gallery_figure",
    "label_for",
    "load_faces",
    "load_model",
    "person_name",
    "save_model",
    "split_folders",
    "variance_figure",
]
