"""Set 6.3: Face recognition — project a face on the saved eigenfaces and find the closest training faces."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
import numpy as np
from PIL import Image, UnidentifiedImageError

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialog_fields import read_number, spin_row
from .hog import show_figure
from .eigenfaces import (
    DEFAULT_DATASET,
    MODEL_FILENAME,
    TEST_DIR,
    EigenfaceModel,
    face_vector,
    figure_to_image,
    gallery_figure,
    load_faces,
    load_model,
    person_name,
    split_folders,
)

MODEL_TYPES = [("Eigenface model", "*.npz"), ("All files", "*.*")]
MODES = {
    "query": "Closest training faces to the current image",
    "evaluate": "Evaluate the whole test folder (predicted vs true gallery)",
    "explore": "Eigenface explorer: slide k, click a weight to see its eigenface (popup)",
}
NEIGHBOURS_RANGE = (1, 10)
GALLERY_RANGE = (1, 48)
MODEL_SEARCH_DEPTH = 4  # folders above the image searched for a saved model
GALLERY_SEED = 0  # the evaluation gallery shows a fixed random sample of the test set

RIGHT, WRONG, UNKNOWN = "tab:green", "tab:red", "black"


@dataclass(frozen=True)
class RecognitionSettings:
    model_path: str = ""
    mode: str = "query"
    n_neighbours: int = 3  # "display the closest 2-3 images"
    n_components: int = 0  # 0: every eigenface in the model
    n_gallery: int = 12  # the slide shows a 3 × 4 gallery
    show_reconstruction: bool = True
    # Divide each weight by sqrt(eigenvalue) so every eigenface counts equally (sklearn's PCA(whiten=True)).
    # On LFW 1-NN accuracy goes from 51% to 65% with K=100; off by default to follow the set's formula.
    whiten: bool = False


@dataclass(frozen=True)
class Neighbour:
    index: int
    label: str
    distance: float


def find_model(image_path: Path | None) -> Path | None:
    """A saved model next to the image or in one of its parent folders, else in the default dataset."""
    candidates: list[Path] = []
    if image_path is not None:
        candidates += [parent / MODEL_FILENAME for parent in list(Path(image_path).parents)[:MODEL_SEARCH_DEPTH]]
    candidates.append(DEFAULT_DATASET / MODEL_FILENAME)
    return next((path for path in candidates if path.is_file()), None)


def nearest_faces(
    model: EigenfaceModel, weights: np.ndarray, count: int, k: int | None = None, whiten: bool = False
) -> list[Neighbour]:
    """Training faces sorted by ``d_i = ||w_test - w_i||_2`` over the first ``k`` eigenfaces.

    ``whiten`` divides the weights by ``sqrt(lambda_i)`` first, so low-variance eigenfaces count as much.
    """
    k = k or model.n_components
    scale = np.sqrt(model.eigenvalues[:k]) if whiten else 1.0
    distances = np.linalg.norm((model.coefficients[:, :k] - np.ravel(weights)[:k]) / scale, axis=1)
    order = np.argsort(distances, kind="stable")[:count]
    return [Neighbour(int(i), str(model.labels[i]), float(distances[i])) for i in order]


def predict(model: EigenfaceModel, data: np.ndarray, k: int | None = None, whiten: bool = False) -> list[str]:
    """1-nearest-neighbour name (``i* = argmin_i d_i``) for each row of ``data``."""
    weights = model.project(data, k)
    return [nearest_faces(model, row, 1, k, whiten)[0].label for row in weights]


def true_label(path: Path | None, model: EigenfaceModel) -> str | None:
    """The person the image belongs to, if its folder is named after someone in the model."""
    if path is None:
        return None
    name = person_name(Path(path).parent.name)
    return name if name in set(model.labels) else None


def training_face(model: EigenfaceModel, index: int) -> tuple[np.ndarray, bool]:
    """The stored training image; if it was moved, its eigenface reconstruction (second value ``False``)."""
    if model.root is not None:
        path = model.root / str(model.paths[index])
        try:
            with Image.open(path) as opened:
                return face_vector(opened, model.shape).reshape(model.shape), True
        except (OSError, UnidentifiedImageError):
            pass
    return model.reconstruct(model.coefficients[index]).reshape(model.shape), False


def short_name(name: str) -> str:
    """Surname only, as in the slide's gallery titles (``George W Bush`` -> ``Bush``)."""
    return name.rsplit(" ", 1)[-1]


def query_figure_image(
    model: EigenfaceModel,
    query: np.ndarray,
    neighbours: list[Neighbour],
    k: int,
    truth: str | None,
    show_reconstruction: bool,
) -> Image.Image:
    """Test face, optional reconstruction, then the closest training faces with names and distances."""
    images = [query.reshape(model.shape)]
    titles = ["test image\n" + (f"true: {short_name(truth)}" if truth else "true: ?")]
    colours = [UNKNOWN]
    if show_reconstruction:
        weights = model.project(query, k)
        images.append(np.clip(model.reconstruct(weights).reshape(model.shape), 0, 1))
        titles.append(f"reconstruction\nK = {k}")
        colours.append(UNKNOWN)
    for rank, neighbour in enumerate(neighbours, start=1):
        face, _found = training_face(model, neighbour.index)
        images.append(face)
        titles.append(f"#{rank} {short_name(neighbour.label)}\nd = {neighbour.distance:.2f}")
        colours.append(UNKNOWN if truth is None else RIGHT if neighbour.label == truth else WRONG)
    figure = gallery_figure(images, titles, colours, columns=len(images))
    figure.suptitle(f"predicted: {neighbours[0].label}", size=11)
    figure.subplots_adjust(top=0.72)
    return figure_to_image(figure)


def evaluation_gallery(
    data: np.ndarray, shape: tuple[int, int], truths: list[str], predictions: list[str], count: int
) -> Image.Image:
    """``predicted / true`` titles over a fixed random sample of the test faces (red when wrong)."""
    rng = np.random.default_rng(GALLERY_SEED)
    chosen = np.sort(rng.choice(len(data), size=min(count, len(data)), replace=False))
    images = [data[i].reshape(shape) for i in chosen]
    titles = [f"predicted: {short_name(predictions[i])}\ntrue:      {short_name(truths[i])}" for i in chosen]
    colours = [RIGHT if predictions[i] == truths[i] else WRONG for i in chosen]
    return figure_to_image(gallery_figure(images, titles, colours))


def reconstruction_errors(model: EigenfaceModel, query: np.ndarray) -> np.ndarray:
    """``||x - x_hat_k||`` for k = 1 … K.

    The eigenfaces are orthonormal, so by Pythagoras ``||x - x_hat_k||² = ||x - mu||² - sum_{i<=k} w_i²``:
    every extra eigenface removes exactly its squared weight from the error.
    """
    weights = model.project(query).ravel()
    residual = float(np.sum((np.ravel(query) - model.mean) ** 2)) - np.cumsum(weights**2)
    return np.sqrt(np.maximum(residual, 0.0))


class EigenfaceExplorer:
    """Interactive view of ``x_hat = mu + sum_{i<=k} w_i u_i`` for one face.

    Drag the k slider (or ←/→) to add eigenfaces one by one and watch the reconstruction, its error and the
    nearest training face change; click a bar of the weight vector (or ↑/↓) to see that eigenface.
    """

    def __init__(
        self, model: EigenfaceModel, query: np.ndarray, k: int, truth: str | None = None, whiten: bool = False
    ) -> None:
        self.model = model
        self.query = np.ravel(query)
        self.truth = truth
        self.whiten = whiten
        self.weights = model.project(self.query).ravel()
        self.errors = reconstruction_errors(model, self.query)
        self.k = 0
        self.selected = -1
        shape = model.shape
        n = model.n_components

        self.figure = Figure(figsize=(12, 7.2), layout="constrained")
        grid = self.figure.add_gridspec(3, 4, height_ratios=[4, 3, 0.35])
        self.ax_query, self.ax_recon, self.ax_eigen, self.ax_nearest = (
            self.figure.add_subplot(grid[0, i]) for i in range(4)
        )
        self.ax_weights = self.figure.add_subplot(grid[1, :2])
        self.ax_error = self.figure.add_subplot(grid[1, 2:])
        self.ax_slider = self.figure.add_subplot(grid[2, 1:3])
        for ax in (self.ax_query, self.ax_recon, self.ax_eigen, self.ax_nearest):
            ax.set_xticks(())
            ax.set_yticks(())

        self.ax_query.imshow(self.query.reshape(shape), cmap="gray", vmin=0, vmax=1)
        self.ax_query.set_title("test face x" + (f"\ntrue: {truth}" if truth else ""), size=10)
        self.recon_image = self.ax_recon.imshow(np.zeros(shape), cmap="gray", vmin=0, vmax=1)
        self.eigen_image = self.ax_eigen.imshow(np.zeros(shape), cmap="gray")
        self.nearest_image = self.ax_nearest.imshow(np.zeros(shape), cmap="gray", vmin=0, vmax=1)

        self.bars = self.ax_weights.bar(np.arange(n), self.weights, width=0.85)
        self.ax_weights.axhline(0, color="black", linewidth=0.6)
        self.ax_weights.set_xlim(-1, n)
        self.ax_weights.set_xlabel("eigenface i (click a bar)")
        self.ax_weights.set_ylabel("weight w_i")
        self.ax_weights.set_title("Face-space vector w = U_Kᵀ(x − μ): blue bars are used", size=10)

        self.ax_error.plot(np.arange(1, n + 1), self.errors, color="tab:blue")
        self.error_marker = self.ax_error.axvline(1, color="tab:red")
        self.ax_error.set_xlabel("k (eigenfaces used)")
        self.ax_error.set_ylabel("‖x − x̂_k‖")
        self.ax_error.grid(True, linestyle=":")

        self.slider = Slider(self.ax_slider, "k", 1, n, valinit=k, valstep=1)
        self.slider.on_changed(lambda value: self.set_k(int(value)))
        self.set_k(k)
        self.select(0)

    def set_k(self, k: int) -> None:
        k = int(np.clip(k, 1, self.model.n_components))
        if k == self.k:
            return
        self.k = k
        if int(self.slider.val) != k:
            self.slider.set_val(k)  # calls back here, where k == self.k stops the loop
        recon = self.model.reconstruct(self.weights[:k]).reshape(self.model.shape)
        self.recon_image.set_data(np.clip(recon, 0, 1))
        self.ax_recon.set_title(f"x̂ = μ + Σ w_i u_i,  k = {k}\nPoV {self.model.pov[k - 1]:.0%}", size=10)
        self.nearest = nearest_faces(self.model, self.weights, 1, k, self.whiten)[0]
        face, _found = training_face(self.model, self.nearest.index)
        self.nearest_image.set_data(face)
        colour = UNKNOWN if self.truth is None else RIGHT if self.nearest.label == self.truth else WRONG
        self.ax_nearest.set_title(
            f"nearest with k = {k}:\n{self.nearest.label} (d = {self.nearest.distance:.2f})", size=10, color=colour
        )
        self.error_marker.set_xdata([k, k])
        self.ax_error.set_title(f"Reconstruction error: {self.errors[k - 1]:.3f} with k = {k}", size=10)
        self._colour_bars()

    def select(self, index: int) -> None:
        index = int(np.clip(index, 0, self.model.n_components - 1))
        self.selected = index
        eigenface = self.model.eigenfaces[index].reshape(self.model.shape)
        self.eigen_image.set_data(eigenface)
        self.eigen_image.set_clim(eigenface.min(), eigenface.max())
        self.ax_eigen.set_title(f"eigenface {index}\nw_{index} = {self.weights[index]:+.2f}", size=10)
        self._colour_bars()

    def _colour_bars(self) -> None:
        for i, bar in enumerate(self.bars):
            bar.set_color("tab:red" if i == self.selected else "tab:blue" if i < self.k else "lightgray")

    def on_click(self, event: MouseEvent) -> None:
        if event.inaxes is not self.ax_weights or event.xdata is None:
            return
        index = int(round(event.xdata))
        if 0 <= index < self.model.n_components:
            self.select(index)
            self.figure.canvas.draw_idle()

    def on_key(self, event: KeyEvent) -> None:
        steps = {"right": (1, 0), "left": (-1, 0), "up": (0, 1), "down": (0, -1)}
        if event.key not in steps:
            return
        dk, di = steps[event.key]
        if dk:
            self.set_k(self.k + dk)
        else:
            self.select(self.selected + di)
        self.figure.canvas.draw_idle()


class RecognitionDialog(simpledialog.Dialog):
    def __init__(self, parent: tk.Misc, model_path: Path | None) -> None:
        self.found = model_path
        super().__init__(parent, title="Face recognition (eigenfaces)")

    def body(self, master: tk.Frame) -> tk.Widget:
        d = RecognitionSettings()
        self.model_path = tk.StringVar(value=str(self.found) if self.found else "")
        self.mode = tk.StringVar(value=d.mode)
        self.n_neighbours = tk.StringVar(value=str(d.n_neighbours))
        self.n_components = tk.StringVar(value=str(d.n_components))
        self.n_gallery = tk.StringVar(value=str(d.n_gallery))
        self.show_reconstruction = tk.BooleanVar(value=d.show_reconstruction)
        self.whiten = tk.BooleanVar(value=d.whiten)

        model = ttk.LabelFrame(master, text=f"Saved model ({MODEL_FILENAME} from the Eigenface builder)")
        model.grid(row=0, column=0, columnspan=2, sticky="we")
        ttk.Entry(model, textvariable=self.model_path, width=50).grid(row=0, column=0, padx=6, pady=2)
        ttk.Button(model, text="Browse…", command=self._browse).grid(row=0, column=1, padx=(0, 6))

        mode = ttk.LabelFrame(master, text="Mode")
        mode.grid(row=1, column=0, columnspan=2, sticky="we", pady=(6, 0))
        for row, (key, text) in enumerate(MODES.items()):
            ttk.Radiobutton(mode, text=text, value=key, variable=self.mode).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=6
            )

        params = ttk.LabelFrame(master, text="Matching")
        params.grid(row=2, column=0, columnspan=2, sticky="we", pady=(6, 0))
        first = spin_row(params, 0, "Closest faces to show:", self.n_neighbours, NEIGHBOURS_RANGE, 1)
        spin_row(params, 1, "Eigenfaces used (0 = all K):", self.n_components, (0, 1000), 10)
        spin_row(params, 2, "Evaluation gallery size:", self.n_gallery, GALLERY_RANGE, 4)
        ttk.Checkbutton(params, text="Show the eigenface reconstruction of the test face", variable=self.show_reconstruction).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=6
        )
        ttk.Checkbutton(
            params, text="Whiten weights (divide by √λ, like sklearn's whiten=True)", variable=self.whiten
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=6)
        return first

    def _browse(self) -> None:
        filename = filedialog.askopenfilename(title="Eigenface model", filetypes=MODEL_TYPES, parent=self)
        if filename:
            self.model_path.set(filename)

    def validate(self) -> bool:
        try:
            path = self.model_path.get().strip()
            if not path or not Path(path).is_file():
                raise ValueError("Choose a saved eigenface model (run the Eigenface builder first).")
            self._settings = RecognitionSettings(
                model_path=path,
                mode=self.mode.get(),
                n_neighbours=read_number(self.n_neighbours, "Closest faces", NEIGHBOURS_RANGE, int),
                n_components=read_number(self.n_components, "Eigenfaces used", (0, 1000), int),
                n_gallery=read_number(self.n_gallery, "Gallery size", GALLERY_RANGE, int),
                show_reconstruction=self.show_reconstruction.get(),
                whiten=self.whiten.get(),
            )
        except ValueError as error:
            messagebox.showerror(self.title(), str(error), parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self._settings


class FaceRecognitionTool(ForensicsTool):
    tool_id = "face_recognition"
    title = "Face recognition (eigenfaces)"
    category = "Set 6: Faces"
    description = (
        "Project the face on the saved eigenfaces (w = U_Kᵀ(x − μ)) and show the closest training faces "
        "with names, or evaluate the whole test folder."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by main window

        settings = RecognitionDialog(parent, find_model(document.path)).result
        if settings is None:
            return None
        model = load_model(Path(settings.model_path))
        if settings.n_components > model.n_components:
            raise ValueError(f"The model has only {model.n_components} eigenfaces.")
        k = settings.n_components or model.n_components
        details: dict[str, Any] = {
            "Operation": "Eigenface recognition (1-NN)",
            "Model": settings.model_path,
            "Training faces": len(model.coefficients),
            "People": len(set(model.labels)),
            "Eigenfaces used (k)": f"{k} of {model.n_components}",
            "PoV with k": f"{model.pov[k - 1]:.1%}",
            "Distance": "d_i = ‖(w_test − w_i) / √λ‖₂ (whitened)" if settings.whiten else "d_i = ‖w_test − w_i‖₂",
        }
        if settings.mode == "evaluate":
            return self._evaluate(parent, model, k, settings, details)
        if settings.mode == "explore":
            return self._explore(parent, document, model, k, settings, details)
        return self._query(document, model, k, settings, details)

    @staticmethod
    def _explore(
        parent: tk.Misc,
        document: ImageDocument,
        model: EigenfaceModel,
        k: int,
        settings: RecognitionSettings,
        details: dict[str, Any],
    ) -> ToolResult:
        query = face_vector(document.current, model.shape)
        explorer = EigenfaceExplorer(model, query, k, true_label(document.path, model), settings.whiten)
        popup = show_figure(parent, "Eigenface explorer", explorer.figure)
        explorer.figure.canvas.mpl_connect("button_press_event", explorer.on_click)
        explorer.figure.canvas.mpl_connect("key_press_event", explorer.on_key)
        popup.explorer = explorer  # matplotlib keeps only weak references to the handlers and slider
        details["Reconstruction error (k = 1 / K)"] = f"{explorer.errors[0]:.3f} / {explorer.errors[-1]:.3f}"
        return ToolResult(
            message="Opened the eigenface explorer: drag k (or ←/→), click a weight bar (or ↑/↓).",
            details=details,
        )

    @staticmethod
    def _query(
        document: ImageDocument, model: EigenfaceModel, k: int, settings: RecognitionSettings, details: dict[str, Any]
    ) -> ToolResult:
        query = face_vector(document.current, model.shape)
        weights = model.project(query, k)
        neighbours = nearest_faces(model, weights, settings.n_neighbours, k, settings.whiten)
        truth = true_label(document.path, model)
        error = float(np.linalg.norm(query - model.reconstruct(weights)))

        details["Test image"] = document.path.name if document.path else "current image"
        details["Resized to (h × w)"] = f"{model.shape[0]} × {model.shape[1]}"
        details["Reconstruction error ‖x − x̂‖"] = f"{error:.3f}"
        details["Predicted"] = neighbours[0].label
        if truth is not None:
            details["True"] = truth
            details["Correct"] = "yes" if neighbours[0].label == truth else "no"
        for rank, neighbour in enumerate(neighbours, start=1):
            details[f"#{rank} {neighbour.label}"] = f"d = {neighbour.distance:.3f} ({model.paths[neighbour.index]})"
        details["Weights w_test (first 5)"] = ", ".join(f"{w:.2f}" for w in np.ravel(weights)[:5])

        verdict = "" if truth is None else (" ✓" if neighbours[0].label == truth else f" ✗ (true: {truth})")
        return ToolResult(
            image=query_figure_image(model, query, neighbours, k, truth, settings.show_reconstruction),
            message=f"Predicted: {neighbours[0].label} (d = {neighbours[0].distance:.2f}){verdict}.",
            details=details,
        )

    @staticmethod
    def _evaluate(
        parent: tk.Misc, model: EigenfaceModel, k: int, settings: RecognitionSettings, details: dict[str, Any]
    ) -> ToolResult | None:
        test = split_folders(model.root)[1] if model.root else None
        if test is None:
            directory = filedialog.askdirectory(title="Test folder (one folder per person)", parent=parent, mustexist=True)
            if not directory:
                return None
            test = Path(directory)
        faces, failed = load_faces(test, model.shape)
        known = [i for i, label in enumerate(faces.labels) if label in set(model.labels)]
        if not known:
            return ToolResult(message=f"No test faces of people in the model found in {test}.", details=details)
        data = faces.data[known]
        truths = [faces.labels[i] for i in known]
        predictions = predict(model, data, k, settings.whiten)
        correct = [p == t for p, t in zip(predictions, truths)]

        details["Test folder"] = str(test)
        details["Test faces"] = len(known)
        details["Accuracy"] = f"{np.mean(correct):.1%} ({sum(correct)}/{len(correct)})"
        totals, hits = Counter(truths), Counter(t for t, ok in zip(truths, correct) if ok)
        for name in sorted(totals):
            details[f"  {name}"] = f"{hits[name]}/{totals[name]} ({hits[name] / totals[name]:.0%})"
        skipped = len(faces.labels) - len(known)
        if skipped:
            details["Skipped (unknown people)"] = skipped
        if failed:
            details["Unreadable files"] = ", ".join(path.name for path in failed)
        return ToolResult(
            image=evaluation_gallery(data, model.shape, truths, predictions, settings.n_gallery),
            message=f"Test accuracy {np.mean(correct):.1%} on {len(known)} faces with k = {k} eigenfaces (1-NN).",
            details=details,
        )


__all__ = [
    "EigenfaceExplorer",
    "FaceRecognitionTool",
    "Neighbour",
    "RecognitionDialog",
    "RecognitionSettings",
    "evaluation_gallery",
    "find_model",
    "nearest_faces",
    "predict",
    "query_figure_image",
    "reconstruction_errors",
    "short_name",
    "training_face",
    "true_label",
]
