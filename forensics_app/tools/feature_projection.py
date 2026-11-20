"""Set 8.3: Load extracted test-set feature vectors and project them to 2D with t-SNE and UMAP."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from matplotlib import colormaps
from matplotlib.figure import Figure
import numpy as np
from sklearn.manifold import TSNE

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .cnn_classification import DEFAULT_SET8_DIR, PROJECT_ROOT
from .dialog_fields import read_number, spin_row
from .eigenfaces import figure_to_image
from .hog import show_figure
from .progress import ProgressPopup

FEATURE_FILETYPES = [("Feature vectors (.npz)", "*.npz"), ("All files", "*.*")]

PROJECTION_METHODS = {
    "both": "Both t-SNE and UMAP (side by side)",
    "tsne": "t-SNE only",
    "umap": "UMAP only",
}

PERPLEXITY_RANGE = (2.0, 100.0)
TSNE_ITER_RANGE = (250, 2000)
UMAP_NEIGHBORS_RANGE = (2, 100)
UMAP_MINDIST_RANGE = (0.0, 0.99)
MAX_SAMPLES_RANGE = (0, 20000)  # 0 = use all samples
RANDOM_SEED = 42


@dataclass(frozen=True)
class FeatureDataset:
    """Extracted CNN feature vectors and class labels saved from Set 8.1 / 8.3."""

    features: np.ndarray  # (N, D) float32
    labels: np.ndarray  # (N,) int32
    class_names: tuple[str, ...]
    model_name: str = "Unknown model"
    layer_name: str = "Unknown layer"
    dataset: str = "Test set"
    source_path: Path | None = None

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_dims(self) -> int:
        return int(self.features.shape[1])


@dataclass(frozen=True)
class FeatureProjectionSettings:
    feature_path: str = ""
    method: str = "both"
    max_samples: int = 0  # 0 = all samples
    tsne_perplexity: float = 30.0
    tsne_max_iter: int = 500
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1


# ---------------------------------------------------------------- discovery & loading


def discover_feature_files(folder: Path = DEFAULT_SET8_DIR) -> list[Path]:
    """Return sorted ``.npz`` feature files found in ``folder``."""
    if not folder.is_dir():
        return []
    found = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".npz"]
    return sorted(found, key=lambda p: p.name.lower())


def load_feature_vectors(path: Path) -> FeatureDataset:
    """Load extracted feature vectors and labels from a ``.npz`` archive."""
    with np.load(path, allow_pickle=False) as data:
        if "features" not in data or "labels" not in data:
            raise ValueError(f"Expected 'features' and 'labels' arrays inside {path.name}.")
        features = np.asarray(data["features"], dtype=np.float32)
        if features.ndim != 2 or features.shape[0] < 3:
            raise ValueError(f"Expected a 2D feature matrix with at least 3 samples, got shape {features.shape}.")
        labels = np.asarray(data["labels"]).ravel()
        if len(labels) != features.shape[0]:
            raise ValueError("Number of labels does not match number of feature vectors.")

        # Normalize string labels to integer indices if needed
        if labels.dtype.kind in ("U", "S", "O"):
            unique_names = sorted(set(str(x) for x in labels))
            lookup = {name: idx for idx, name in enumerate(unique_names)}
            labels_int = np.array([lookup[str(x)] for x in labels], dtype=np.int32)
            class_names = tuple(unique_names)
        else:
            labels_int = labels.astype(np.int32)
            if "class_names" in data:
                class_names = tuple(str(c) for c in data["class_names"].tolist())
            else:
                n_cls = int(labels_int.max()) + 1
                class_names = tuple(f"class_{i}" for i in range(n_cls))

        model_name = str(data["model_name"].item()) if "model_name" in data else path.stem
        layer_name = str(data["layer_name"].item()) if "layer_name" in data else "penultimate"
        dataset = str(data["dataset"].item()) if "dataset" in data else "Test set"

    return FeatureDataset(
        features=features,
        labels=labels_int,
        class_names=class_names,
        model_name=model_name,
        layer_name=layer_name,
        dataset=dataset,
        source_path=path,
    )


def subsample_stratified(
    dataset: FeatureDataset,
    max_samples: int,
    seed: int = RANDOM_SEED,
) -> FeatureDataset:
    """Optionally subsample ``dataset`` while preserving class balance."""
    if max_samples <= 0 or max_samples >= dataset.n_samples:
        return dataset
    rng = np.random.default_rng(seed)
    unique_classes = np.unique(dataset.labels)
    per_class = max(1, max_samples // max(len(unique_classes), 1))
    chosen: list[np.ndarray] = []
    for cls_id in unique_classes:
        idxs = np.where(dataset.labels == cls_id)[0]
        take = min(per_class, len(idxs))
        chosen.append(rng.choice(idxs, size=take, replace=False))
    selected = np.concatenate(chosen)
    rng.shuffle(selected)
    return FeatureDataset(
        features=dataset.features[selected],
        labels=dataset.labels[selected],
        class_names=dataset.class_names,
        model_name=dataset.model_name,
        layer_name=dataset.layer_name,
        dataset=dataset.dataset,
        source_path=dataset.source_path,
    )


# ---------------------------------------------------------------- projections


def project_tsne(
    features: np.ndarray,
    *,
    perplexity: float = 30.0,
    max_iter: int = 500,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Project ``(N, D)`` feature vectors onto 2D using scikit-learn's t-SNE."""
    n_samples = int(features.shape[0])
    if n_samples < 3:
        raise ValueError("t-SNE requires at least 3 feature vectors.")
    safe_perplexity = float(min(perplexity, max(1.0, (n_samples - 1) / 3.0)))

    kwargs: dict[str, Any] = {
        "n_components": 2,
        "perplexity": safe_perplexity,
        "random_state": random_state,
    }
    params = inspect.signature(TSNE.__init__).parameters
    if "max_iter" in params:
        kwargs["max_iter"] = int(max_iter)
    elif "n_iter" in params:
        kwargs["n_iter"] = int(max_iter)

    embedding = TSNE(**kwargs).fit_transform(np.asarray(features, dtype=np.float32))
    return np.asarray(embedding, dtype=np.float32)


def project_umap(
    features: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Project ``(N, D)`` feature vectors onto 2D using ``umap-learn``."""
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "umap-learn is required for UMAP projection. Install it with: pip install umap-learn"
        ) from exc

    n_samples = int(features.shape[0])
    if n_samples < 3:
        raise ValueError("UMAP requires at least 3 feature vectors.")
    safe_neighbors = int(max(2, min(n_neighbors, n_samples - 1)))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=safe_neighbors,
        min_dist=float(min_dist),
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(np.asarray(features, dtype=np.float32))
    return np.asarray(embedding, dtype=np.float32)


# ---------------------------------------------------------------- visualization


def projection_figure(
    projections: dict[str, np.ndarray],
    labels: np.ndarray,
    class_names: tuple[str, ...],
    *,
    subtitle: str = "",
) -> Figure:
    """Plot 2D scatter visualization(s) for t-SNE and/or UMAP colored by class."""
    n_panels = len(projections)
    fig = Figure(figsize=(5.6 * n_panels + 1.4, 5.2), layout="constrained")
    unique_ids = sorted(int(c) for c in np.unique(labels))
    cmap = colormaps["tab10" if len(unique_ids) <= 10 else "tab20"]

    for panel_idx, (method_name, coords) in enumerate(projections.items(), start=1):
        ax = fig.add_subplot(1, n_panels, panel_idx)
        for color_idx, cls_id in enumerate(unique_ids):
            mask = labels == cls_id
            cls_label = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=14,
                alpha=0.78,
                color=cmap(color_idx % cmap.N),
                label=cls_label,
                edgecolors="none",
            )
        ax.set_title(method_name, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{method_name} 1")
        ax.set_ylabel(f"{method_name} 2")
        ax.grid(alpha=0.2)
        if panel_idx == n_panels:
            ax.legend(
                title="Class",
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
                fontsize=8,
                markerscale=1.8,
                frameon=True,
            )

    if subtitle:
        fig.suptitle(subtitle, fontsize=11)
    return fig


# ---------------------------------------------------------------- dialog & tool


class FeatureProjectionDialog(simpledialog.Dialog):
    """Configuration dialog for Set 8.3: Feature projection (t-SNE / UMAP)."""

    def __init__(self, parent: tk.Misc) -> None:
        self.discovered = discover_feature_files()
        super().__init__(parent, title="Feature projection — t-SNE & UMAP (Set 8.3)")

    def body(self, master: tk.Frame) -> tk.Widget:
        d = FeatureProjectionSettings()
        default_file = str(self.discovered[0]) if self.discovered else ""
        self.feature_path = tk.StringVar(value=default_file)
        self.method = tk.StringVar(value=d.method)
        self.max_samples = tk.StringVar(value=str(d.max_samples))
        self.perplexity = tk.StringVar(value=str(d.tsne_perplexity))
        self.max_iter = tk.StringVar(value=str(d.tsne_max_iter))
        self.n_neighbors = tk.StringVar(value=str(d.umap_n_neighbors))
        self.min_dist = tk.StringVar(value=str(d.umap_min_dist))

        file_frame = ttk.LabelFrame(master, text="Extracted test-set feature vectors (.npz)")
        file_frame.grid(row=0, column=0, columnspan=2, sticky="we", pady=(0, 6))
        if self.discovered:
            self._preset = tk.StringVar(value=self.discovered[0].name)
            combo = ttk.Combobox(
                file_frame,
                textvariable=self._preset,
                values=[p.name for p in self.discovered],
                state="readonly",
                width=42,
            )
            combo.grid(row=0, column=0, padx=6, pady=(4, 2), sticky="we")
            combo.bind("<<ComboboxSelected>>", self._on_preset_selected)
            ttk.Label(file_frame, text="models/set8").grid(row=0, column=1, padx=(0, 6))

        entry = ttk.Entry(file_frame, textvariable=self.feature_path, width=50)
        entry.grid(row=1, column=0, padx=6, pady=4)
        ttk.Button(file_frame, text="Browse…", command=self._browse).grid(row=1, column=1, padx=(0, 6))

        method_frame = ttk.LabelFrame(master, text="Projection method")
        method_frame.grid(row=1, column=0, columnspan=2, sticky="we", pady=(0, 6))
        for idx, (key, label) in enumerate(PROJECTION_METHODS.items()):
            ttk.Radiobutton(method_frame, text=label, value=key, variable=self.method).grid(
                row=idx, column=0, sticky="w", padx=6, pady=1
            )

        tsne_frame = ttk.LabelFrame(master, text="t-SNE parameters")
        tsne_frame.grid(row=2, column=0, columnspan=2, sticky="we", pady=(0, 6))
        spin_row(tsne_frame, 0, "Perplexity:", self.perplexity, PERPLEXITY_RANGE, 5)
        spin_row(tsne_frame, 1, "Iterations:", self.max_iter, TSNE_ITER_RANGE, 50)

        umap_frame = ttk.LabelFrame(master, text="UMAP parameters")
        umap_frame.grid(row=3, column=0, columnspan=2, sticky="we", pady=(0, 6))
        spin_row(umap_frame, 0, "n_neighbors:", self.n_neighbors, UMAP_NEIGHBORS_RANGE, 5)
        spin_row(umap_frame, 1, "min_dist:", self.min_dist, UMAP_MINDIST_RANGE, 0.05)

        sample_frame = ttk.LabelFrame(master, text="Subsampling (optional)")
        sample_frame.grid(row=4, column=0, columnspan=2, sticky="we")
        spin_row(sample_frame, 0, "Max samples (0 = all):", self.max_samples, MAX_SAMPLES_RANGE, 100)
        return entry

    def _on_preset_selected(self, _event: object) -> None:
        name = self._preset.get()
        for path in self.discovered:
            if path.name == name:
                self.feature_path.set(str(path))
                break

    def _browse(self) -> None:
        initial = str(DEFAULT_SET8_DIR) if DEFAULT_SET8_DIR.is_dir() else str(PROJECT_ROOT)
        chosen = filedialog.askopenfilename(
            title="Select extracted feature vectors (.npz)",
            initialdir=initial,
            filetypes=FEATURE_FILETYPES,
            parent=self,
        )
        if chosen:
            self.feature_path.set(chosen)

    def validate(self) -> bool:
        try:
            path = self.feature_path.get().strip()
            if not path or not Path(path).is_file():
                raise ValueError("Choose a valid .npz feature file (generated in Set 8.1 / 8.3).")
            self._settings = FeatureProjectionSettings(
                feature_path=path,
                method=self.method.get(),
                max_samples=read_number(self.max_samples, "Max samples", MAX_SAMPLES_RANGE, int),
                tsne_perplexity=read_number(self.perplexity, "Perplexity", PERPLEXITY_RANGE, float),
                tsne_max_iter=read_number(self.max_iter, "Iterations", TSNE_ITER_RANGE, int),
                umap_n_neighbors=read_number(self.n_neighbors, "n_neighbors", UMAP_NEIGHBORS_RANGE, int),
                umap_min_dist=read_number(self.min_dist, "min_dist", UMAP_MINDIST_RANGE, float),
            )
        except ValueError as error:
            messagebox.showerror(self.title(), str(error), parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self._settings


class FeatureProjectionTool(ForensicsTool):
    tool_id = "feature_projection"
    title = "Feature projection (t-SNE / UMAP)"
    category = "Set 8: Deep Learning & Feature Projection"
    description = (
        "Load extracted test-set CNN feature vectors (.npz), project them onto 2D space "
        "using t-SNE and/or UMAP, and visualize class clusters."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        settings = FeatureProjectionDialog(parent).result
        if settings is None:
            return None

        raw_dataset = load_feature_vectors(Path(settings.feature_path))
        dataset = subsample_stratified(raw_dataset, settings.max_samples)

        projections: dict[str, np.ndarray] = {}
        with ProgressPopup(parent, "Feature projection (Set 8.3)") as progress:
            if settings.method in ("both", "tsne"):
                projections["t-SNE"] = progress.run(
                    f"Running t-SNE on {dataset.n_samples} × {dataset.n_dims} features…",
                    lambda: project_tsne(
                        dataset.features,
                        perplexity=settings.tsne_perplexity,
                        max_iter=settings.tsne_max_iter,
                    ),
                )
            if settings.method in ("both", "umap"):
                projections["UMAP"] = progress.run(
                    f"Running UMAP on {dataset.n_samples} × {dataset.n_dims} features…",
                    lambda: project_umap(
                        dataset.features,
                        n_neighbors=settings.umap_n_neighbors,
                        min_dist=settings.umap_min_dist,
                    ),
                )

        subtitle = (
            f"{dataset.model_name} · layer '{dataset.layer_name}' · "
            f"{dataset.n_samples} samples ({dataset.n_dims}-D → 2-D)"
        )
        fig = projection_figure(projections, dataset.labels, dataset.class_names, subtitle=subtitle)

        if isinstance(parent, tk.Misc):
            show_figure(parent, f"Feature Projection — {dataset.model_name}", fig)

        details: dict[str, Any] = {
            "Operation": "Feature Projection (Set 8.3)",
            "Feature file": settings.feature_path,
            "Model": dataset.model_name,
            "Layer": dataset.layer_name,
            "Dataset": dataset.dataset,
            "Samples projected": f"{dataset.n_samples} / {raw_dataset.n_samples}",
            "Feature dimensionality": dataset.n_dims,
            "Classes": ", ".join(dataset.class_names),
            "Methods": ", ".join(projections.keys()),
        }
        if "t-SNE" in projections:
            details["t-SNE params"] = f"perplexity={settings.tsne_perplexity}, max_iter={settings.tsne_max_iter}"
        if "UMAP" in projections:
            details["UMAP params"] = f"n_neighbors={settings.umap_n_neighbors}, min_dist={settings.umap_min_dist}"

        return ToolResult(
            image=figure_to_image(fig),
            message=f"Projected {dataset.n_samples} feature vectors ({dataset.n_dims}-D → 2-D) with {' & '.join(projections)}.",
            details=details,
        )


__all__ = [
    "FEATURE_FILETYPES",
    "FeatureDataset",
    "FeatureProjectionDialog",
    "FeatureProjectionSettings",
    "FeatureProjectionTool",
    "PROJECTION_METHODS",
    "discover_feature_files",
    "load_feature_vectors",
    "project_tsne",
    "project_umap",
    "projection_figure",
    "subsample_stratified",
]
