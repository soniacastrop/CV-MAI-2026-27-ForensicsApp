"""Set 8.2: Load a pretrained CNN model, display its Model Passport, and classify images."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, UnidentifiedImageError

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .eigenfaces import figure_to_image
from .progress import ProgressPopup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SET8_DIR = PROJECT_ROOT / "models" / "set8"
DEFAULT_SAMPLE_IMG_DIR = DEFAULT_SET8_DIR / "sample_images"

MODEL_FILETYPES = [("Keras model", "*.keras *.h5"), ("All files", "*.*")]
PASSPORT_FILETYPES = [("Model passport (JSON)", "*.json"), ("All files", "*.*")]
IMAGE_FILETYPES = [
    ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]

MODES = {
    "classify": "Display model passport & classify image",
    "passport_only": "Display model passport & training history only",
}

# Cache loaded Keras models by resolved path so repeated classifications are fast.
_MODEL_CACHE: dict[Path, Any] = {}


@dataclass(frozen=True)
class ModelPassport:
    """Metadata describing a trained classification model (Set 8.1 / 8.2)."""

    model_name: str
    architecture: str = "Unknown"
    model_file: str = ""
    created_at: str = ""
    framework: str = ""
    dataset: str = "Unknown"
    sample_size: dict[str, int] = field(default_factory=dict)
    class_names: tuple[str, ...] = ()
    preprocessing: dict[str, Any] = field(default_factory=dict)
    training_history: dict[str, list[float]] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    passport_path: Path | None = None

    @property
    def input_height(self) -> int:
        return int(self.preprocessing.get("input_height", 32))

    @property
    def input_width(self) -> int:
        return int(self.preprocessing.get("input_width", 32))

    @property
    def channels(self) -> int:
        return int(self.preprocessing.get("channels", 3))

    @property
    def color_mode(self) -> str:
        return str(self.preprocessing.get("color_mode", "RGB" if self.channels == 3 else "L"))

    @property
    def normalization(self) -> str:
        return str(self.preprocessing.get("normalization", "rescale_0_1"))

    @property
    def preprocessing_description(self) -> str:
        desc = self.preprocessing.get("description")
        if desc:
            return str(desc)
        return f"{self.color_mode}, resize to {self.input_width}x{self.input_height}, normalization={self.normalization}"

    def summary_lines(self) -> list[tuple[str, str]]:
        """Key-value lines suitable for a passport dialog or sidebar details panel."""
        lines: list[tuple[str, str]] = [
            ("Model name", self.model_name),
            ("Architecture", self.architecture),
            ("Dataset", self.dataset),
        ]
        if self.sample_size:
            parts = [f"{k}: {v:,}" for k, v in self.sample_size.items()]
            lines.append(("Sample size", ", ".join(parts)))
        if self.class_names:
            lines.append(("Classes", f"{len(self.class_names)} ({', '.join(self.class_names)})"))
        lines.append(("Input shape", f"{self.input_height} × {self.input_width} × {self.channels} ({self.color_mode})"))
        lines.append(("Preprocessing", self.preprocessing_description))
        if self.training_history:
            epochs = max((len(v) for v in self.training_history.values()), default=0)
            hist_parts = [f"{epochs} epochs"]
            if self.training_history.get("accuracy"):
                hist_parts.append(f"final train acc={self.training_history['accuracy'][-1]:.3f}")
            if self.training_history.get("val_accuracy"):
                hist_parts.append(f"final val acc={self.training_history['val_accuracy'][-1]:.3f}")
            lines.append(("Training history", ", ".join(hist_parts)))
        if self.test_metrics:
            metric_parts = [f"{k}={v:.4f}" for k, v in self.test_metrics.items()]
            lines.append(("Test metrics", ", ".join(metric_parts)))
        if self.framework:
            lines.append(("Framework", self.framework))
        if self.created_at:
            lines.append(("Created", self.created_at))
        return lines


@dataclass(frozen=True)
class PretrainedCnnSettings:
    model_path: str = ""
    passport_path: str = ""
    mode: str = "classify"
    image_path: str = ""  # empty = use currently open canvas image


# ---------------------------------------------------------------- discovery & loading


def discover_set8_models(folder: Path = DEFAULT_SET8_DIR) -> list[Path]:
    """Return sorted ``.keras`` / ``.h5`` model files found in ``folder``."""
    if not folder.is_dir():
        return []
    found = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".keras", ".h5")]
    return sorted(found, key=lambda p: p.name.lower())


def find_passport_for_model(model_path: Path) -> Path | None:
    """Locate the matching ``<stem>_passport.json`` (or ``passport.json``) next to ``model_path``."""
    candidates = [
        model_path.with_name(f"{model_path.stem}_passport.json"),
        model_path.with_suffix(".json"),
        model_path.parent / "passport.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_model_passport(passport_path: Path) -> ModelPassport:
    """Load a ``ModelPassport`` from a JSON file created in Set 8.1."""
    raw = json.loads(passport_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid model passport JSON: {passport_path}")
    sample_size = {str(k): int(v) for k, v in raw.get("sample_size", {}).items()}
    class_names = tuple(str(c) for c in raw.get("class_names", ()))
    history = {
        str(k): [float(x) for x in vals]
        for k, vals in raw.get("training_history", {}).items()
        if isinstance(vals, list)
    }
    test_metrics = {str(k): float(v) for k, v in raw.get("test_metrics", {}).items()}
    return ModelPassport(
        model_name=str(raw.get("model_name", passport_path.stem)),
        architecture=str(raw.get("architecture", "Unknown")),
        model_file=str(raw.get("model_file", "")),
        created_at=str(raw.get("created_at", "")),
        framework=str(raw.get("framework", "")),
        dataset=str(raw.get("dataset", "Unknown")),
        sample_size=sample_size,
        class_names=class_names,
        preprocessing=dict(raw.get("preprocessing", {})),
        training_history=history,
        test_metrics=test_metrics,
        passport_path=passport_path,
    )


def fallback_passport_for_model(model_path: Path, model: Any | None = None) -> ModelPassport:
    """Synthesize a minimal passport when a model has no companion JSON file."""
    height, width, channels = 32, 32, 3
    class_names: tuple[str, ...] = ()
    if model is not None:
        in_shape = getattr(model, "input_shape", None)
        if isinstance(in_shape, tuple) and len(in_shape) == 4:
            height = int(in_shape[1] or height)
            width = int(in_shape[2] or width)
            channels = int(in_shape[3] or channels)
        out_shape = getattr(model, "output_shape", None)
        if isinstance(out_shape, tuple) and len(out_shape) >= 2 and out_shape[-1]:
            class_names = tuple(f"class_{i}" for i in range(int(out_shape[-1])))
    return ModelPassport(
        model_name=model_path.stem,
        model_file=model_path.name,
        class_names=class_names,
        preprocessing={
            "input_height": height,
            "input_width": width,
            "channels": channels,
            "color_mode": "RGB" if channels == 3 else "L",
            "normalization": "rescale_0_1",
        },
    )


def load_keras_model(model_path: Path, *, use_cache: bool = True) -> Any:
    """Load a saved Keras ``.keras`` / ``.h5`` model (lazy-importing TensorFlow)."""
    resolved = model_path.resolve()
    if use_cache and resolved in _MODEL_CACHE:
        return _MODEL_CACHE[resolved]
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow / Keras is required to load pretrained CNN models. "
            "Install it with: pip install tensorflow"
        ) from exc
    model = keras.models.load_model(resolved)
    if use_cache:
        _MODEL_CACHE[resolved] = model
    return model


# ---------------------------------------------------------------- preprocessing & inference


def preprocess_for_passport(image: Image.Image, passport: ModelPassport) -> np.ndarray:
    """Convert ``image`` into a ``(1, H, W, C)`` float32 batch matching ``passport.preprocessing``."""
    mode = passport.color_mode.upper()
    converted = image.convert("L" if mode == "L" or passport.channels == 1 else "RGB")
    resized = converted.resize((passport.input_width, passport.input_height), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]

    norm = passport.normalization.lower()
    if norm in ("rescale_0_1", "0_1", "divide_255"):
        arr = arr / 255.0
    elif norm in ("inception_v3_minus1_1", "minus1_1", "rescale_-1_1"):
        arr = (arr / 127.5) - 1.0
    elif norm in ("none", "raw"):
        pass
    else:
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0)


def classify_image_with_model(
    image: Image.Image,
    model: Any,
    passport: ModelPassport,
) -> tuple[str, float, dict[str, float]]:
    """Run ``model`` on ``image`` and return ``(predicted_class, top_confidence, prob_by_class)``."""
    batch = preprocess_for_passport(image, passport)
    raw_preds = np.asarray(model.predict(batch, verbose=0), dtype=np.float64)
    probs = raw_preds[0] if raw_preds.ndim >= 2 else raw_preds.ravel()

    # If the output does not look like probabilities (e.g. logits), apply softmax.
    if probs.min() < 0.0 or not np.isclose(probs.sum(), 1.0, atol=1e-2):
        shifted = probs - np.max(probs)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp)

    names = list(passport.class_names)
    if len(names) != len(probs):
        names = [f"class_{i}" for i in range(len(probs))]

    prob_map = {name: float(p) for name, p in zip(names, probs)}
    best_idx = int(np.argmax(probs))
    return names[best_idx], float(probs[best_idx]), prob_map


# ---------------------------------------------------------------- figures


def passport_history_figure(passport: ModelPassport) -> Figure:
    """Plot training & validation loss and accuracy curves from ``passport.training_history``."""
    hist = passport.training_history
    fig = Figure(figsize=(8.4, 3.4), layout="constrained")
    if not hist:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No training history stored in passport.", ha="center", va="center")
        ax.axis("off")
        return fig

    ax_loss = fig.add_subplot(121)
    if "loss" in hist:
        epochs = range(1, len(hist["loss"]) + 1)
        ax_loss.plot(epochs, hist["loss"], marker="o", markersize=3, label="train")
    if "val_loss" in hist:
        epochs = range(1, len(hist["val_loss"]) + 1)
        ax_loss.plot(epochs, hist["val_loss"], marker="o", markersize=3, label="validation")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.3)
    if ax_loss.lines:
        ax_loss.legend()

    ax_acc = fig.add_subplot(122)
    if "accuracy" in hist:
        epochs = range(1, len(hist["accuracy"]) + 1)
        ax_acc.plot(epochs, hist["accuracy"], marker="o", markersize=3, label="train")
    if "val_accuracy" in hist:
        epochs = range(1, len(hist["val_accuracy"]) + 1)
        ax_acc.plot(epochs, hist["val_accuracy"], marker="o", markersize=3, label="validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.grid(alpha=0.3)
    if ax_acc.lines:
        ax_acc.legend()

    fig.suptitle(f"Training history — {passport.model_name}", fontsize=11)
    return fig


def classification_summary_figure(
    image: Image.Image,
    predicted_class: str,
    confidence: float,
    probabilities: dict[str, float],
    passport: ModelPassport,
    *,
    top_k: int = 5,
) -> Figure:
    """Side-by-side figure: input image with predicted label and top-k probability bar chart."""
    fig = Figure(figsize=(8.6, 4.2), layout="constrained")
    ax_img = fig.add_subplot(121)
    ax_img.imshow(np.asarray(image.convert("RGB")))
    ax_img.set_title(f"Predicted: {predicted_class} ({confidence:.1%})", fontsize=12, fontweight="bold")
    ax_img.set_xticks(())
    ax_img.set_yticks(())

    ax_bar = fig.add_subplot(122)
    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    labels = [k for k, _ in ranked][::-1]
    values = [v for _, v in ranked][::-1]
    colors = ["tab:blue"] * (len(labels) - 1) + ["tab:green"] if labels else []
    bars = ax_bar.barh(labels, values, color=colors)
    ax_bar.set_xlim(0.0, 1.05)
    ax_bar.set_xlabel("Probability")
    ax_bar.set_title(f"Top-{len(labels)} predictions ({passport.model_name})", fontsize=10)
    ax_bar.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, values):
        ax_bar.text(min(val + 0.02, 1.02), bar.get_y() + bar.get_height() / 2, f"{val:.1%}", va="center", fontsize=9)
    return fig


# ---------------------------------------------------------------- UI: passport popup & dialog


class ModelPassportPopup(tk.Toplevel):
    """Non-blocking window displaying the Model Passport metadata and training curves."""

    def __init__(self, parent: tk.Misc, passport: ModelPassport) -> None:
        super().__init__(parent)
        self.title(f"Model Passport — {passport.model_name}")
        self.geometry("880x640")

        info_frame = ttk.LabelFrame(self, text="Model Passport (Set 8.1 / 8.2)", padding=8)
        info_frame.pack(fill="x", padx=10, pady=(10, 6))

        for row_idx, (key, value) in enumerate(passport.summary_lines()):
            ttk.Label(info_frame, text=f"{key}:", font=("TkDefaultFont", 9, "bold")).grid(
                row=row_idx, column=0, sticky="nw", padx=(0, 8), pady=1
            )
            ttk.Label(info_frame, text=value, wraplength=680, justify="left").grid(
                row=row_idx, column=1, sticky="nw", pady=1
            )

        plot_frame = ttk.LabelFrame(self, text="Training History", padding=4)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.figure = passport_history_figure(passport)
        canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


class PretrainedCnnDialog(simpledialog.Dialog):
    """Configuration dialog for Set 8.2: select model, passport, mode, and optional image."""

    def __init__(self, parent: tk.Misc, has_canvas_image: bool) -> None:
        self.has_canvas_image = has_canvas_image
        self.discovered = discover_set8_models()
        super().__init__(parent, title="Pretrained CNN & Model Passport (Set 8.2)")

    def body(self, master: tk.Frame) -> tk.Widget:
        default_model = str(self.discovered[0]) if self.discovered else ""
        default_passport = ""
        if self.discovered:
            found_passport = find_passport_for_model(self.discovered[0])
            if found_passport is not None:
                default_passport = str(found_passport)

        self.model_path = tk.StringVar(value=default_model)
        self.passport_path = tk.StringVar(value=default_passport)
        self.mode = tk.StringVar(value="classify")
        self.image_path = tk.StringVar(value="")

        # Quick-pick dropdown for models in models/set8/
        model_frame = ttk.LabelFrame(master, text="Pretrained Keras model (.keras / .h5)")
        model_frame.grid(row=0, column=0, columnspan=2, sticky="we", pady=(0, 6))
        if self.discovered:
            self._preset = tk.StringVar(value=self.discovered[0].name)
            combo = ttk.Combobox(
                model_frame,
                textvariable=self._preset,
                values=[p.name for p in self.discovered],
                state="readonly",
                width=44,
            )
            combo.grid(row=0, column=0, padx=6, pady=(4, 2), sticky="we")
            combo.bind("<<ComboboxSelected>>", self._on_preset_selected)
            ttk.Label(model_frame, text="models/set8").grid(row=0, column=1, padx=(0, 6))

        entry = ttk.Entry(model_frame, textvariable=self.model_path, width=52)
        entry.grid(row=1, column=0, padx=6, pady=4)
        ttk.Button(model_frame, text="Browse…", command=self._browse_model).grid(row=1, column=1, padx=(0, 6))

        # Passport JSON path
        pass_frame = ttk.LabelFrame(master, text="Model passport (.json, auto-detected if next to model)")
        pass_frame.grid(row=1, column=0, columnspan=2, sticky="we", pady=(0, 6))
        ttk.Entry(pass_frame, textvariable=self.passport_path, width=52).grid(row=0, column=0, padx=6, pady=4)
        ttk.Button(pass_frame, text="Browse…", command=self._browse_passport).grid(row=0, column=1, padx=(0, 6))

        # Action mode
        mode_frame = ttk.LabelFrame(master, text="Action")
        mode_frame.grid(row=2, column=0, columnspan=2, sticky="we", pady=(0, 6))
        for idx, (key, label) in enumerate(MODES.items()):
            ttk.Radiobutton(mode_frame, text=label, value=key, variable=self.mode).grid(
                row=idx, column=0, sticky="w", padx=6, pady=1
            )

        # Optional image file override (or required if no image is open on canvas)
        img_label = (
            "Image to classify (leave blank to use current canvas image)"
            if self.has_canvas_image
            else "Image to classify (required when no image is open on canvas)"
        )
        img_frame = ttk.LabelFrame(master, text=img_label)
        img_frame.grid(row=3, column=0, columnspan=2, sticky="we")
        ttk.Entry(img_frame, textvariable=self.image_path, width=52).grid(row=0, column=0, padx=6, pady=4)
        ttk.Button(img_frame, text="Browse…", command=self._browse_image).grid(row=0, column=1, padx=(0, 6))
        return entry

    def _on_preset_selected(self, _event: object) -> None:
        name = self._preset.get()
        for path in self.discovered:
            if path.name == name:
                self.model_path.set(str(path))
                passport = find_passport_for_model(path)
                self.passport_path.set(str(passport) if passport else "")
                break

    def _browse_model(self) -> None:
        initial = str(DEFAULT_SET8_DIR) if DEFAULT_SET8_DIR.is_dir() else str(PROJECT_ROOT)
        chosen = filedialog.askopenfilename(
            title="Select pretrained Keras model",
            initialdir=initial,
            filetypes=MODEL_FILETYPES,
            parent=self,
        )
        if chosen:
            path = Path(chosen)
            self.model_path.set(chosen)
            passport = find_passport_for_model(path)
            if passport is not None:
                self.passport_path.set(str(passport))

    def _browse_passport(self) -> None:
        initial = str(DEFAULT_SET8_DIR) if DEFAULT_SET8_DIR.is_dir() else str(PROJECT_ROOT)
        chosen = filedialog.askopenfilename(
            title="Select model passport JSON",
            initialdir=initial,
            filetypes=PASSPORT_FILETYPES,
            parent=self,
        )
        if chosen:
            self.passport_path.set(chosen)

    def _browse_image(self) -> None:
        initial = str(DEFAULT_SAMPLE_IMG_DIR) if DEFAULT_SAMPLE_IMG_DIR.is_dir() else str(PROJECT_ROOT)
        chosen = filedialog.askopenfilename(
            title="Select image to classify",
            initialdir=initial,
            filetypes=IMAGE_FILETYPES,
            parent=self,
        )
        if chosen:
            self.image_path.set(chosen)

    def validate(self) -> bool:
        try:
            model_str = self.model_path.get().strip()
            passport_str = self.passport_path.get().strip()
            mode = self.mode.get()
            img_str = self.image_path.get().strip()

            if mode == "classify":
                if not model_str or not Path(model_str).is_file():
                    raise ValueError("Choose a valid pretrained model file (.keras or .h5).")
                if not img_str and not self.has_canvas_image:
                    raise ValueError("Open an image on the canvas or browse for an image to classify.")
                if img_str and not Path(img_str).is_file():
                    raise ValueError("The selected image file does not exist.")
            else:
                if not passport_str and model_str and Path(model_str).is_file():
                    found = find_passport_for_model(Path(model_str))
                    if found is not None:
                        passport_str = str(found)
                if not passport_str or not Path(passport_str).is_file():
                    raise ValueError("Choose a valid model passport JSON file.")

            if passport_str and not Path(passport_str).is_file():
                raise ValueError("The selected model passport JSON file does not exist.")

            self._settings = PretrainedCnnSettings(
                model_path=model_str,
                passport_path=passport_str,
                mode=mode,
                image_path=img_str,
            )
        except ValueError as error:
            messagebox.showerror(self.title(), str(error), parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self._settings


# ---------------------------------------------------------------- tool class


class PretrainedCnnTool(ForensicsTool):
    tool_id = "pretrained_cnn"
    title = "Pretrained CNN & Model Passport"
    category = "Set 8: Deep Learning & Feature Projection"
    description = (
        "Load a pretrained CNN (.keras / .h5), inspect its Model Passport "
        "(dataset, sample size, classes, preprocessing, training history), "
        "and classify an image."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        settings = PretrainedCnnDialog(parent, has_canvas_image=document.current is not None).result
        if settings is None:
            return None

        passport: ModelPassport | None = None
        if settings.passport_path:
            passport = load_model_passport(Path(settings.passport_path))
        elif settings.model_path:
            found = find_passport_for_model(Path(settings.model_path))
            if found is not None:
                passport = load_model_passport(found)

        if settings.mode == "passport_only":
            assert passport is not None
            if isinstance(parent, tk.Misc):
                popup = ModelPassportPopup(parent, passport)
                parent.winfo_toplevel()._set8_passport_popup = popup
            details = {"Operation": "Model Passport (Set 8.2)"} | dict(passport.summary_lines())
            return ToolResult(
                image=figure_to_image(passport_history_figure(passport)),
                message=f"Displayed model passport for {passport.model_name}.",
                details=details,
            )

        # Load image to classify (selected file overrides canvas image)
        if settings.image_path:
            try:
                with Image.open(settings.image_path) as opened:
                    query_image = opened.copy()
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError(f"Could not open image: {exc}") from exc
            image_source = settings.image_path
        else:
            assert document.current is not None
            query_image = document.current
            image_source = str(document.path) if document.path else "Current canvas image"

        model_path = Path(settings.model_path)
        with ProgressPopup(parent, "Pretrained CNN Classification") as progress:
            model = progress.run(
                f"Loading pretrained model ({model_path.name})…",
                lambda: load_keras_model(model_path),
            )
            if passport is None:
                passport = fallback_passport_for_model(model_path, model)
            predicted_class, confidence, probabilities = progress.run(
                f"Classifying image with {passport.model_name}…",
                lambda: classify_image_with_model(query_image, model, passport),
            )

        if isinstance(parent, tk.Misc):
            popup = ModelPassportPopup(parent, passport)
            parent.winfo_toplevel()._set8_passport_popup = popup

        summary_fig = classification_summary_figure(
            query_image, predicted_class, confidence, probabilities, passport
        )
        details: dict[str, Any] = {
            "Operation": "Pretrained CNN Classification (Set 8.2)",
            "Image": image_source,
            "Predicted class": predicted_class,
            "Confidence": f"{confidence:.2%}",
        }
        for key, val in passport.summary_lines():
            details[f"Passport · {key}"] = val
        for cls_name, prob in sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[:5]:
            details[f"  p({cls_name})"] = f"{prob:.4f}"

        return ToolResult(
            image=figure_to_image(summary_fig),
            message=f"{passport.model_name}: {predicted_class} ({confidence:.1%})",
            details=details,
        )


__all__ = [
    "DEFAULT_SAMPLE_IMG_DIR",
    "DEFAULT_SET8_DIR",
    "MODES",
    "ModelPassport",
    "ModelPassportPopup",
    "PretrainedCnnDialog",
    "PretrainedCnnSettings",
    "PretrainedCnnTool",
    "classification_summary_figure",
    "classify_image_with_model",
    "discover_set8_models",
    "fallback_passport_for_model",
    "find_passport_for_model",
    "load_keras_model",
    "load_model_passport",
    "passport_history_figure",
    "preprocess_for_passport",
]
