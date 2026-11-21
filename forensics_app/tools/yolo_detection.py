"""Set 9: YOLO object detection (Ultralytics) — pretrained COCO model, our blood-cell model and analysis tools.

* 9.1  run a pretrained Ultralytics model (``yolo11n.pt``, 80 COCO classes) on an image (+ "why are people blue?");
* 9.3  run our blood-cell model (PC / RBC / WBC) and draw ground-truth vs predicted boxes;
* extras: cell counting, a gradient-free activation map of an inner layer, per-class box-size / colour histograms.

Ground truth is read from the YOLO dataset layout: ``.../images/<name>.jpg`` → ``.../labels/<name>.txt``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import random
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

import cv2
from matplotlib import colormaps
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .cnn_classification import IMAGE_FILETYPES, PROJECT_ROOT
from .dialog_fields import read_number, spin_row
from .eigenfaces import figure_to_image
from .progress import ProgressPopup

CATEGORY = "Set 9: Object Detection (YOLO)"

DEFAULT_SET9_DIR = PROJECT_ROOT / "models" / "set9"
NOTEBOOK_MODELS_DIR = PROJECT_ROOT / "notebooks" / "models"
DEFAULT_PRETRAINED_WEIGHTS = DEFAULT_SET9_DIR / "yolo11n.pt"  # downloaded here by Ultralytics on first use
DEFAULT_BLOOD_TEST_DIR = PROJECT_ROOT / "datasets" / "blood_cell" / "test" / "images"

MODEL_FILETYPES = [("YOLO weights", "*.pt *.onnx"), ("All files", "*.*")]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BLOOD_CLASS_COLORS = {"PC": "#d62728", "RBC": "#2ca02c", "WBC": "#1f77b4"}  # same colours as the notebook

CONF_RANGE = (0.01, 0.99)
IOU_RANGE = (0.1, 0.95)
SAMPLE_RANGE = (1, 5000)

_MODEL_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------- data containers


@dataclass(frozen=True)
class Detections:
    """Boxes in pixel ``xyxy`` coordinates, one row per object."""

    boxes: np.ndarray  # (N, 4) float
    classes: np.ndarray  # (N,) int
    names: dict[int, str]
    scores: np.ndarray | None = None  # (N,) float; None for ground truth

    def __len__(self) -> int:
        return int(len(self.classes))

    def counts(self) -> Counter[str]:
        return Counter(self.names.get(int(c), str(int(c))) for c in self.classes)

    def above(self, conf: float) -> Detections:
        if self.scores is None:
            return self
        keep = self.scores >= conf
        return Detections(self.boxes[keep], self.classes[keep], self.names, self.scores[keep])


def empty_detections(names: dict[int, str]) -> Detections:
    return Detections(np.zeros((0, 4)), np.zeros(0, dtype=int), names)


# ---------------------------------------------------------------- model discovery & loading


def discover_yolo_models(folders: tuple[Path, ...] = (DEFAULT_SET9_DIR, NOTEBOOK_MODELS_DIR)) -> list[Path]:
    """``.pt`` weight files in ``models/set9`` and ``notebooks/models`` (our trained models)."""
    found: list[Path] = []
    for folder in folders:
        if folder.is_dir():
            found.extend(p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in (".pt", ".onnx"))
    return found


def default_custom_model(models: list[Path] | None = None) -> str:
    """Our blood-cell model: the first discovered weight file that is not a stock ``yolo*`` checkpoint."""
    models = discover_yolo_models() if models is None else models
    custom = [p for p in models if not p.stem.lower().startswith("yolo")]
    return str(custom[0]) if custom else ""


def load_yolo_model(weights: str | Path, *, use_cache: bool = True) -> Any:
    """``ultralytics.YOLO(weights)``, cached; stock names such as ``yolo11n.pt`` are downloaded on first use."""
    key = str(weights)
    if use_cache and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for the Set 9 tools. Install it with: pip install ultralytics") from exc
    path = Path(weights)
    if path.parent != Path(".") and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)  # Ultralytics downloads stock weights into this folder
    model = YOLO(str(path))
    if use_cache:
        _MODEL_CACHE[key] = model
    return model


def model_names(model: Any) -> dict[int, str]:
    return {int(k): str(v) for k, v in dict(model.names).items()}


def predict(model: Any, image: Image.Image, *, conf: float = 0.25, iou: float = 0.7) -> tuple[Detections, Any]:
    """Run ``model`` on a PIL image; returns the detections and the raw Ultralytics ``Results``."""
    result = model.predict(image.convert("RGB"), conf=conf, iou=iou, verbose=False)[0]
    boxes = result.boxes
    detections = Detections(
        boxes=boxes.xyxy.cpu().numpy().astype(float),
        classes=boxes.cls.cpu().numpy().astype(int),
        names=model_names(model),
        scores=boxes.conf.cpu().numpy().astype(float),
    )
    return detections, result


# ---------------------------------------------------------------- ground-truth labels


def label_path_for(image_path: Path) -> Path:
    """YOLO layout: ``<split>/images/<name>.jpg`` → ``<split>/labels/<name>.txt`` (else next to the image)."""
    image_path = Path(image_path)
    if image_path.parent.name == "images":
        return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    return image_path.with_suffix(".txt")


def read_yolo_labels(label_path: Path, width: int, height: int, names: dict[int, str]) -> Detections:
    """YOLO txt (``cls xc yc w h`` normalised to [0, 1]) → pixel ``xyxy`` boxes."""
    label_path = Path(label_path)
    if not label_path.is_file():
        return empty_detections(names)
    rows = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return empty_detections(names)
    a = np.array([r[:5] for r in rows], dtype=float)
    xc, yc, w, h = a[:, 1] * width, a[:, 2] * height, a[:, 3] * width, a[:, 4] * height
    boxes = np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)
    return Detections(boxes, a[:, 0].astype(int), names)


def ground_truth_for(image_path: Path | None, image: Image.Image, names: dict[int, str]) -> Detections | None:
    """GT boxes for ``image_path`` if a label file exists, else ``None``."""
    if image_path is None:
        return None
    label_path = label_path_for(image_path)
    if not label_path.is_file():
        return None
    return read_yolo_labels(label_path, image.width, image.height, names)


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def open_rgb(path: Path) -> Image.Image:
    try:
        with Image.open(path) as opened:
            return opened.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Could not open image {path}: {exc}") from exc


# ---------------------------------------------------------------- drawing


def class_color(name: str, class_id: int) -> str:
    if name in BLOOD_CLASS_COLORS:
        return BLOOD_CLASS_COLORS[name]
    r, g, b, _ = colormaps["tab20"](class_id % 20)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: Detections,
    *,
    dashed: bool = False,
    width: int | None = None,
    show_labels: bool = True,
) -> Image.Image:
    """Draw boxes coloured by class (with ``label conf`` tags) on a copy of ``image``."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    width = width or max(2, round(max(out.size) / 400))
    font = _font(max(11, round(max(out.size) / 55)))
    for k, ((x1, y1, x2, y2), cls) in enumerate(zip(detections.boxes, detections.classes)):
        name = detections.names.get(int(cls), str(int(cls)))
        color = class_color(name, int(cls))
        if dashed:
            _dashed_rectangle(draw, (x1, y1, x2, y2), color, width)
        else:
            draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        if show_labels:
            text = name if detections.scores is None else f"{name} {detections.scores[k]:.2f}"
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            tw, th = right - left, bottom - top
            ty = y1 - th - 4 if y1 - th - 4 >= 0 else y1
            draw.rectangle((x1, ty, x1 + tw + 4, ty + th + 4), fill=color)
            draw.text((x1 + 2 - left, ty + 2 - top), text, fill="white", font=font)
    return out


def _dashed_rectangle(draw: ImageDraw.ImageDraw, box: tuple[float, ...], color: str, width: int, dash: int = 8) -> None:
    x1, y1, x2, y2 = box
    for (ax, ay), (bx, by) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        length = max(abs(bx - ax), abs(by - ay))
        steps = max(int(length // dash), 1)
        for s in range(0, steps, 2):
            t0, t1 = s / steps, min((s + 1) / steps, 1.0)
            draw.line((ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1),
                      fill=color, width=width)


def gt_vs_pred_image(image: Image.Image, gt: Detections | None, pred: Detections) -> Image.Image:
    """Ground truth as thick solid boxes (no tags), predictions as thin dashed boxes with confidence."""
    base = image.convert("RGB")
    if gt is not None:
        base = draw_detections(base, gt, width=max(3, round(max(base.size) / 250)), show_labels=False)
    return draw_detections(base, pred, dashed=True)


def bgr_explanation_figure(result: Any) -> Figure:
    """"Why are people blue?": ``Results.plot()`` returns BGR; shown as RGB the red and blue channels swap."""
    bgr = result.plot()
    fig = Figure(figsize=(11, 5.4), layout="constrained")
    for k, (img, title) in enumerate(
        ((bgr, "plot() shown as-is (BGR read as RGB)"), (bgr[..., ::-1], "channels reversed → correct RGB"))
    ):
        ax = fig.add_subplot(1, 2, k + 1)
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Why are people blue? OpenCV / Ultralytics images are BGR, matplotlib expects RGB", fontsize=11)
    return fig


# ---------------------------------------------------------------- analysis: counting


def count_table(names: dict[int, str], pred: Detections, gt: Detections | None) -> dict[str, tuple[int, int | None]]:
    """``{class: (predicted, ground_truth or None)}`` for every class of the model."""
    p, g = pred.counts(), gt.counts() if gt is not None else None
    return {n: (p.get(n, 0), None if g is None else g.get(n, 0)) for n in names.values()}


def count_figure(annotated: Image.Image, table: dict[str, tuple[int, int | None]], title: str) -> Figure:
    """Annotated image next to a grouped bar chart of predicted (and GT) counts per class."""
    fig = Figure(figsize=(12, 5.2), layout="constrained")
    ax_img = fig.add_subplot(1, 2, 1)
    ax_img.imshow(np.asarray(annotated))
    ax_img.set_title(title, fontsize=10)
    ax_img.axis("off")

    ax = fig.add_subplot(1, 2, 2)
    names = list(table)
    x = np.arange(len(names))
    has_gt = any(gt is not None for _, gt in table.values())
    w = 0.38 if has_gt else 0.6
    colors = [class_color(n, i) for i, n in enumerate(names)]
    bars = ax.bar(x - (w / 2 if has_gt else 0), [p for p, _ in table.values()], w, color=colors, label="predicted")
    ax.bar_label(bars)
    if has_gt:
        gt_bars = ax.bar(x + w / 2, [g or 0 for _, g in table.values()], w, color=colors, alpha=0.4,
                         hatch="//", edgecolor="black", label="ground truth")
        ax.bar_label(gt_bars)
        ax.legend()
    ax.set_xticks(x, names)
    ax.set_ylabel("cells")
    ax.set_title("Cell count per class")
    ax.grid(axis="y", alpha=0.3)
    return fig


def count_scatter_figure(gt_counts: dict[str, list[int]], pred_counts: dict[str, list[int]], conf: float) -> Figure:
    """Notebook E3: predicted vs GT count per image and class, with MAE and exact-match rate."""
    names = list(gt_counts)
    fig = Figure(figsize=(5 * len(names), 4.6), layout="constrained")
    rng = np.random.default_rng(0)
    for k, name in enumerate(names):
        ax = fig.add_subplot(1, len(names), k + 1)
        g, p = np.array(gt_counts[name]), np.array(pred_counts[name])
        jitter = rng.uniform(-0.15, 0.15, size=(2, len(g)))
        ax.scatter(g + jitter[0], p + jitter[1], s=14, alpha=0.6, color=class_color(name, k))
        lim = max(int(g.max(initial=0)), int(p.max(initial=0))) + 1
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        mae = float(np.abs(g - p).mean()) if len(g) else 0.0
        exact = float(np.mean(g == p)) if len(g) else 0.0
        ax.set(xlabel="GT count", ylabel="predicted count", title=f"{name}: MAE={mae:.2f}, exact={exact:.0%}")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Cell counting on {len(next(iter(gt_counts.values()), []))} images (conf ≥ {conf:.2f})")
    return fig


# ---------------------------------------------------------------- analysis: activation map


def _network_layers(model: Any, image: Image.Image) -> Any:
    """The layer list of the network that ``predict()`` really runs (a fused copy living in ``model.predictor``)."""
    if getattr(model, "predictor", None) is None:
        model.predict(image.convert("RGB"), verbose=False)
    return model.predictor.model.model.model


def detection_scale_layers(model: Any, image: Image.Image) -> list[int]:
    """Indices of the feature maps fed to the Detect head (YOLO11: 16 = P3/8, 19 = P4/16, 22 = P5/32)."""
    head = _network_layers(model, image)[-1]
    inputs = getattr(head, "f", None)
    return [int(i) for i in inputs] if isinstance(inputs, (list, tuple)) else [16, 19, 22]


def activation_maps(
    model: Any, image: Image.Image, layer_idxs: list[int], *, conf: float = 0.25
) -> tuple[Detections, dict[int, np.ndarray]]:
    """Activation maps of several layers from a single forward pass (see :func:`activation_map`)."""
    rgb = image.convert("RGB")
    layers = _network_layers(model, rgb)
    for layer_idx in layer_idxs:
        if not 0 <= layer_idx < len(layers) - 1:
            raise ValueError(f"Layer index must be between 0 and {len(layers) - 2} (the last layer is the Detect head).")
    feats: dict[int, Any] = {}

    def make_hook(layer_idx: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            feats[layer_idx] = (output[0] if isinstance(output, (tuple, list)) else output).detach()

        return hook

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in layer_idxs]
    try:
        detections, _ = predict(model, rgb, conf=conf)
    finally:
        for handle in handles:
            handle.remove()
    return detections, {i: principal_activation(feats[i][0].float().cpu().numpy()) for i in layer_idxs}


def activation_map(model: Any, image: Image.Image, layer_idx: int, *, conf: float = 0.25) -> tuple[Detections, np.ndarray]:
    """Gradient-free activation map: first principal component of layer ``layer_idx``'s feature map (C × h × w).

    The (h·w) × C matrix of activations is centred and projected onto its dominant direction (SVD), giving the main
    spatial activation pattern of that layer, normalised to [0, 1].
    """
    detections, maps = activation_maps(model, image, [layer_idx], conf=conf)
    return detections, maps[layer_idx]


def principal_activation(A: np.ndarray) -> np.ndarray:
    """First principal component of a C × h × w feature map, oriented, ReLU'd and scaled to [0, 1]."""
    C, h, w = A.shape
    M = A.reshape(C, -1).T
    energy = np.abs(M).mean(axis=1)  # how strongly the layer fires at each location
    M = M - M.mean(axis=0)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    pc = M @ Vt[0]
    # The sign of a principal direction is arbitrary (and pc.sum() is always ~0 because M is centred, so it
    # cannot be used to pick it). Orient it so that "high" means where the layer's activations are strong.
    if np.corrcoef(pc, energy)[0, 1] < 0:
        pc = -pc
    cam = pc.reshape(h, w)
    cam = np.maximum(cam, 0)
    cam /= cam.max() + 1e-9
    return cam


def letterbox_crop(cam: np.ndarray, width: int, height: int) -> np.ndarray:
    """Undo Ultralytics' letterbox padding: keep the part of the map that covers the image, resized to it."""
    h, w = cam.shape
    scale = min(w / width, h / height)
    cw, ch = width * scale, height * scale
    x0, y0 = int(round((w - cw) / 2)), int(round((h - ch) / 2))
    cropped = cam[y0 : y0 + max(int(round(ch)), 1), x0 : x0 + max(int(round(cw)), 1)]
    return cv2.resize(cropped.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC).clip(0, 1)


SCALE_HINTS = {  # position of a Detect-head input -> what that scale is for
    0: "P3, stride 8: small objects (platelets)",
    1: "P4, stride 16: medium objects (RBCs)",
    2: "P5, stride 32: large objects (WBCs)",
}


def activation_comparison_figure(
    image: Image.Image, detections: Detections, maps: dict[int, np.ndarray], layer_names: dict[int, str]
) -> Figure:
    """One column per detection scale: activation map on top, overlay on the image below, shared colour scale."""
    rgb = image.convert("RGB")
    n = len(maps)
    fig = Figure(figsize=(5 * n + 1, 10), layout="constrained")
    axes, mappable = [], None
    for k, (layer_idx, cam) in enumerate(maps.items()):
        heat = letterbox_crop(cam, rgb.width, rgb.height)
        ax_map = fig.add_subplot(2, n, k + 1)
        mappable = ax_map.imshow(heat, cmap="jet", vmin=0, vmax=1)
        ax_map.set_title(f"Layer {layer_idx} ({layer_names.get(layer_idx, '?')}) — {cam.shape[1]}×{cam.shape[0]} map",
                         fontsize=10)
        ax_over = fig.add_subplot(2, n, n + k + 1)
        ax_over.imshow(np.asarray(rgb))
        ax_over.imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax_over.set_title(f"Overlay — {SCALE_HINTS.get(k, '')}".rstrip(" —"), fontsize=10)
        for ax in (ax_map, ax_over):
            ax.axis("off")
            axes.append(ax)
    cbar = fig.colorbar(mappable, ax=axes, location="right", fraction=0.02, pad=0.01)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0 (low)", "0.25", "0.5", "0.75", "1 (high)"])
    cbar.set_label("activation strength (1st PC, normalised)\nred = layer responds strongly, blue = weak / none",
                   fontsize=9)
    fig.suptitle(f"Detection scales compared ({len(detections)} boxes predicted): "
                 "small → large objects from left to right", fontsize=11)
    return fig


def activation_figure(image: Image.Image, detections: Detections, cam: np.ndarray, layer_idx: int, layer_name: str) -> Figure:
    rgb = image.convert("RGB")
    heat = letterbox_crop(cam, rgb.width, rgb.height)
    fig = Figure(figsize=(16, 5.4), layout="constrained")
    panels = (
        (np.asarray(draw_detections(rgb, detections)), f"Predictions ({len(detections)} boxes)"),
        (heat, f"Activation map — layer {layer_idx} ({layer_name}), {cam.shape[1]}×{cam.shape[0]}"),
        (None, "Overlay"),
    )
    heat_axes, mappable = [], None
    for k, (img, title) in enumerate(panels):
        ax = fig.add_subplot(1, 3, k + 1)
        if img is None:
            ax.imshow(np.asarray(rgb))
            ax.imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            heat_axes.append(ax)
        elif img.ndim == 2:
            mappable = ax.imshow(img, cmap="jet", vmin=0, vmax=1)
            heat_axes.append(ax)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    # Colour legend shared by the activation map and the overlay
    cbar = fig.colorbar(mappable, ax=heat_axes, location="right", fraction=0.025, pad=0.01)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0 (low)", "0.25", "0.5", "0.75", "1 (high)"])
    cbar.set_label("activation strength (1st PC, normalised)\nred = layer responds strongly, blue = weak / none",
                   fontsize=9)
    return fig


# ---------------------------------------------------------------- analysis: per-class histograms


@dataclass
class ClassHistograms:
    """Box areas and HSV hue / saturation histograms of the pixels inside the boxes, per class."""

    areas: dict[str, list[float]] = field(default_factory=dict)
    hue: dict[str, np.ndarray] = field(default_factory=dict)
    saturation: dict[str, np.ndarray] = field(default_factory=dict)
    n_images: int = 0

    def add(self, image: Image.Image, boxes: Detections) -> None:
        self.n_images += 1
        hsv = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2HSV)
        H, W = hsv.shape[:2]
        for (x1, y1, x2, y2), cls in zip(boxes.boxes, boxes.classes):
            name = boxes.names.get(int(cls), str(int(cls)))
            self.areas.setdefault(name, []).append(float(max(x2 - x1, 0) * max(y2 - y1, 0)))
            xa, ya = int(np.clip(x1, 0, W)), int(np.clip(y1, 0, H))
            xb, yb = int(np.clip(np.ceil(x2), 0, W)), int(np.clip(np.ceil(y2), 0, H))
            crop = hsv[ya:yb, xa:xb].reshape(-1, 3)
            if not len(crop):
                continue
            self.hue[name] = self.hue.get(name, np.zeros(180)) + np.bincount(crop[:, 0], minlength=180)[:180]
            self.saturation[name] = self.saturation.get(name, np.zeros(256)) + np.bincount(crop[:, 1], minlength=256)

    def summary(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, areas in self.areas.items():
            hue = self.hue.get(name)
            peak = f", peak hue {int(np.argmax(hue))}" if hue is not None and hue.sum() else ""
            out[name] = f"{len(areas)} boxes, median area {np.median(areas):,.0f} px²{peak}"
        return out


def histogram_figure(hists: ClassHistograms, class_order: list[str], subtitle: str) -> Figure:
    """Notebook E2: box-area distribution (log x), hue and saturation histograms per class."""
    fig = Figure(figsize=(16, 4.6), layout="constrained")
    ax_area, ax_hue, ax_sat = (fig.add_subplot(1, 3, k + 1) for k in range(3))
    all_areas = [a for v in hists.areas.values() for a in v if a > 0]
    lo, hi = (np.log10(min(all_areas)), np.log10(max(all_areas))) if all_areas else (1, 5)
    bins = np.logspace(lo - 0.05, hi + 0.05, 50)
    for k, name in enumerate(class_order):
        color = class_color(name, k)
        if hists.areas.get(name):
            ax_area.hist([a for a in hists.areas[name] if a > 0], bins=bins, alpha=0.5, color=color,
                         label=f"{name} (n={len(hists.areas[name])})", density=True)
        if name in hists.hue and hists.hue[name].sum():
            ax_hue.plot(hists.hue[name] / hists.hue[name].sum(), color=color, label=name)
        if name in hists.saturation and hists.saturation[name].sum():
            ax_sat.plot(hists.saturation[name] / hists.saturation[name].sum(), color=color, label=name)
    ax_area.set(xscale="log", xlabel="box area [px²]", ylabel="density", title="Box area per class")
    ax_hue.set(xlabel="hue (OpenCV, 0–179)", ylabel="fraction of pixels", title="Hue inside the boxes")
    ax_sat.set(xlabel="saturation (0–255)", ylabel="fraction of pixels", title="Saturation inside the boxes")
    for ax in (ax_area, ax_hue, ax_sat):
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle(subtitle, fontsize=11)
    return fig


# ---------------------------------------------------------------- dialog


@dataclass(frozen=True)
class YoloSettings:
    model_path: str
    conf: float = 0.25
    iou: float = 0.7
    source: str = "canvas"  # "canvas" | "file" | "folder"
    image_path: str = ""
    folder: str = ""
    max_images: int = 144
    layer_idx: int = 16
    box_source: str = "gt"  # histograms: "gt" | "pred"
    explain_bgr: bool = False
    compare_layers: bool = False  # activation map: all detection scales side by side


class YoloDialog(simpledialog.Dialog):
    """Shared dialog: model weights, thresholds, image source and the options of the chosen tool."""

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        *,
        has_canvas_image: bool,
        default_model: str,
        allow_folder: bool = False,
        default_source: str = "canvas",
        show_iou: bool = False,
        show_layer: bool = False,
        show_box_source: bool = False,
        show_bgr: bool = False,
    ) -> None:
        self.has_canvas_image = has_canvas_image
        self.default_model = default_model
        self.allow_folder = allow_folder
        self.default_source = default_source if (default_source != "canvas" or has_canvas_image) else "file"
        self.show_iou, self.show_layer = show_iou, show_layer
        self.show_box_source, self.show_bgr = show_box_source, show_bgr
        self.discovered = discover_yolo_models()
        super().__init__(parent, title=title)

    def body(self, master: tk.Frame) -> tk.Widget:
        self.model_path = tk.StringVar(value=self.default_model)
        model_frame = ttk.LabelFrame(master, text="YOLO weights (.pt) — stock names like yolo11n.pt are downloaded")
        model_frame.grid(row=0, column=0, sticky="we", pady=(0, 6))
        if self.discovered:
            preset = ttk.Combobox(model_frame, values=[str(p.relative_to(PROJECT_ROOT)) for p in self.discovered],
                                  state="readonly", width=56)
            preset.grid(row=0, column=0, columnspan=2, padx=6, pady=(4, 2), sticky="we")
            preset.bind("<<ComboboxSelected>>",
                        lambda _e: self.model_path.set(str(PROJECT_ROOT / preset.get())))
        entry = ttk.Entry(model_frame, textvariable=self.model_path, width=58)
        entry.grid(row=1, column=0, padx=6, pady=4)
        ttk.Button(model_frame, text="Browse…", command=self._browse_model).grid(row=1, column=1, padx=(0, 6))

        params = ttk.LabelFrame(master, text="Parameters")
        params.grid(row=1, column=0, sticky="we", pady=(0, 6))
        self.conf = tk.StringVar(value="0.25")
        spin_row(params, 0, "Confidence threshold", self.conf, CONF_RANGE, 0.05)
        row = 1
        if self.show_iou:
            self.iou = tk.StringVar(value="0.7")
            spin_row(params, row, "NMS IoU threshold", self.iou, IOU_RANGE, 0.05)
            row += 1
        if self.show_layer:
            self.layer = tk.StringVar(value="16")
            spin_row(params, row, "Layer index (16 = P3/8, 19 = P4/16, 22 = P5/32)", self.layer, (0, 30), 1)
            row += 1
            self.compare_layers = tk.BooleanVar(value=False)
            ttk.Checkbutton(params, text="Compare the 3 detection scales (layers 16 / 19 / 22) side by side",
                            variable=self.compare_layers).grid(row=row, column=0, columnspan=2, sticky="w",
                                                               padx=6, pady=2)
            row += 1
        if self.allow_folder:
            self.max_images = tk.StringVar(value="144")
            spin_row(params, row, "Max images from folder (random sample)", self.max_images, SAMPLE_RANGE, 10)
            row += 1
        if self.show_box_source:
            self.box_source = tk.StringVar(value="gt")
            box_frame = ttk.Frame(params)
            box_frame.grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=2)
            ttk.Label(box_frame, text="Boxes from:").pack(side="left")
            ttk.Radiobutton(box_frame, text="ground-truth labels", value="gt", variable=self.box_source).pack(side="left")
            ttk.Radiobutton(box_frame, text="model predictions", value="pred", variable=self.box_source).pack(side="left")
            row += 1
        if self.show_bgr:
            self.explain_bgr = tk.BooleanVar(value=False)
            ttk.Checkbutton(params, text='Also show "Why are people blue?" (BGR vs RGB) figure',
                            variable=self.explain_bgr).grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=2)

        src = ttk.LabelFrame(master, text="Input")
        src.grid(row=2, column=0, sticky="we")
        self.source = tk.StringVar(value=self.default_source)
        options = [("canvas", "Current canvas image"), ("file", "Image file")]
        if self.allow_folder:
            options.append(("folder", "Folder of images (YOLO layout: images/ + labels/)"))
        for k, (key, label) in enumerate(options):
            button = ttk.Radiobutton(src, text=label, value=key, variable=self.source)
            button.grid(row=k, column=0, columnspan=2, sticky="w", padx=6, pady=1)
            if key == "canvas" and not self.has_canvas_image:
                button.state(["disabled"])
        self.image_path = tk.StringVar(value="")
        self.folder = tk.StringVar(value=str(DEFAULT_BLOOD_TEST_DIR) if DEFAULT_BLOOD_TEST_DIR.is_dir() else "")
        ttk.Label(src, text="Image file:").grid(row=4, column=0, sticky="w", padx=6)
        ttk.Entry(src, textvariable=self.image_path, width=46).grid(row=5, column=0, padx=6, pady=2)
        ttk.Button(src, text="Browse…", command=self._browse_image).grid(row=5, column=1, padx=(0, 6))
        if self.allow_folder:
            ttk.Label(src, text="Folder:").grid(row=6, column=0, sticky="w", padx=6)
            ttk.Entry(src, textvariable=self.folder, width=46).grid(row=7, column=0, padx=6, pady=(2, 6))
            ttk.Button(src, text="Browse…", command=self._browse_folder).grid(row=7, column=1, padx=(0, 6))
        return entry

    def _browse_model(self) -> None:
        initial = DEFAULT_SET9_DIR if DEFAULT_SET9_DIR.is_dir() else PROJECT_ROOT
        chosen = filedialog.askopenfilename(title="Select YOLO weights", initialdir=str(initial),
                                            filetypes=MODEL_FILETYPES, parent=self)
        if chosen:
            self.model_path.set(chosen)

    def _browse_image(self) -> None:
        initial = DEFAULT_BLOOD_TEST_DIR if DEFAULT_BLOOD_TEST_DIR.is_dir() else PROJECT_ROOT
        chosen = filedialog.askopenfilename(title="Select image", initialdir=str(initial),
                                            filetypes=IMAGE_FILETYPES, parent=self)
        if chosen:
            self.image_path.set(chosen)
            self.source.set("file")

    def _browse_folder(self) -> None:
        chosen = filedialog.askdirectory(title="Select image folder", initialdir=self.folder.get() or str(PROJECT_ROOT),
                                         parent=self)
        if chosen:
            self.folder.set(chosen)
            self.source.set("folder")

    def validate(self) -> bool:
        try:
            model = self.model_path.get().strip()
            if not model:
                raise ValueError("Choose YOLO weights (.pt).")
            if not Path(model).is_file() and not Path(model).name.lower().startswith("yolo"):
                raise ValueError(f"Weights file not found: {model}")
            source = self.source.get()
            image_path, folder = self.image_path.get().strip(), self.folder.get().strip()
            if source == "canvas" and not self.has_canvas_image:
                raise ValueError("No image is open — choose an image file instead.")
            if source == "file" and not Path(image_path).is_file():
                raise ValueError("Choose an existing image file.")
            if source == "folder":
                if not Path(folder).is_dir() or not list_images(Path(folder)):
                    raise ValueError("Choose a folder that contains images.")
            self._settings = YoloSettings(
                model_path=model,
                conf=read_number(self.conf, "Confidence threshold", CONF_RANGE),
                iou=read_number(self.iou, "NMS IoU threshold", IOU_RANGE) if self.show_iou else 0.7,
                source=source,
                image_path=image_path,
                folder=folder,
                max_images=read_number(self.max_images, "Max images", SAMPLE_RANGE, int) if self.allow_folder else 1,
                layer_idx=read_number(self.layer, "Layer index", (0, 30), int) if self.show_layer else 16,
                box_source=self.box_source.get() if self.show_box_source else "gt",
                explain_bgr=bool(self.explain_bgr.get()) if self.show_bgr else False,
                compare_layers=bool(self.compare_layers.get()) if self.show_layer else False,
            )
        except ValueError as error:
            messagebox.showerror(self.title(), str(error), parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self._settings


# ---------------------------------------------------------------- tool helpers


def _single_image(settings: YoloSettings, document: ImageDocument) -> tuple[Image.Image, Path | None, str]:
    """The image to analyse, its path (to look up GT labels) and a label for the results panel."""
    if settings.source == "file":
        path = Path(settings.image_path)
        return open_rgb(path), path, path.name
    if document.current is None:
        raise ValueError("No image is open on the canvas.")
    path = Path(document.path) if document.path else None
    return document.current.convert("RGB"), path, path.name if path else "Current canvas image"


def _sample_folder(settings: YoloSettings) -> list[Path]:
    images = list_images(Path(settings.folder))
    if len(images) > settings.max_images:
        images = sorted(random.Random(0).sample(images, settings.max_images))
    return images


def _count_details(table: dict[str, tuple[int, int | None]]) -> dict[str, str]:
    return {f"  {name}": (f"{p} predicted" + ("" if g is None else f" / {g} GT")) for name, (p, g) in table.items()}


def _detection_details(detections: Detections, limit: int = 40) -> dict[str, str]:
    order = np.argsort(-detections.scores) if detections.scores is not None else np.arange(len(detections))
    details: dict[str, str] = {}
    for rank, i in enumerate(order[:limit]):
        x1, y1, x2, y2 = detections.boxes[i]
        name = detections.names.get(int(detections.classes[i]), "?")
        score = "" if detections.scores is None else f" {detections.scores[i]:.2f}"
        details[f"  #{rank + 1} {name}{score}"] = f"({x1:.0f}, {y1:.0f}) – ({x2:.0f}, {y2:.0f})"
    if len(order) > limit:
        details["  …"] = f"{len(order) - limit} more boxes"
    return details


def _show_popup(parent: Any, title: str, figure: Figure) -> None:
    if isinstance(parent, tk.Misc):
        from .hog import show_figure

        popup = show_figure(parent, title, figure)
        parent.winfo_toplevel()._set9_popup = popup  # keep a reference while it is open


# ---------------------------------------------------------------- tools


class YoloPretrainedTool(ForensicsTool):
    tool_id = "yolo_pretrained"
    title = "YOLO detection (pretrained COCO)"
    category = CATEGORY
    description = (
        "Set 9.1: run a pretrained Ultralytics YOLO11 model (80 COCO classes) on an image and draw the boxes. "
        "Optionally explains why people look blue when the BGR output is shown as RGB."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        settings = YoloDialog(
            parent, "YOLO detection — pretrained (Set 9.1)", has_canvas_image=document.current is not None,
            default_model=str(DEFAULT_PRETRAINED_WEIGHTS), show_iou=True, show_bgr=True,
        ).result
        if settings is None:
            return None
        image, _path, source = _single_image(settings, document)
        with ProgressPopup(parent, "YOLO detection") as progress:
            model = progress.run("Loading YOLO model (downloading on first use)…",
                                 lambda: load_yolo_model(settings.model_path))
            detections, result = progress.run("Detecting objects…",
                                              lambda: predict(model, image, conf=settings.conf, iou=settings.iou))
        if settings.explain_bgr:
            _show_popup(parent, "Why are people blue?", bgr_explanation_figure(result))
        counts = detections.counts()
        details: dict[str, Any] = {
            "Operation": "YOLO detection (Set 9.1)",
            "Model": Path(settings.model_path).name,
            "Image": source,
            "Classes in model": len(detections.names),
            "Confidence / IoU": f"{settings.conf:.2f} / {settings.iou:.2f}",
            "Detections": len(detections),
        }
        details |= {f"  {name}": n for name, n in counts.most_common()}
        details |= _detection_details(detections)
        summary = ", ".join(f"{n} {name}" for name, n in counts.most_common()) or "nothing"
        return ToolResult(
            message=f"YOLO ({Path(settings.model_path).name}) found {summary}.",
            image=draw_detections(image, detections),
            details=details,
        )


class YoloBloodCellTool(ForensicsTool):
    tool_id = "yolo_blood_cells"
    title = "Our model: blood cells (GT vs pred)"
    category = CATEGORY
    description = (
        "Set 9.3: run our YOLO model trained on the blood-cell dataset (PC, RBC, WBC). Ground-truth boxes "
        "(solid, from the YOLO labels/ folder) are drawn together with the predictions (dashed, with confidence)."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        settings = YoloDialog(
            parent, "Our blood-cell YOLO model (Set 9.3)", has_canvas_image=document.current is not None,
            default_model=default_custom_model(), show_iou=True,
        ).result
        if settings is None:
            return None
        image, path, source = _single_image(settings, document)
        with ProgressPopup(parent, "Blood-cell detection") as progress:
            model = progress.run("Loading our YOLO model…", lambda: load_yolo_model(settings.model_path))
            detections, _ = progress.run("Detecting cells…",
                                         lambda: predict(model, image, conf=settings.conf, iou=settings.iou))
        gt = ground_truth_for(path, image, model_names(model))
        table = count_table(model_names(model), detections, gt)
        details: dict[str, Any] = {
            "Operation": "Our model: GT vs prediction (Set 9.3)",
            "Model": Path(settings.model_path).name,
            "Image": source,
            "Ground truth": f"{len(gt)} boxes (solid)" if gt is not None else "no label file found",
            "Predictions": f"{len(detections)} boxes (dashed), conf ≥ {settings.conf:.2f}",
        }
        details |= _count_details(table)
        details |= _detection_details(detections)
        counts = ", ".join(f"{name} {p}" + ("" if g is None else f"/{g}") for name, (p, g) in table.items())
        return ToolResult(
            message=f"Blood cells (pred{'/GT' if gt is not None else ''}): {counts}.",
            image=gt_vs_pred_image(image, gt, detections),
            details=details,
        )


class CellCountTool(ForensicsTool):
    tool_id = "yolo_cell_count"
    title = "Cell count"
    category = CATEGORY
    description = (
        "Count PC / RBC / WBC cells with our YOLO model. On one image: bar chart of predicted vs GT counts. "
        "On a folder (e.g. the test split): predicted vs GT count per image with MAE and exact-match rate."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        settings = YoloDialog(
            parent, "Cell count (Set 9 extra)", has_canvas_image=document.current is not None,
            default_model=default_custom_model(), allow_folder=True,
            default_source="canvas" if document.current is not None else "folder",
        ).result
        if settings is None:
            return None
        if settings.source == "folder":
            return self._run_folder(parent, settings)

        image, path, source = _single_image(settings, document)
        with ProgressPopup(parent, "Cell count") as progress:
            model = progress.run("Loading our YOLO model…", lambda: load_yolo_model(settings.model_path))
            detections, _ = progress.run("Counting cells…", lambda: predict(model, image, conf=settings.conf))
        names = model_names(model)
        gt = ground_truth_for(path, image, names)
        table = count_table(names, detections, gt)
        figure = count_figure(gt_vs_pred_image(image, gt, detections), table,
                              f"{source}: {len(detections)} cells detected")
        details: dict[str, Any] = {
            "Operation": "Cell count (Set 9)",
            "Model": Path(settings.model_path).name,
            "Image": source,
            "Confidence": f"{settings.conf:.2f}",
            "Total cells": f"{len(detections)} predicted" + ("" if gt is None else f" / {len(gt)} GT"),
        }
        details |= _count_details(table)
        return ToolResult(
            message="Cell count: " + ", ".join(f"{n} {p}" for n, (p, _) in table.items()),
            image=figure_to_image(figure),
            details=details,
        )

    def _run_folder(self, parent: tk.Misc, settings: YoloSettings) -> ToolResult:
        images = _sample_folder(settings)
        with ProgressPopup(parent, "Cell count") as progress:
            model = progress.run("Loading our YOLO model…", lambda: load_yolo_model(settings.model_path))
            names = model_names(model)
            gt_counts: dict[str, list[int]] = {n: [] for n in names.values()}
            pred_counts: dict[str, list[int]] = {n: [] for n in names.values()}
            n_with_gt = 0
            for k, path in enumerate(images):
                image = open_rgb(path)
                pred, _ = progress.run_step(k, len(images), f"Counting cells in {path.name[:40]}…",
                                            lambda: predict(model, image, conf=settings.conf))
                gt = ground_truth_for(path, image, names)
                if gt is None:
                    continue
                n_with_gt += 1
                pc, gc = pred.counts(), gt.counts()
                for n in names.values():
                    pred_counts[n].append(pc.get(n, 0))
                    gt_counts[n].append(gc.get(n, 0))
        if not n_with_gt:
            raise ValueError("No YOLO label files found for these images (expected <split>/labels/<name>.txt).")
        details: dict[str, Any] = {
            "Operation": "Cell count on a folder (Set 9)",
            "Model": Path(settings.model_path).name,
            "Folder": settings.folder,
            "Images": f"{n_with_gt} with labels (of {len(images)} sampled)",
            "Confidence": f"{settings.conf:.2f}",
        }
        for n in names.values():
            g, p = np.array(gt_counts[n]), np.array(pred_counts[n])
            details[f"  {n}"] = (f"GT {g.sum()} / pred {p.sum()}, MAE {np.abs(g - p).mean():.2f}, "
                                 f"exact {np.mean(g == p):.0%}")
        return ToolResult(
            message=f"Cell count on {n_with_gt} images: "
                    + ", ".join(f"{n} MAE {np.abs(np.array(gt_counts[n]) - np.array(pred_counts[n])).mean():.2f}"
                                for n in names.values()),
            image=figure_to_image(count_scatter_figure(gt_counts, pred_counts, settings.conf)),
            details=details,
        )


class YoloActivationMapTool(ForensicsTool):
    tool_id = "yolo_activation_map"
    title = "Activation map"
    category = CATEGORY
    description = (
        "Where does the detector look? Hooks an inner YOLO layer and shows the first principal component of its "
        "feature map (gradient-free activation map) overlaid on the image. Layer 16 = stride-8 P3 map (small cells)."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        settings = YoloDialog(
            parent, "YOLO activation map (Set 9 extra)", has_canvas_image=document.current is not None,
            default_model=default_custom_model() or str(DEFAULT_PRETRAINED_WEIGHTS), show_layer=True,
        ).result
        if settings is None:
            return None
        image, _path, source = _single_image(settings, document)
        if settings.compare_layers:
            return self._run_comparison(parent, settings, image, source)
        with ProgressPopup(parent, "Activation map") as progress:
            model = progress.run("Loading YOLO model…", lambda: load_yolo_model(settings.model_path))
            detections, cam = progress.run(
                f"Computing the activation map of layer {settings.layer_idx}…",
                lambda: activation_map(model, image, settings.layer_idx, conf=settings.conf),
            )
        layer = model.predictor.model.model.model[settings.layer_idx]
        layer_name = type(layer).__name__
        details: dict[str, Any] = {
            "Operation": "Activation map (Set 9)",
            "Model": Path(settings.model_path).name,
            "Image": source,
            "Layer": f"{settings.layer_idx} ({layer_name})",
            "Feature map": f"{cam.shape[1]} × {cam.shape[0]}",
            "Method": "1st principal component of the C × h × w activations (SVD), ReLU, normalised",
            "Detections": len(detections),
        }
        details |= {f"  {name}": n for name, n in detections.counts().most_common()}
        return ToolResult(
            message=f"Activation map of layer {settings.layer_idx} ({layer_name}) for {source}.",
            image=figure_to_image(activation_figure(image, detections, cam, settings.layer_idx, layer_name)),
            details=details,
        )

    def _run_comparison(self, parent: tk.Misc, settings: YoloSettings, image: Image.Image, source: str) -> ToolResult:
        with ProgressPopup(parent, "Activation map") as progress:
            model = progress.run("Loading YOLO model…", lambda: load_yolo_model(settings.model_path))
            layer_idxs = detection_scale_layers(model, image)
            detections, maps = progress.run(
                f"Computing the activation maps of layers {', '.join(map(str, layer_idxs))}…",
                lambda: activation_maps(model, image, layer_idxs, conf=settings.conf),
            )
        layers = _network_layers(model, image)
        names = {i: type(layers[i]).__name__ for i in layer_idxs}
        details: dict[str, Any] = {
            "Operation": "Activation maps — detection scales compared (Set 9)",
            "Model": Path(settings.model_path).name,
            "Image": source,
            "Method": "1st principal component of the C × h × w activations (SVD), ReLU, normalised",
        }
        for k, i in enumerate(layer_idxs):
            details[f"  Layer {i} ({names[i]})"] = f"{maps[i].shape[1]} × {maps[i].shape[0]} map — {SCALE_HINTS.get(k, '')}"
        details["Detections"] = len(detections)
        details |= {f"  {name}": n for name, n in detections.counts().most_common()}
        return ToolResult(
            message=f"Activation maps of layers {', '.join(map(str, layer_idxs))} compared for {source}.",
            image=figure_to_image(activation_comparison_figure(image, detections, maps, names)),
            details=details,
        )


class PerClassHistogramTool(ForensicsTool):
    tool_id = "yolo_class_histograms"
    title = "Per-class histograms"
    category = CATEGORY
    description = (
        "What separates the classes? Box-area distribution (log scale) and HSV hue / saturation histograms of the "
        "pixels inside the boxes, per class — from ground-truth labels or model predictions, on one image or a folder."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        settings = YoloDialog(
            parent, "Per-class histograms (Set 9 extra)", has_canvas_image=document.current is not None,
            default_model=default_custom_model(), allow_folder=True, show_box_source=True,
            default_source="folder",
        ).result
        if settings is None:
            return None
        use_gt = settings.box_source == "gt"
        if settings.source == "folder":
            items: list[tuple[Path | None, Image.Image | None]] = [(p, None) for p in _sample_folder(settings)]
            where = f"{len(items)} images from {Path(settings.folder).parent.name or settings.folder}"
        else:
            image, path, source = _single_image(settings, document)
            items, where = [(path, image)], source

        hists = ClassHistograms()
        skipped = 0
        with ProgressPopup(parent, "Per-class histograms") as progress:
            model = progress.run("Loading YOLO model…", lambda: load_yolo_model(settings.model_path))
            names = model_names(model)
            for k, (path, image) in enumerate(items):
                text = f"Collecting boxes from {path.name[:40] if path else 'canvas image'}…"
                image = image if image is not None else open_rgb(path)
                if use_gt:
                    progress.step(k, len(items), text)
                    boxes = ground_truth_for(path, image, names)
                    if boxes is None:
                        skipped += 1
                        continue
                else:
                    boxes, _ = progress.run_step(k, len(items), text,
                                                 lambda: predict(model, image, conf=settings.conf))
                hists.add(image, boxes)
        if not hists.areas:
            raise ValueError("No boxes found. For ground truth, images need YOLO label files "
                             "(<split>/labels/<name>.txt); otherwise choose 'model predictions'.")
        box_label = "ground-truth boxes" if use_gt else f"predicted boxes (conf ≥ {settings.conf:.2f})"
        figure = histogram_figure(hists, list(names.values()), f"Per-class histograms — {box_label}, {where}")
        details: dict[str, Any] = {
            "Operation": "Per-class histograms (Set 9)",
            "Boxes": box_label,
            "Images": f"{hists.n_images}" + (f" ({skipped} skipped: no labels)" if skipped else ""),
        }
        details |= {f"  {name}": text for name, text in hists.summary().items()}
        return ToolResult(
            message=f"Per-class histograms of {sum(map(len, hists.areas.values()))} {box_label}.",
            image=figure_to_image(figure),
            details=details,
        )


__all__ = [
    "CATEGORY",
    "CellCountTool",
    "ClassHistograms",
    "DEFAULT_BLOOD_TEST_DIR",
    "DEFAULT_PRETRAINED_WEIGHTS",
    "Detections",
    "PerClassHistogramTool",
    "YoloActivationMapTool",
    "YoloBloodCellTool",
    "YoloDialog",
    "YoloPretrainedTool",
    "YoloSettings",
    "activation_comparison_figure",
    "activation_figure",
    "activation_maps",
    "detection_scale_layers",
    "activation_map",
    "count_figure",
    "count_scatter_figure",
    "count_table",
    "default_custom_model",
    "discover_yolo_models",
    "draw_detections",
    "ground_truth_for",
    "gt_vs_pred_image",
    "histogram_figure",
    "label_path_for",
    "letterbox_crop",
    "load_yolo_model",
    "predict",
    "read_yolo_labels",
]
