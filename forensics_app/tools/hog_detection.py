"""Set 4.4: Person detection by sliding a window and comparing its HOG to a template's HOG."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from matplotlib import colormaps
import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import hog

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .hog import BLOCK_RANGE, CELL_RANGE, ORIENTATIONS_RANGE, to_rgb_array
from .modes import split_alpha

TEMPLATE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]
STEP_RANGE = (1, 128)
STEP_CELLS_RANGE = (1, 16)  # fast variant: step counted in cells
WINDOW_RANGE = (4, 4096)
SCALE_RANGE = (0.1, 10.0)

# Search variants (key -> (label, explanation))
VARIANTS = {
    "exact": (
        "Exact: HOG of every window (as in the slides)",
        "Crops each window and runs hog() on it. Any step in pixels; slow on big images.",
    ),
    "fast": (
        "Fast: HOG of the whole image once, sliced per window",
        "100× faster. The step is a whole number of cells, and edge cells differ slightly\n"
        "(the window border keeps its real gradients).",
    ),
}

# Similarity measures between the window's and the template's HOG vectors (key -> label)
METRICS = {
    "euclidean": "Euclidean distance (smallest wins)",
    "dot": "Dot product (largest wins)",
    "cosine": "Cosine similarity (largest wins)",
}
METRIC_NAMES = {"euclidean": "Euclidean", "dot": "Dot product", "cosine": "Cosine"}
LOWER_IS_BETTER = {"euclidean"}
BOX_COLOURS = {"euclidean": (255, 0, 0), "dot": (0, 120, 255), "cosine": (0, 200, 0)}
BOX_WIDTH = 2

# How the result is shown (key -> label)
DISPLAYS = {
    "boxes": "Best-match boxes on the image",
    "heatmap": "Score heatmap (with the boxes on top)",
}
HEATMAP_COLORMAP = "inferno"
HEATMAP_OPACITY = 0.65  # weight of the heatmap colour over the image
DIM_FACTOR = 0.5  # brightness kept by pixels the heatmap does not cover


@dataclass(frozen=True)
class DetectionSettings:
    orientations: int = 16
    pixels_per_cell: int = 8
    cells_per_block: int = 2
    step: int = 8
    metrics: tuple[str, ...] = ("euclidean", "dot", "cosine")
    # Sliding window size; ``None`` keeps the template's own width/height
    window_width: int | None = None
    window_height: int | None = None
    display: str = "boxes"
    heatmap_metric: str = "cosine"
    # Factors applied to the window size; the search runs once per scale
    scales: tuple[float, ...] = (1.0,)
    variant: str = "exact"  # a key of VARIANTS


@dataclass(frozen=True)
class Detection:
    """Best window found by one similarity measure: top-left corner, size, score and scale."""

    method: str
    row: int
    col: int
    height: int
    width: int
    score: float
    scale: float = 1.0

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.col, self.row, self.col + self.width - 1, self.row + self.height - 1


@dataclass(frozen=True)
class ScoreMap:
    """Every metric's score for every window; ``scores[m][i, j]`` is the window at ``(rows[i], cols[j])``."""

    rows: np.ndarray
    cols: np.ndarray
    height: int
    width: int
    scores: dict[str, np.ndarray]
    features: int = 1  # length of the HOG vectors compared
    scale: float = 1.0

    @property
    def windows(self) -> int:
        return self.rows.size * self.cols.size


def hog_features(rgb: np.ndarray, settings: DetectionSettings, feature_vector: bool = True) -> np.ndarray:
    """HOG of ``rgb``; with ``feature_vector=False`` the blocks stay on the cell grid
    (shape: block rows × block cols × cells per block × cells per block × orientations)."""
    cell, block = settings.pixels_per_cell, settings.cells_per_block
    return hog(
        rgb,
        orientations=settings.orientations,
        pixels_per_cell=(cell, cell),
        cells_per_block=(block, block),
        channel_axis=2,
        feature_vector=feature_vector,
    )


def resize_template(template: Image.Image, settings: DetectionSettings) -> Image.Image:
    """Resize the template to the sliding window size, so the search can run at different scales."""
    size = (settings.window_width or template.width, settings.window_height or template.height)
    if size == template.size:
        return template
    base, _alpha = split_alpha(template)
    return base.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """``a·b / (|a| |b|)``: 1 = same gradient pattern, whatever the strength; 0 for an empty vector."""
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norms) if norms > 0 else 0.0


def score_windows(
    image: Image.Image, template: Image.Image, settings: DetectionSettings, image_hog: np.ndarray | None = None
) -> ScoreMap:
    """Slide a window over ``image`` and score each window's HOG against the template's with every metric.

    The template is first resized to the window size from ``settings``. In the ``"fast"`` variant the
    window features are sliced out of the whole image's HOG (``image_hog``, computed here if not given).
    """
    img = to_rgb_array(image)
    tmpl = to_rgb_array(resize_template(template, settings))
    win_h, win_w = tmpl.shape[:2]
    min_side = settings.pixels_per_cell * settings.cells_per_block
    if min(win_h, win_w) < min_side:
        raise ValueError(f"The window must be at least {min_side} px on each side (pixels per cell × cells per block).")
    if win_h > img.shape[0] or win_w > img.shape[1]:
        raise ValueError(
            f"The window ({win_w} × {win_h}) is larger than the image ({img.shape[1]} × {img.shape[0]})."
        )

    cell, block = settings.pixels_per_cell, settings.cells_per_block
    if settings.variant == "fast":
        # Windows must start on the full image's cell grid to be sliced out of it
        if settings.step % cell:
            raise ValueError(f"The fast variant needs a step that is a multiple of the cell size ({cell} px).")
        blocks = image_hog if image_hog is not None else hog_features(img, settings, feature_vector=False)
        blocks_h, blocks_w = win_h // cell - block + 1, win_w // cell - block + 1

        def window_features(i: int, j: int) -> np.ndarray:
            return blocks[i // cell : i // cell + blocks_h, j // cell : j // cell + blocks_w].ravel()

    else:

        def window_features(i: int, j: int) -> np.ndarray:
            return hog_features(img[i : i + win_h, j : j + win_w], settings)

    tmpl_fd = hog_features(tmpl, settings)
    rows = np.arange(0, img.shape[0] - win_h + 1, settings.step)
    cols = np.arange(0, img.shape[1] - win_w + 1, settings.step)
    scores = {metric: np.empty((rows.size, cols.size)) for metric in METRICS}
    for a, i in enumerate(rows):
        for b, j in enumerate(cols):
            window_fd = window_features(i, j)
            scores["euclidean"][a, b] = np.linalg.norm(tmpl_fd - window_fd)
            scores["dot"][a, b] = np.dot(window_fd, tmpl_fd)
            scores["cosine"][a, b] = cosine_similarity(window_fd, tmpl_fd)
    return ScoreMap(rows, cols, win_h, win_w, scores, features=int(tmpl_fd.size))


def best_matches(score_map: ScoreMap, metrics: tuple[str, ...]) -> list[Detection]:
    """The best-scoring window for each metric (lowest distance, highest similarity)."""
    detections = []
    for metric in metrics:
        grid = score_map.scores[metric]
        flat = np.argmin(grid) if metric in LOWER_IS_BETTER else np.argmax(grid)
        a, b = np.unravel_index(flat, grid.shape)
        detections.append(
            Detection(
                metric,
                int(score_map.rows[a]),
                int(score_map.cols[b]),
                score_map.height,
                score_map.width,
                float(grid[a, b]),
                score_map.scale,
            )
        )
    return detections


def comparable_score(metric: str, score: float, features: int) -> float:
    """Make scores from different window sizes comparable (higher = better).

    Longer HOG vectors (bigger windows) have larger distances and dot products just because
    they sum more terms, so those two are divided by the vector length. Cosine is already
    length-independent.
    """
    if metric == "euclidean":
        return -score / np.sqrt(features)
    if metric == "dot":
        return score / features
    return score


def scaled_settings(settings: DetectionSettings, template: Image.Image, scale: float) -> DetectionSettings:
    """``settings`` with the window size multiplied by ``scale``."""
    width = max(1, round((settings.window_width or template.width) * scale))
    height = max(1, round((settings.window_height or template.height) * scale))
    return replace(settings, window_width=width, window_height=height)


def detect(
    image: Image.Image, template: Image.Image, settings: DetectionSettings
) -> tuple[list[Detection], list[ScoreMap]]:
    """Search at every scale and return the best match per selected metric, plus each scale's scores.

    Scales whose window does not fit in the image (or is too small for one HOG block) are skipped.
    """
    # Fast variant: the image's HOG does not depend on the window size, so one HOG serves every scale
    image_hog = None
    if settings.variant == "fast":
        image_hog = hog_features(to_rgb_array(image), settings, feature_vector=False)
    score_maps: list[ScoreMap] = []
    errors: list[str] = []
    for scale in settings.scales:
        try:
            score_map = score_windows(image, template, scaled_settings(settings, template, scale), image_hog)
        except ValueError as error:
            errors.append(f"× {scale:g}: {error}")
            continue
        score_maps.append(replace(score_map, scale=scale))
    if not score_maps:
        raise ValueError("No scale could be searched.\n" + "\n".join(errors))
    return best_across_scales(score_maps, settings.metrics), score_maps


def best_across_scales(score_maps: list[ScoreMap], metrics: tuple[str, ...]) -> list[Detection]:
    detections = []
    for metric in metrics:
        candidates = [(best_matches(m, (metric,))[0], m.features) for m in score_maps]
        best, _features = max(candidates, key=lambda c: comparable_score(metric, c[0].score, c[1]))
        detections.append(best)
    return detections


def parse_scales(text: str) -> tuple[float, ...]:
    """Parse ``"0.75, 1, 1.25"`` into sorted unique scale factors."""
    try:
        scales = sorted({float(part) for part in text.replace(";", ",").split(",") if part.strip()})
    except ValueError:
        raise ValueError("Scales must be numbers separated by commas, e.g. 0.75, 1, 1.25.") from None
    if not scales:
        raise ValueError("Enter at least one scale.")
    if not all(SCALE_RANGE[0] <= s <= SCALE_RANGE[1] for s in scales):
        raise ValueError(f"Each scale must be between {SCALE_RANGE[0]:g} and {SCALE_RANGE[1]:g}.")
    return tuple(scales)


def match_strength(score_map: ScoreMap, metric: str) -> np.ndarray:
    """The metric's scores rescaled to [0, 1] with 1 = most similar window (distances are flipped)."""
    grid = score_map.scores[metric]
    span = grid.max() - grid.min()
    strength = (grid - grid.min()) / span if span > 0 else np.ones_like(grid)
    return 1.0 - strength if metric in LOWER_IS_BETTER else strength


def render_heatmap(image: Image.Image, score_map: ScoreMap, metric: str, step: int) -> Image.Image:
    """Colour each window's centre (a ``step`` × ``step`` patch) by how well it matches the template."""
    base, _alpha = split_alpha(image)
    rgb = np.asarray(base.convert("RGB"), dtype=np.float64) / 255.0
    out = rgb * DIM_FACTOR

    colours = colormaps[HEATMAP_COLORMAP](match_strength(score_map, metric))[..., :3]
    patches = colours.repeat(step, axis=0).repeat(step, axis=1)
    top, left = score_map.height // 2 - step // 2, score_map.width // 2 - step // 2
    top, left = max(top, 0), max(left, 0)
    h = min(patches.shape[0], rgb.shape[0] - top)
    w = min(patches.shape[1], rgb.shape[1] - left)
    region = (slice(top, top + h), slice(left, left + w))
    out[region] = (1.0 - HEATMAP_OPACITY) * rgb[region] + HEATMAP_OPACITY * patches[:h, :w]
    return Image.fromarray(np.clip(np.round(out * 255), 0, 255).astype(np.uint8), mode="RGB")


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    base, _alpha = split_alpha(image)
    output = base.convert("RGB")
    draw = ImageDraw.Draw(output)
    for detection in detections:
        draw.rectangle(detection.box, outline=BOX_COLOURS[detection.method], width=BOX_WIDTH)
    return output


class VariantDialog(simpledialog.Dialog):
    """First step: choose how the windows' HOG is computed."""

    def body(self, master: tk.Frame) -> tk.Widget:
        self.variant = tk.StringVar(value="exact")
        ttk.Label(master, text="How should each window's HOG be computed?").grid(row=0, column=0, sticky="w")
        first = None
        for row, (key, (label, explanation)) in enumerate(VARIANTS.items()):
            button = ttk.Radiobutton(master, text=label, value=key, variable=self.variant)
            button.grid(row=1 + 2 * row, column=0, sticky="w", padx=(12, 0), pady=(8, 0))
            ttk.Label(master, text=explanation, foreground="gray").grid(
                row=2 + 2 * row, column=0, sticky="w", padx=(32, 0)
            )
            first = first or button
        return first

    def apply(self) -> None:
        self.result = self.variant.get()


class DetectionDialog(simpledialog.Dialog):
    """Ask for the window size, HOG parameters, sliding step, metrics and display of one variant."""

    def __init__(self, parent: tk.Misc, title: str, template: Image.Image, variant: str = "exact") -> None:
        self.template = template
        self.variant = variant
        super().__init__(parent, title=title)

    def body(self, master: tk.Frame) -> tk.Widget:
        self.orientations = tk.StringVar(value="16")
        self.cell = tk.StringVar(value="8")
        self.block = tk.StringVar(value="2")
        self.fast = self.variant == "fast"
        self.step = tk.StringVar(value="1" if self.fast else "8")
        self.window_width = tk.StringVar(value=str(self.template.width))
        self.window_height = tk.StringVar(value=str(self.template.height))
        self.keep_aspect = tk.BooleanVar(value=True)
        self.scales = tk.StringVar(value="1")
        self.metrics ={key: tk.BooleanVar(value=True) for key in METRICS}
        self.display = tk.StringVar(value="boxes")
        self.heatmap_metric = tk.StringVar(value=METRIC_NAMES["cosine"])

        ttk.Label(
            master,
            text=f"Variant: {VARIANTS[self.variant][0]}\n"
            f"Template size: {self.template.width} × {self.template.height} px",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        window = ttk.Frame(master)
        window.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(window, text="Sliding window (W × H):").pack(side="left")
        ttk.Spinbox(
            window, from_=WINDOW_RANGE[0], to=WINDOW_RANGE[1], increment=8, textvariable=self.window_width, width=6
        ).pack(side="left", padx=(6, 2))
        ttk.Label(window, text="×").pack(side="left")
        self._height_box = ttk.Spinbox(
            window, from_=WINDOW_RANGE[0], to=WINDOW_RANGE[1], increment=8, textvariable=self.window_height, width=6
        )
        self._height_box.pack(side="left", padx=(2, 0))
        ttk.Checkbutton(
            master, text="Keep template aspect ratio", variable=self.keep_aspect, command=self._sync_height
        ).grid(row=2, column=0, columnspan=2, sticky="w")
        scales = ttk.Frame(master)
        scales.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 8))
        scales_row = ttk.Frame(scales)
        scales_row.pack(anchor="w")
        ttk.Label(scales_row, text="Scales (× window, comma-separated):").pack(side="left")
        ttk.Entry(scales_row, textvariable=self.scales, width=18).pack(side="left", padx=(6, 0))
        ttk.Label(
            scales, text="The template is resized to each scaled window, e.g. 0.75, 1, 1.25, 1.5.\n"
            "Each scale is a full search, so more scales = slower."
        ).pack(anchor="w")
        self.window_width.trace_add("write", lambda *_: self._sync_height())
        self._sync_height()

        rows = (
            ("Orientations (histogram bins):", self.orientations, ORIENTATIONS_RANGE),
            ("Pixels per cell (N × N):", self.cell, CELL_RANGE),
            ("Cells per block (N × N):", self.block, BLOCK_RANGE),
            ("Window step (cells):", self.step, STEP_CELLS_RANGE) if self.fast else ("Window step (px):", self.step, STEP_RANGE),
        )
        first = None
        for row, (text, var, (low, high)) in enumerate(rows, start=4):
            ttk.Label(master, text=text).grid(row=row, column=0, sticky="w")
            box = ttk.Spinbox(master, from_=low, to=high, increment=1, textvariable=var, width=6)
            box.grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
            first = first or box
        step_hint = (
            "1 cell = pixels per cell; windows can only start on the cell grid."
            if self.fast
            else "Smaller step = more precise but slower."
        )
        ttk.Label(master, text=step_hint).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        ttk.Label(master, text="Similarity metrics (one box each):").grid(row=9, column=0, columnspan=2, sticky="w")
        for row, (key, text) in enumerate(METRICS.items(), start=10):
            ttk.Checkbutton(master, text=text, variable=self.metrics[key]).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=(12, 0)
            )
        ttk.Label(master, text="Boxes: red = Euclidean, blue = dot product, green = cosine.").grid(
            row=13, column=0, columnspan=2, sticky="w", pady=(2, 8)
        )

        ttk.Label(master, text="Show:").grid(row=14, column=0, columnspan=2, sticky="w")
        for row, (key, text) in enumerate(DISPLAYS.items(), start=15):
            ttk.Radiobutton(master, text=text, value=key, variable=self.display, command=self._sync_heatmap).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=(12, 0)
            )
        heatmap = ttk.Frame(master)
        heatmap.grid(row=17, column=0, columnspan=2, sticky="w", padx=(32, 0))
        ttk.Label(heatmap, text="Heatmap metric:").pack(side="left")
        self._heatmap_box = ttk.Combobox(
            heatmap, textvariable=self.heatmap_metric, values=list(METRIC_NAMES.values()), state="readonly", width=12
        )
        self._heatmap_box.pack(side="left", padx=(6, 0))
        ttk.Label(master, text="Bright = window looks like the template (scaled from worst to best).").grid(
            row=18, column=0, columnspan=2, sticky="w", padx=(32, 0)
        )
        self._sync_heatmap()
        return first

    def _sync_height(self) -> None:
        """With the aspect ratio locked, the height follows the width and cannot be edited."""
        if not self.keep_aspect.get():
            self._height_box.state(["!disabled"])
            return
        self._height_box.state(["disabled"])
        try:
            width = int(self.window_width.get())
        except ValueError:
            return
        self.window_height.set(str(max(1, round(width * self.template.height / self.template.width))))

    def _sync_heatmap(self) -> None:
        self._heatmap_box.configure(state="readonly" if self.display.get() == "heatmap" else "disabled")

    def validate(self) -> bool:
        values = []
        for name, var, (low, high) in (
            ("Orientations", self.orientations, ORIENTATIONS_RANGE),
            ("Pixels per cell", self.cell, CELL_RANGE),
            ("Cells per block", self.block, BLOCK_RANGE),
            ("Window step", self.step, STEP_CELLS_RANGE if self.fast else STEP_RANGE),
            ("Window width", self.window_width, WINDOW_RANGE),
            ("Window height", self.window_height, WINDOW_RANGE),
        ):
            try:
                value = int(var.get())
            except ValueError:
                value = low - 1
            if not low <= value <= high:
                messagebox.showerror("Person detection", f"{name} must be a whole number between {low} and {high}.", parent=self)
                return False
            values.append(value)
        try:
            scales = parse_scales(self.scales.get())
        except ValueError as error:
            messagebox.showerror("Person detection", str(error), parent=self)
            return False
        metrics = tuple(key for key, var in self.metrics.items() if var.get())
        if not metrics and self.display.get() == "boxes":
            messagebox.showerror("Person detection", "Select at least one similarity metric.", parent=self)
            return False
        heatmap_metric = next(key for key, name in METRIC_NAMES.items() if name == self.heatmap_metric.get())
        orientations, cell, block, step, width, height = values
        self._settings = DetectionSettings(
            orientations=orientations,
            pixels_per_cell=cell,
            cells_per_block=block,
            step=step * cell if self.fast else step,  # always stored in pixels
            metrics=metrics,
            window_width=width,
            window_height=height,
            display=self.display.get(),
            heatmap_metric=heatmap_metric,
            scales=scales,
            variant=self.variant,
        )
        return True

    def apply(self) -> None:
        self.result = self._settings


class HogDetectionTool(ForensicsTool):
    tool_id = "hog_person_detection"
    title = "Person detection (HOG)"
    category = "Set 4: Features & Detection"
    description = "Find the window most similar to a template person image by comparing HOG descriptors."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by main window

        # 1. Search variant: exact (HOG per window) or fast (HOG once, sliced)
        variant = VariantDialog(parent, title="Person detection (HOG)").result
        if variant is None:
            return None

        # 2. Template image of a person; it is resized to the sliding window size
        filename = filedialog.askopenfilename(title="Select person template", filetypes=TEMPLATE_TYPES, parent=parent)
        if not filename:
            return None
        with Image.open(filename) as opened:
            template = opened.copy()

        # 3. The chosen variant's options: window size, HOG parameters, step, metrics and display
        title = f"Person detection (HOG) – {'fast' if variant == 'fast' else 'exact'}"
        settings = DetectionDialog(parent, title, template, variant).result
        if settings is None:
            return None

        # 4. Slide the window at every scale and score it with every metric (can take a while)
        parent.configure(cursor="watch")
        parent.update_idletasks()
        try:
            detections, score_maps = detect(document.current, template, settings)
        finally:
            parent.configure(cursor="")

        # 5. Heatmap (optional, at the scale where its metric matched best) and the winning boxes
        background = document.current
        if settings.display == "heatmap":
            heat_best = best_across_scales(score_maps, (settings.heatmap_metric,))[0]
            heat_map = next(m for m in score_maps if m.scale == heat_best.scale)
            background = render_heatmap(document.current, heat_map, settings.heatmap_metric, settings.step)
        output = draw_detections(background, detections)

        windows = sum(m.windows for m in score_maps)
        searched = [m.scale for m in score_maps]
        details: dict[str, Any] = {
            "Operation": "HOG person detection",
            "Library": "skimage.feature.hog",
            "Variant": VARIANTS[settings.variant][0],
            "Template": Path(filename).name,
            "Template size": f"{template.width} × {template.height}",
            "Window sizes": ", ".join(f"{m.width} × {m.height}" for m in score_maps),
            "Scales searched": ", ".join(f"{s:g}" for s in searched),
            "Orientations": settings.orientations,
            "Pixels per cell": f"{settings.pixels_per_cell} × {settings.pixels_per_cell}",
            "Cells per block": f"{settings.cells_per_block} × {settings.cells_per_block}",
            "Step": settings.step,
            "Windows evaluated": windows,
        }
        skipped = [s for s in settings.scales if s not in searched]
        if skipped:
            details["Scales skipped (window too big/small)"] = ", ".join(f"{s:g}" for s in skipped)
        for detection in detections:
            name = METRIC_NAMES[detection.method]
            details[f"{name} box (x, y, w, h)"] = f"({detection.col}, {detection.row}, {detection.width}, {detection.height})"
            details[f"{name} score"] = f"{detection.score:.4f}"
            if len(score_maps) > 1:
                details[f"{name} best scale"] = f"{detection.scale:g}"
        if settings.display == "heatmap":
            grid = heat_map.scores[settings.heatmap_metric]
            details["Heatmap metric"] = f"{METRIC_NAMES[settings.heatmap_metric]} (scale {heat_map.scale:g})"
            details["Heatmap range"] = f"{grid.min():.4f} … {grid.max():.4f}"

        shown = ", ".join(METRIC_NAMES[m].lower() for m in settings.metrics) or "no boxes"
        if settings.display == "heatmap":
            shown += f"; {METRIC_NAMES[settings.heatmap_metric].lower()} heatmap"
        return ToolResult(
            image=output,
            message=(
                f"Compared {windows} windows at {len(score_maps)} scale(s) against {Path(filename).name} ({shown})."
            ),
            details=details,
        )
