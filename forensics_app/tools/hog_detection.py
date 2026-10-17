"""Set 4.4: Person detection by sliding a window and comparing its HOG to a template's HOG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

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
WINDOW_RANGE = (4, 4096)

# Similarity measures (key -> label) and the box colour used to draw each match
METHODS = {
    "euclidean": "Euclidean distance (smallest wins)",
    "dot": "Dot product (largest wins)",
    "both": "Both",
}
BOX_COLOURS = {"euclidean": (255, 0, 0), "dot": (0, 120, 255)}
BOX_WIDTH = 2


@dataclass(frozen=True)
class DetectionSettings:
    orientations: int = 16
    pixels_per_cell: int = 8
    cells_per_block: int = 2
    step: int = 8
    method: str = "both"
    # Sliding window size; ``None`` keeps the template's own width/height
    window_width: int | None = None
    window_height: int | None = None


@dataclass(frozen=True)
class Detection:
    """Best window found by one similarity measure: top-left corner, size and score."""

    method: str
    row: int
    col: int
    height: int
    width: int
    score: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.col, self.row, self.col + self.width - 1, self.row + self.height - 1


def hog_features(rgb: np.ndarray, settings: DetectionSettings) -> np.ndarray:
    cell, block = settings.pixels_per_cell, settings.cells_per_block
    return hog(
        rgb,
        orientations=settings.orientations,
        pixels_per_cell=(cell, cell),
        cells_per_block=(block, block),
        channel_axis=2,
    )


def resize_template(template: Image.Image, settings: DetectionSettings) -> Image.Image:
    """Resize the template to the sliding window size, so the search can run at different scales."""
    size = (settings.window_width or template.width, settings.window_height or template.height)
    if size == template.size:
        return template
    base, _alpha = split_alpha(template)
    return base.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def detect(image: Image.Image, template: Image.Image, settings: DetectionSettings) -> tuple[list[Detection], int]:
    """Slide a window over ``image`` and score each window's HOG against the template's.

    The template is first resized to the window size from ``settings``. Returns the best
    detection per requested method and the number of windows evaluated.
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

    tmpl_fd = hog_features(tmpl, settings)
    coords: list[tuple[int, int]] = []
    euclidean: list[float] = []
    dot: list[float] = []
    for i in range(0, img.shape[0] - win_h + 1, settings.step):
        for j in range(0, img.shape[1] - win_w + 1, settings.step):
            window_fd = hog_features(img[i : i + win_h, j : j + win_w], settings)
            coords.append((i, j))
            euclidean.append(float(np.linalg.norm(tmpl_fd - window_fd)))
            dot.append(float(np.dot(window_fd, tmpl_fd)))

    detections = []
    if settings.method in ("euclidean", "both"):
        best = int(np.argmin(euclidean))
        detections.append(Detection("euclidean", *coords[best], win_h, win_w, euclidean[best]))
    if settings.method in ("dot", "both"):
        best = int(np.argmax(dot))
        detections.append(Detection("dot", *coords[best], win_h, win_w, dot[best]))
    return detections, len(coords)


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    base, _alpha = split_alpha(image)
    output = base.convert("RGB")
    draw = ImageDraw.Draw(output)
    for detection in detections:
        draw.rectangle(detection.box, outline=BOX_COLOURS[detection.method], width=BOX_WIDTH)
    return output


class DetectionDialog(simpledialog.Dialog):
    """Ask for the HOG parameters, the sliding step and the similarity measure."""

    def __init__(self, parent: tk.Misc, title: str, template: Image.Image) -> None:
        self.template = template
        super().__init__(parent, title=title)

    def body(self, master: tk.Frame) -> tk.Widget:
        self.orientations = tk.StringVar(value="16")
        self.cell = tk.StringVar(value="8")
        self.block = tk.StringVar(value="2")
        self.step = tk.StringVar(value="8")
        self.method = tk.StringVar(value="both")
        self.window_width = tk.StringVar(value=str(self.template.width))
        self.window_height = tk.StringVar(value=str(self.template.height))
        self.keep_aspect = tk.BooleanVar(value=True)

        ttk.Label(master, text=f"Template size: {self.template.width} × {self.template.height} px").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
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
        ttk.Label(master, text="The template is resized to the window: try several sizes to search other scales.").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        self.window_width.trace_add("write", lambda *_: self._sync_height())
        self._sync_height()

        rows = (
            ("Orientations (histogram bins):", self.orientations, ORIENTATIONS_RANGE),
            ("Pixels per cell (N × N):", self.cell, CELL_RANGE),
            ("Cells per block (N × N):", self.block, BLOCK_RANGE),
            ("Window step (px):", self.step, STEP_RANGE),
        )
        first = None
        for row, (text, var, (low, high)) in enumerate(rows, start=4):
            ttk.Label(master, text=text).grid(row=row, column=0, sticky="w")
            box = ttk.Spinbox(master, from_=low, to=high, increment=1, textvariable=var, width=6)
            box.grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
            first = first or box
        ttk.Label(master, text="Smaller step = more precise but slower.").grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        ttk.Label(master, text="Similarity measure:").grid(row=9, column=0, columnspan=2, sticky="w")
        for row, (key, text) in enumerate(METHODS.items(), start=10):
            ttk.Radiobutton(master, text=text, value=key, variable=self.method).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=(12, 0)
            )
        ttk.Label(master, text="Boxes: red = Euclidean, blue = dot product.").grid(
            row=13, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
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

    def validate(self) -> bool:
        values = []
        for name, var, (low, high) in (
            ("Orientations", self.orientations, ORIENTATIONS_RANGE),
            ("Pixels per cell", self.cell, CELL_RANGE),
            ("Cells per block", self.block, BLOCK_RANGE),
            ("Window step", self.step, STEP_RANGE),
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
        *hog_values, width, height = values
        self._settings = DetectionSettings(
            *hog_values, method=self.method.get(), window_width=width, window_height=height
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

        # 1. Template image of a person; it is resized to the sliding window size
        filename = filedialog.askopenfilename(title="Select person template", filetypes=TEMPLATE_TYPES, parent=parent)
        if not filename:
            return None
        with Image.open(filename) as opened:
            template = opened.copy()

        # 2. Window size, HOG parameters, step and similarity measure
        settings = DetectionDialog(parent, "Person detection (HOG)", template).result
        if settings is None:
            return None

        # 3. Slide the window and keep the best match per measure (can take a while)
        parent.configure(cursor="watch")
        parent.update_idletasks()
        try:
            detections, windows = detect(document.current, template, settings)
        finally:
            parent.configure(cursor="")

        # 4. Draw the winning boxes
        output = draw_detections(document.current, detections)
        window_w = settings.window_width or template.width
        window_h = settings.window_height or template.height

        details: dict[str, Any] = {
            "Operation": "HOG person detection",
            "Library": "skimage.feature.hog",
            "Template": Path(filename).name,
            "Template size": f"{template.width} × {template.height}",
            "Window size": f"{window_w} × {window_h}",
            "Scale": f"{window_w / template.width:.2f} × {window_h / template.height:.2f}",
            "Orientations": settings.orientations,
            "Pixels per cell": f"{settings.pixels_per_cell} × {settings.pixels_per_cell}",
            "Cells per block": f"{settings.cells_per_block} × {settings.cells_per_block}",
            "Step": settings.step,
            "Windows evaluated": windows,
        }
        for detection in detections:
            name = "Euclidean" if detection.method == "euclidean" else "Dot product"
            details[f"{name} box (x, y)"] = f"({detection.col}, {detection.row})"
            details[f"{name} score"] = f"{detection.score:.4f}"

        return ToolResult(
            image=output,
            message=f"Compared {windows} windows against {Path(filename).name} ({METHODS[settings.method].lower()}).",
            details=details,
        )
