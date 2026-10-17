"""Set 4.3: HOG (Histogram of Oriented Gradients) extraction and visualization with skimage.feature.hog."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Any

import numpy as np
from PIL import Image
from skimage.exposure import rescale_intensity
from skimage.feature import hog

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .modes import split_alpha

ORIENTATIONS_RANGE = (1, 36)
CELL_RANGE = (2, 64)
BLOCK_RANGE = (1, 8)

# How the HOG is shown (key -> label in the dialog)
DISPLAYS = {
    "hog": "HOG image (white gradients on black)",
    "overlay": "HOG drawn in red over the dimmed image",
}
DIM_FACTOR = 0.4  # brightness kept by the image under the overlay


@dataclass(frozen=True)
class HogSettings:
    orientations: int = 9
    pixels_per_cell: int = 8
    cells_per_block: int = 3
    display: str = "hog"


def to_rgb_array(image: Image.Image) -> np.ndarray:
    """Opaque RGB float array in [0, 1]; gray images get three identical channels."""
    base, _alpha = split_alpha(image)
    return np.asarray(base.convert("RGB"), dtype=np.float64) / 255.0


def extract_hog(image: Image.Image, settings: HogSettings) -> tuple[np.ndarray, np.ndarray]:
    """Return the HOG feature vector and its visualization (float image) for ``image``."""
    min_side = settings.pixels_per_cell * settings.cells_per_block
    if min(image.size) < min_side:
        raise ValueError(
            f"The image must be at least {min_side} px on each side "
            f"(pixels per cell × cells per block)."
        )
    cell, block = settings.pixels_per_cell, settings.cells_per_block
    return hog(
        to_rgb_array(image),
        orientations=settings.orientations,
        pixels_per_cell=(cell, cell),
        cells_per_block=(block, block),
        visualize=True,
        channel_axis=2,
    )


def render_hog(image: Image.Image, hog_image: np.ndarray, display: str) -> Image.Image:
    """Draw the HOG visualization according to ``display`` (a key of ``DISPLAYS``)."""
    if display not in DISPLAYS:
        raise ValueError(f"Display must be one of {', '.join(DISPLAYS)}.")
    # Contrast-stretch the visualization so faint gradients are visible
    strength = rescale_intensity(hog_image, out_range=(0.0, 1.0)) if hog_image.max() > 0 else hog_image
    if display == "hog":
        return Image.fromarray(np.round(strength * 255).astype(np.uint8), mode="L")

    rgb = to_rgb_array(image) * DIM_FACTOR
    alpha = strength[..., None]
    out = (1.0 - alpha) * rgb + alpha * np.array([1.0, 0.0, 0.0])
    return Image.fromarray(np.clip(np.round(out * 255), 0, 255).astype(np.uint8), mode="RGB")


class HogDialog(simpledialog.Dialog):
    """Ask for the HOG parameters and how to display the result."""

    def body(self, master: tk.Frame) -> tk.Widget:
        self.orientations = tk.StringVar(value="9")
        self.cell = tk.StringVar(value="8")
        self.block = tk.StringVar(value="3")
        self.display = tk.StringVar(value="hog")

        rows = (
            ("Orientations (histogram bins):", self.orientations, ORIENTATIONS_RANGE),
            ("Pixels per cell (N × N):", self.cell, CELL_RANGE),
            ("Cells per block (N × N):", self.block, BLOCK_RANGE),
        )
        first = None
        for row, (text, var, (low, high)) in enumerate(rows):
            ttk.Label(master, text=text).grid(row=row, column=0, sticky="w")
            box = ttk.Spinbox(master, from_=low, to=high, increment=1, textvariable=var, width=6)
            box.grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
            first = first or box
        ttk.Label(master, text="Show as:").grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))
        for row, (key, text) in enumerate(DISPLAYS.items(), start=4):
            ttk.Radiobutton(master, text=text, value=key, variable=self.display).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=(12, 0)
            )
        return first

    def validate(self) -> bool:
        values = []
        for name, var, (low, high) in (
            ("Orientations", self.orientations, ORIENTATIONS_RANGE),
            ("Pixels per cell", self.cell, CELL_RANGE),
            ("Cells per block", self.block, BLOCK_RANGE),
        ):
            try:
                value = int(var.get())
            except ValueError:
                value = low - 1
            if not low <= value <= high:
                messagebox.showerror("HOG", f"{name} must be a whole number between {low} and {high}.", parent=self)
                return False
            values.append(value)
        self._settings = HogSettings(*values, display=self.display.get())
        return True

    def apply(self) -> None:
        self.result = self._settings


class HogTool(ForensicsTool):
    tool_id = "hog"
    title = "HOG features"
    category = "Set 4: Features & Detection"
    description = "Extract Histogram of Oriented Gradients features and visualize them (configurable bins, cell and block size)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by main window

        settings = HogDialog(parent, title="HOG features").result
        if settings is None:
            return None

        features, hog_image = extract_hog(document.current, settings)
        output = render_hog(document.current, hog_image, settings.display)

        cell, block = settings.pixels_per_cell, settings.cells_per_block
        cells_y, cells_x = document.current.height // cell, document.current.width // cell
        details: dict[str, Any] = {
            "Operation": "HOG extraction",
            "Library": "skimage.feature.hog",
            "Orientations": settings.orientations,
            "Pixels per cell": f"{cell} × {cell}",
            "Cells per block": f"{block} × {block}",
            "Cells": f"{cells_x} × {cells_y}",
            "Blocks": f"{cells_x - block + 1} × {cells_y - block + 1}",
            "Feature length": int(features.size),
            "Display": DISPLAYS[settings.display],
        }
        return ToolResult(
            image=output,
            message=(
                f"Extracted HOG ({settings.orientations} orientations, {cell}×{cell} cells, "
                f"{block}×{block} blocks): {features.size} features."
            ),
            details=details,
        )
