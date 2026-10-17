"""Set 4.3: HOG (Histogram of Oriented Gradients) extraction and visualization with skimage.feature.hog."""

from __future__ import annotations

from dataclasses import dataclass, replace
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Any

from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
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
MAX_COMPARED = 6  # orientation counts shown side by side

# How the HOG is shown (key -> label in the dialog)
DISPLAYS = {
    "hog": "HOG image (white gradients on black)",
    "overlay": "HOG drawn in red over the dimmed image",
    "compare": "Side-by-side comparison of orientation counts (popup)",
    "inspect": "Cell inspector: click a cell to see its histogram (popup)",
}
POPUP_DISPLAYS = ("compare", "inspect")  # leave the working image unchanged
DIM_FACTOR = 0.4  # brightness kept by the image under the overlay


@dataclass(frozen=True)
class HogSettings:
    orientations: int = 9
    pixels_per_cell: int = 8
    cells_per_block: int = 3
    display: str = "hog"
    compare_orientations: tuple[int, ...] = (2, 4, 9)


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


def stretch(hog_image: np.ndarray) -> np.ndarray:
    """Contrast-stretch the visualization to [0, 1] so faint gradients are visible."""
    return rescale_intensity(hog_image, out_range=(0.0, 1.0)) if hog_image.max() > 0 else hog_image


def render_hog(image: Image.Image, hog_image: np.ndarray, display: str) -> Image.Image:
    """Draw the HOG visualization as an image (``display`` is ``"hog"`` or ``"overlay"``)."""
    if display not in ("hog", "overlay"):
        raise ValueError("Display must be 'hog' or 'overlay'.")
    strength = stretch(hog_image)
    if display == "hog":
        return Image.fromarray(np.round(strength * 255).astype(np.uint8), mode="L")

    rgb = to_rgb_array(image) * DIM_FACTOR
    alpha = strength[..., None]
    out = (1.0 - alpha) * rgb + alpha * np.array([1.0, 0.0, 0.0])
    return Image.fromarray(np.clip(np.round(out * 255), 0, 255).astype(np.uint8), mode="RGB")


def parse_orientations(text: str) -> tuple[int, ...]:
    """Parse ``"2, 4, 9"`` into the orientation counts to compare (kept in the given order)."""
    try:
        values = [int(part) for part in text.replace(";", ",").split(",") if part.strip()]
    except ValueError:
        raise ValueError("Orientations to compare must be whole numbers separated by commas, e.g. 2, 4, 9.") from None
    values = list(dict.fromkeys(values))  # drop repeats
    low, high = ORIENTATIONS_RANGE
    if not 1 <= len(values) <= MAX_COMPARED:
        raise ValueError(f"Enter between 1 and {MAX_COMPARED} orientation counts to compare.")
    if not all(low <= v <= high for v in values):
        raise ValueError(f"Each orientation count must be between {low} and {high}.")
    return tuple(values)


def cell_histograms(image: Image.Image, pixels_per_cell: int, orientations: int) -> np.ndarray:
    """Raw orientation histogram of every cell, before block normalization (as skimage computes it).

    Returns an array of shape (cell rows, cell cols, orientations): the mean gradient magnitude
    voting for each orientation bin. Bin ``k`` covers gradient angles ``[k, k + 1) × 180 / orientations``
    degrees (unsigned: a gradient and its opposite count the same).
    """
    rgb = to_rgb_array(image)
    # 1. Central-difference gradients, zero on the outermost pixels (exactly as skimage does)
    g_row = np.zeros_like(rgb)
    g_col = np.zeros_like(rgb)
    g_row[1:-1] = rgb[2:] - rgb[:-2]
    g_col[:, 1:-1] = rgb[:, 2:] - rgb[:, :-2]
    # 2. Per pixel, keep the colour channel with the strongest gradient
    magnitude = np.hypot(g_row, g_col)
    strongest = magnitude.argmax(axis=2)[..., None]
    g_row = np.take_along_axis(g_row, strongest, axis=2)[..., 0]
    g_col = np.take_along_axis(g_col, strongest, axis=2)[..., 0]
    magnitude = np.take_along_axis(magnitude, strongest, axis=2)[..., 0]
    # 3. Unsigned orientation in [0, 180) -> bin index
    angle = np.rad2deg(np.arctan2(g_row, g_col)) % 180
    bins = np.minimum((angle // (180 / orientations)).astype(int), orientations - 1)
    # 4. Each pixel votes its magnitude into its cell's bin (leftover pixels past the last full cell are ignored)
    rows, cols = rgb.shape[0] // pixels_per_cell, rgb.shape[1] // pixels_per_cell
    h, w = rows * pixels_per_cell, cols * pixels_per_cell
    cell_r = np.arange(h)[:, None] // pixels_per_cell
    cell_c = np.arange(w)[None, :] // pixels_per_cell
    hist = np.zeros((rows, cols, orientations))
    np.add.at(hist, (np.broadcast_to(cell_r, (h, w)), np.broadcast_to(cell_c, (h, w)), bins[:h, :w]), magnitude[:h, :w])
    return hist / (pixels_per_cell * pixels_per_cell)


def bin_centres(orientations: int) -> np.ndarray:
    width = 180 / orientations
    return (np.arange(orientations) + 0.5) * width


def comparison_figure(image: Image.Image, settings: HogSettings) -> tuple[Figure, dict[int, int]]:
    """The original image next to its HOG for each of ``settings.compare_orientations``.

    Returns the figure and the feature-vector length for each orientation count.
    """
    counts = settings.compare_orientations
    figure = Figure(figsize=(3.6 * (len(counts) + 1), 4.4), layout="constrained")
    axes = figure.subplots(1, len(counts) + 1, sharex=True, sharey=True)
    base, _alpha = split_alpha(image)
    axes[0].imshow(base.convert("RGB"))
    axes[0].set_title("Original image")
    lengths = {}
    for ax, count in zip(axes[1:], counts):
        features, hog_image = extract_hog(image, replace(settings, orientations=count))
        lengths[count] = int(features.size)
        ax.imshow(stretch(hog_image), cmap="gray")
        ax.set_title(f"HOG, orientations={count}\n{features.size} features")
    cell, block = settings.pixels_per_cell, settings.cells_per_block
    figure.suptitle(f"pixels_per_cell=({cell}, {cell}), cells_per_block=({block}, {block})")
    return figure, lengths


class CellInspector:
    """Image with its HOG on the left; clicking a cell plots that cell's orientation histogram on the right."""

    def __init__(self, image: Image.Image, settings: HogSettings) -> None:
        self.settings = settings
        _features, hog_image = extract_hog(image, settings)
        self.hist = cell_histograms(image, settings.pixels_per_cell, settings.orientations)
        self.strongest = float(self.hist.sum(axis=2).max())

        self.figure = Figure(figsize=(11, 5.2), layout="constrained")
        self.ax_image, self.ax_hist = self.figure.subplots(1, 2, gridspec_kw={"width_ratios": [3, 2]})
        self.ax_image.imshow(render_hog(image, hog_image, "overlay"))
        self.ax_image.set_title("Click a cell (HOG in red)")
        cell = settings.pixels_per_cell
        self.highlight = Rectangle((0, 0), cell, cell, fill=False, edgecolor="cyan", linewidth=2)
        self.ax_image.add_patch(self.highlight)

        # Start on the cell with the strongest gradients so the chart is not empty
        start = np.unravel_index(self.hist.sum(axis=2).argmax(), self.hist.shape[:2])
        self.select(int(start[0]), int(start[1]))

    def cell_at(self, x: float, y: float) -> tuple[int, int] | None:
        """The (row, col) of the cell under image coordinates (x, y), or ``None`` outside the cell grid."""
        row, col = int(y) // self.settings.pixels_per_cell, int(x) // self.settings.pixels_per_cell
        if 0 <= row < self.hist.shape[0] and 0 <= col < self.hist.shape[1] and x >= 0 and y >= 0:
            return row, col
        return None

    def on_click(self, event: MouseEvent) -> None:
        if event.inaxes is not self.ax_image or event.xdata is None or event.ydata is None:
            return
        cell = self.cell_at(event.xdata, event.ydata)
        if cell is not None:
            self.select(*cell)
            self.figure.canvas.draw_idle()

    def select(self, row: int, col: int) -> None:
        self.selected = (row, col)
        cell, n = self.settings.pixels_per_cell, self.settings.orientations
        x0, y0 = col * cell, row * cell
        self.highlight.set_xy((x0 - 0.5, y0 - 0.5))

        values = self.hist[row, col]
        centres = bin_centres(n)
        ax = self.ax_hist
        ax.clear()
        ax.bar(centres, values, width=180 / n * 0.85, color="tab:red", edgecolor="black", linewidth=0.5)
        ax.set_xlim(0, 180)
        ax.set_ylim(0, max(float(values.max()) * 1.15, 1e-3))  # per cell, so weak cells stay readable
        if n <= 12:
            ax.set_xticks(centres, [f"{c:g}°" for c in centres])
        ax.set_xlabel("Gradient orientation (0° = vertical edge)")
        ax.set_ylabel("Mean gradient magnitude")
        if values.any():
            strength = 100 * values.sum() / self.strongest
            summary = f"dominant bin: {centres[values.argmax()]:g}° · strength: {strength:.0f}% of max"
        else:
            summary = "no gradients (flat)"
        ax.set_title(f"Cell (row {row}, col {col}): x {x0}–{x0 + cell - 1}, y {y0}–{y0 + cell - 1}\n{summary}")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)


class HogDialog(simpledialog.Dialog):
    """Ask for the HOG parameters and how to display the result."""

    def body(self, master: tk.Frame) -> tk.Widget:
        self.orientations = tk.StringVar(value="9")
        self.cell = tk.StringVar(value="8")
        self.block = tk.StringVar(value="3")
        self.display = tk.StringVar(value="hog")
        self.compare = tk.StringVar(value="2, 4, 9")

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
        self._orientations_box = first
        ttk.Label(master, text="Show as:").grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))
        row = 4
        for key, text in DISPLAYS.items():
            ttk.Radiobutton(master, text=text, value=key, variable=self.display, command=self._sync).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=(12, 0)
            )
            row += 1
            if key == "compare":  # its own option sits right under it
                compare = ttk.Frame(master)
                compare.grid(row=row, column=0, columnspan=2, sticky="w", padx=(32, 0), pady=(0, 2))
                ttk.Label(compare, text="Orientations to compare:").pack(side="left")
                self._compare_entry = ttk.Entry(compare, textvariable=self.compare, width=14)
                self._compare_entry.pack(side="left", padx=(6, 0))
                row += 1
        self._sync()
        return first

    def _sync(self) -> None:
        """The comparison uses its own list of orientation counts instead of the single value."""
        comparing = self.display.get() == "compare"
        self._compare_entry.state(["!disabled"] if comparing else ["disabled"])
        self._orientations_box.state(["disabled"] if comparing else ["!disabled"])

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
        try:
            compare = parse_orientations(self.compare.get())
        except ValueError as error:
            if self.display.get() == "compare":
                messagebox.showerror("HOG", str(error), parent=self)
                return False
            compare = HogSettings.compare_orientations
        self._settings = HogSettings(*values, display=self.display.get(), compare_orientations=compare)
        return True

    def apply(self) -> None:
        self.result = self._settings


def show_figure(parent: tk.Misc, title: str, figure: Figure) -> tk.Toplevel:
    """Mount a matplotlib figure in a new window."""
    popup = tk.Toplevel(parent)
    popup.title(title)
    canvas = FigureCanvasTkAgg(figure, master=popup)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return popup


class HogTool(ForensicsTool):
    tool_id = "hog"
    title = "HOG features"
    category = "Set 4: Features & Detection"
    description = (
        "Extract Histogram of Oriented Gradients features and visualize them, compare orientation "
        "counts side by side, or inspect a cell's histogram."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by main window

        settings = HogDialog(parent, title="HOG features").result
        if settings is None:
            return None

        image = document.current
        cell, block = settings.pixels_per_cell, settings.cells_per_block
        cells_y, cells_x = image.height // cell, image.width // cell
        details: dict[str, Any] = {
            "Operation": "HOG extraction",
            "Library": "skimage.feature.hog",
            "Pixels per cell": f"{cell} × {cell}",
            "Cells per block": f"{block} × {block}",
            "Cells": f"{cells_x} × {cells_y}",
            "Blocks": f"{cells_x - block + 1} × {cells_y - block + 1}",
            "Display": DISPLAYS[settings.display],
        }

        if settings.display == "compare":
            figure, lengths = comparison_figure(image, settings)
            show_figure(parent, "HOG: orientation comparison", figure)
            for count, length in lengths.items():
                details[f"Features (orientations={count})"] = length
            counts = ", ".join(map(str, settings.compare_orientations))
            return ToolResult(message=f"Compared HOG with orientations {counts} side by side.", details=details)

        details["Orientations"] = settings.orientations
        if settings.display == "inspect":
            inspector = CellInspector(image, settings)
            popup = show_figure(parent, "HOG: cell inspector", inspector.figure)
            inspector.figure.canvas.mpl_connect("button_press_event", inspector.on_click)
            popup.inspector = inspector  # matplotlib keeps only a weak reference to the click handler
            return ToolResult(
                message="Opened the HOG cell inspector: click a cell to see its orientation histogram.",
                details=details,
            )

        features, hog_image = extract_hog(image, settings)
        output = render_hog(image, hog_image, settings.display)
        details["Feature length"] = int(features.size)
        return ToolResult(
            image=output,
            message=(
                f"Extracted HOG ({settings.orientations} orientations, {cell}×{cell} cells, "
                f"{block}×{block} blocks): {features.size} features."
            ),
            details=details,
        )
