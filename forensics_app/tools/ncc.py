"""Set 4.1: Normalized cross-correlation image with skimage.feature.match_template."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk
from typing import Any

from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from skimage.feature import match_template
from skimage.transform import rotate

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .hog import show_figure

TEMPLATE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]

# Task 4.1 display modes
NCC_DISPLAYS = {
    "ncc": "Normalized cross-correlation image (canvas)",
    "popup": "Original, template & NCC comparison (popup)",
}


@dataclass(frozen=True)
class NccSettings:
    pad_input: bool = True
    display: str = "ncc"


def to_gray_float(image: Image.Image) -> np.ndarray:
    """Convert any PIL image to a 2-D float64 luminance array in [0, 1]."""
    return np.asarray(image.convert("L"), dtype=np.float64) / 255.0


def rotate_template(template: Image.Image, angle: float) -> np.ndarray:
    """Return the grayscale float template rotated by ``angle`` degrees using ``skimage.transform.rotate``."""
    gray = to_gray_float(template)
    if angle == 0.0:
        return gray
    return rotate(gray, angle)


def compute_ncc(
    image: Image.Image,
    template: Image.Image,
    *,
    angle: float = 0.0,
    pad_input: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized cross-correlation with ``skimage.feature.match_template``.

    Returns ``(corr_tmpl, rotated_template)`` where ``corr_tmpl`` is the 2-D correlation float array
    and ``rotated_template`` is the 2-D float template actually matched.
    """
    img_gray = to_gray_float(image)
    tmpl_gray = rotate_template(template, angle)
    if tmpl_gray.shape[0] > img_gray.shape[0] or tmpl_gray.shape[1] > img_gray.shape[1]:
        raise ValueError(
            f"The template ({tmpl_gray.shape[1]} × {tmpl_gray.shape[0]}) is larger than "
            f"the image ({img_gray.shape[1]} × {img_gray.shape[0]})."
        )
    corr = match_template(img_gray, tmpl_gray, pad_input=pad_input)
    return corr, tmpl_gray


def ncc_to_image(corr: np.ndarray) -> Image.Image:
    """Rescale a correlation array from [min, max] to a 0..255 grayscale ``L`` image for display."""
    low, high = float(corr.min()), float(corr.max())
    if high > low:
        scaled = (corr - low) / (high - low)
    else:
        scaled = np.zeros_like(corr)
    return Image.fromarray(np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8), mode="L")


def ncc_comparison_figure(
    image: Image.Image,
    tmpl_gray: np.ndarray,
    corr: np.ndarray,
) -> Figure:
    """Build the 3-panel figure from Set 4.1: Original image, Template, Normalized cross-correlation image."""
    figure = Figure(figsize=(12, 4.5), layout="constrained")
    ax = figure.subplots(1, 3)
    ax[0].imshow(to_gray_float(image), cmap="gray")
    ax[0].set_title("Original image")
    ax[1].imshow(tmpl_gray, cmap="gray")
    ax[1].set_title("Template")
    ax[2].imshow(corr, cmap="gray")
    ax[2].set_title("Normalized cross-correlation image")
    return figure


class NccDialog(simpledialog.Dialog):
    """Dialog for Set 4.1: Normalized cross-correlation image."""

    def __init__(self, parent: tk.Misc, title: str, template: Image.Image) -> None:
        self.template = template
        super().__init__(parent, title=title)

    def body(self, master: tk.Frame) -> tk.Widget:
        self.pad_input = tk.BooleanVar(value=True)
        self.display = tk.StringVar(value="ncc")

        ttk.Label(
            master,
            text=f"Template size: {self.template.width} × {self.template.height} px",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        pad_check = ttk.Checkbutton(
            master,
            text="Pad input (pad_input=True, keeps same size as original image)",
            variable=self.pad_input,
        )
        pad_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(master, text="Show as:").grid(row=2, column=0, columnspan=2, sticky="w")
        for row, (key, text) in enumerate(NCC_DISPLAYS.items(), start=3):
            ttk.Radiobutton(master, text=text, value=key, variable=self.display).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=(12, 0)
            )
        return pad_check

    def validate(self) -> bool:
        self._settings = NccSettings(pad_input=self.pad_input.get(), display=self.display.get())
        return True

    def apply(self) -> None:
        self.result = self._settings


class NormalizedCrossCorrelationTool(ForensicsTool):
    tool_id = "ncc_image"
    title = "Normalized cross-correlation"
    category = "Set 4: Features & Detection"
    description = "Compute the normalized cross-correlation (NCC) map between the image and a template."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by main window

        filename = filedialog.askopenfilename(title="Select template image", filetypes=TEMPLATE_TYPES, parent=parent)
        if not filename:
            return None
        with Image.open(filename) as opened:
            template = opened.copy()

        settings = NccDialog(parent, title="Normalized cross-correlation", template=template).result
        if settings is None:
            return None

        corr, tmpl_gray = compute_ncc(document.current, template, pad_input=settings.pad_input)
        peak_row, peak_col = np.unravel_index(int(np.argmax(corr)), corr.shape)
        details: dict[str, Any] = {
            "Operation": "Normalized cross-correlation",
            "Library": "skimage.feature.match_template",
            "Template": Path(filename).name,
            "Template size": f"{template.width} × {template.height}",
            "Pad input": str(settings.pad_input),
            "Output size": f"{corr.shape[1]} × {corr.shape[0]}",
            "Min correlation": f"{float(corr.min()):.4f}",
            "Max correlation": f"{float(corr.max()):.4f}",
            "Peak location (x, y)": f"({int(peak_col)}, {int(peak_row)})",
            "Display": NCC_DISPLAYS[settings.display],
        }

        if settings.display == "popup":
            figure = ncc_comparison_figure(document.current, tmpl_gray, corr)
            show_figure(parent, "Normalized cross-correlation", figure)
            return ToolResult(
                message=f"Opened normalized cross-correlation comparison for {Path(filename).name}.",
                details=details,
            )

        output = ncc_to_image(corr)
        return ToolResult(
            image=output,
            message=(
                f"Computed normalized cross-correlation with {Path(filename).name} "
                f"(peak={float(corr.max()):.4f} at ({int(peak_col)}, {int(peak_row)}))."
            ),
            details=details,
        )

