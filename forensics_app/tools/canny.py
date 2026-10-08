"""Functionality 3: Canny edge detector from skimage.feature with configurable sigma and display."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Any

import numpy as np
from PIL import Image
from skimage.feature import canny
from skimage.measure import label

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .modes import split_alpha

# How the detected edges are shown (key -> label in the dialog)
DISPLAYS = {
    "edge_map": "Edge map (white edges on black)",
    "red_edges": "Red edges on the image",
    "red_overlay": "Transparent red overlay",
    "darken": "Darken everything except edges",
    "contours": "Edges as contours (one colour per contour)",
}
RED = np.array([255, 0, 0], dtype=np.float64)
OVERLAY_OPACITY = 0.5  # weight of red in the transparent overlay
DARKEN_FACTOR = 0.25  # brightness kept by non-edge pixels


@dataclass(frozen=True)
class CannySettings:
    sigma: float = 1.5
    display: str = "edge_map"


def detect_edges(image: Image.Image, sigma: float) -> np.ndarray:
    """Return a boolean edge map of ``image`` (computed on its luminance)."""
    gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    return canny(gray, sigma=sigma)


def label_contours(edges: np.ndarray) -> tuple[np.ndarray, int]:
    """Split the edge map into contours: chains of 8-connected edge pixels."""
    labels, count = label(edges, connectivity=2, return_num=True)
    return labels, int(count)


def contour_colours(count: int) -> np.ndarray:
    """``count`` distinct, saturated RGB colours (golden-ratio hue steps)."""
    hues = (np.arange(count) * 0.618033988749895) % 1.0
    return np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues], dtype=np.float64).reshape(-1, 3) * 255.0


def render_edges(image: Image.Image, edges: np.ndarray, display: str) -> Image.Image:
    """Draw the boolean ``edges`` map according to ``display`` (a key of ``DISPLAYS``)."""
    if display not in DISPLAYS:
        raise ValueError(f"Display must be one of {', '.join(DISPLAYS)}.")
    if display == "edge_map":
        return Image.fromarray((edges * 255).astype(np.uint8), mode="L")

    base, _alpha = split_alpha(image)  # edges are drawn on an opaque RGB copy
    rgb = np.asarray(base.convert("RGB"), dtype=np.float64)
    out = rgb.copy()
    if display == "red_edges":
        out[edges] = RED
    elif display == "red_overlay":
        out[edges] = (1.0 - OVERLAY_OPACITY) * rgb[edges] + OVERLAY_OPACITY * RED
    elif display == "darken":
        out[~edges] *= DARKEN_FACTOR
    elif display == "contours":
        labels, count = label_contours(edges)
        if count:
            colours = contour_colours(count)
            out[edges] = colours[labels[edges] - 1]
    return Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8), mode="RGB")


class CannyDialog(simpledialog.Dialog):
    """Ask for the smoothing sigma and how the edges should be displayed."""

    def body(self, master: tk.Frame) -> tk.Widget:
        self.sigma = tk.StringVar(value="1.5")
        self.display = tk.StringVar(value="edge_map")

        ttk.Label(master, text="Gaussian smoothing sigma:").grid(row=0, column=0, sticky="w")
        sigma_box = ttk.Spinbox(master, from_=0.1, to=10.0, increment=0.1, textvariable=self.sigma, width=6)
        sigma_box.grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(master, text="Higher sigma = fewer, smoother edges.").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )
        ttk.Label(master, text="Show edges as:").grid(row=2, column=0, columnspan=2, sticky="w")
        for row, (key, text) in enumerate(DISPLAYS.items(), start=3):
            ttk.Radiobutton(master, text=text, value=key, variable=self.display).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=(12, 0)
            )
        return sigma_box

    def validate(self) -> bool:
        try:
            sigma = float(self.sigma.get())
        except ValueError:
            sigma = -1.0
        if not 0.1 <= sigma <= 10.0:
            messagebox.showerror("Canny", "Sigma must be a number between 0.1 and 10.", parent=self)
            return False
        self._settings = CannySettings(sigma=sigma, display=self.display.get())
        return True

    def apply(self) -> None:
        self.result = self._settings


class CannyTool(ForensicsTool):
    tool_id = "canny"
    title = "Canny edge detection"
    category = "Set 3: Edge Detection"
    description = "Detect thin edges with Canny and show them as a map, red lines, overlay, highlight or contours."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # 1. User dialog for sigma and display style
        settings = CannyDialog(parent, title="Canny edge detection").result
        if settings is None:
            return None

        # 2. Apply skimage.feature.canny on the luminance
        edges_bool = detect_edges(document.current, settings.sigma)

        # 3. Render the edges in the chosen style
        output_img = render_edges(document.current, edges_bool, settings.display)

        details: dict[str, Any] = {
            "Operation": "Canny Edge Detection",
            "Library": "skimage.feature.canny",
            "Configured Sigma": settings.sigma,
            "Display": DISPLAYS[settings.display],
            "Edge Pixels": int(np.sum(edges_bool)),
            "Edge Density": f"{(float(np.mean(edges_bool)) * 100.0):.2f}%",
        }
        if settings.display == "contours":
            details["Contours"] = label_contours(edges_bool)[1]

        return ToolResult(
            image=output_img,
            message=f"Detected edges using Canny with sigma={settings.sigma} ({DISPLAYS[settings.display].lower()}).",
            details=details,
        )
