"""Functionality 6: Global histogram equalization and CLAHE (adaptive equalization)."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog
from typing import Any, Callable

import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist, equalize_hist

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def _equalize_luminance(image: Image.Image, equalize: Callable[[np.ndarray], np.ndarray]) -> Image.Image:
    """Apply ``equalize`` (float [0, 1] -> float [0, 1]) to the intensity of ``image``.

    Colour images are equalized on the Y (luma) channel only, so hues are preserved
    instead of being shifted by equalizing R, G and B independently.
    """
    if image.mode in ("L", "I", "F", "1"):
        gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
        return Image.fromarray(_to_uint8(equalize(gray)), mode="L")

    y, cb, cr = image.convert("RGB").convert("YCbCr").split()
    y_arr = np.asarray(y, dtype=np.float64) / 255.0
    y_eq = Image.fromarray(_to_uint8(equalize(y_arr)), mode="L")
    return Image.merge("YCbCr", (y_eq, cb, cr)).convert("RGB")


def _to_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values * 255.0), 0, 255).astype(np.uint8)


def _intensity_stats(image: Image.Image) -> tuple[float, float]:
    arr = np.asarray(image.convert("L"), dtype=np.float64)
    return float(arr.mean()), float(arr.std())


class HistogramEqualizationTool(ForensicsTool):
    tool_id = "histogram_equalization"
    title = "Histogram equalization"
    category = "Set 2: Image Processing"
    description = "Spread intensities over the full range using a global cumulative histogram."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by main window

        source = document.current
        output_image = _equalize_luminance(source, lambda arr: equalize_hist(arr, nbins=256))

        mean_in, std_in = _intensity_stats(source)
        mean_out, std_out = _intensity_stats(output_image)
        details: dict[str, Any] = {
            "Operation": "Histogram Equalization",
            "Library": "skimage.exposure.equalize_hist",
            "Channel": "L" if output_image.mode == "L" else "Y (luma)",
            "Mean in / out": f"{mean_in:.1f} / {mean_out:.1f}",
            "Std in / out": f"{std_in:.1f} / {std_out:.1f}",
        }

        return ToolResult(
            image=output_image,
            message="Applied global histogram equalization.",
            details=details,
        )


class ClaheTool(ForensicsTool):
    tool_id = "histogram_equalization_clahe"
    title = "Histogram equalization (CLAHE)"
    category = "Set 2: Image Processing"
    description = "Contrast Limited Adaptive Histogram Equalization: local, tile-based equalization."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by main window

        clip_limit = simpledialog.askfloat(
            "CLAHE Parameter",
            "Enter clip limit (higher = more contrast, 0.001–1.0):",
            initialvalue=0.01,
            minvalue=0.001,
            maxvalue=1.0,
            parent=parent,
        )
        if clip_limit is None:
            return None

        source = document.current
        output_image = _equalize_luminance(
            source, lambda arr: equalize_adapthist(arr, clip_limit=clip_limit, nbins=256)
        )

        mean_in, std_in = _intensity_stats(source)
        mean_out, std_out = _intensity_stats(output_image)
        tile_h, tile_w = source.height // 8, source.width // 8  # skimage default kernel_size
        details: dict[str, Any] = {
            "Operation": "CLAHE",
            "Library": "skimage.exposure.equalize_adapthist",
            "Clip limit": clip_limit,
            "Tile size": f"{tile_w} × {tile_h}",
            "Channel": "L" if output_image.mode == "L" else "Y (luma)",
            "Mean in / out": f"{mean_in:.1f} / {mean_out:.1f}",
            "Std in / out": f"{std_in:.1f} / {std_out:.1f}",
        }

        return ToolResult(
            image=output_image,
            message=f"Applied CLAHE with clip limit={clip_limit}.",
            details=details,
        )
