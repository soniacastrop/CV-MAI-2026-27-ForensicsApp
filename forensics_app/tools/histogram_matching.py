"""Functionality 7: Histogram matching (specification) against a user-selected reference image."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any

import numpy as np
from PIL import Image
from skimage.exposure import match_histograms

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

REFERENCE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]
GRAY_MODES = ("L", "I", "F", "1")


def _to_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def match_grayscale(source: Image.Image, reference: Image.Image) -> Image.Image:
    src = np.asarray(source.convert("L"))
    ref = np.asarray(reference.convert("L"))
    return Image.fromarray(_to_uint8(match_histograms(src, ref)), mode="L")


def match_per_channel(source: Image.Image, reference: Image.Image) -> Image.Image:
    """Match R, G and B independently: transfers the reference's colour tone as well as contrast."""
    src = np.asarray(source.convert("RGB"))
    ref = np.asarray(reference.convert("RGB"))
    return Image.fromarray(_to_uint8(match_histograms(src, ref, channel_axis=-1)), mode="RGB")


def match_luminance(source: Image.Image, reference: Image.Image) -> Image.Image:
    """Match only the Y (luma) channel, keeping the source's own colours (Cb, Cr)."""
    y, cb, cr = source.convert("RGB").convert("YCbCr").split()
    ref_y = reference.convert("RGB").convert("YCbCr").getchannel("Y")
    y_matched = Image.fromarray(_to_uint8(match_histograms(np.asarray(y), np.asarray(ref_y))), mode="L")
    return Image.merge("YCbCr", (y_matched, cb, cr)).convert("RGB")


class HistogramMatchingTool(ForensicsTool):
    tool_id = "histogram_matching"
    title = "Histogram matching"
    category = "Set 2: Image Processing"
    description = "Reshape the image's histogram to match that of another selected reference image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by main window

        filename = filedialog.askopenfilename(
            title="Select reference image", filetypes=REFERENCE_TYPES, parent=parent
        )
        if not filename:
            return None
        with Image.open(filename) as opened:
            reference = opened.copy()

        source = document.current
        if source.mode in GRAY_MODES:
            output_image = match_grayscale(source, reference)
            method = "Grayscale"
        else:
            per_channel = messagebox.askyesnocancel(
                "Histogram matching",
                "Match each colour channel (R, G, B) separately?\n\n"
                "Yes: also copies the reference's colour tone.\n"
                "No: match brightness (luma) only and keep the original colours.",
                parent=parent,
            )
            if per_channel is None:
                return None
            if per_channel:
                output_image = match_per_channel(source, reference)
                method = "Per channel (R, G, B)"
            else:
                output_image = match_luminance(source, reference)
                method = "Luminance only (Y)"

        def mean_std(image: Image.Image) -> str:
            arr = np.asarray(image.convert("L"), dtype=np.float64)
            return f"{arr.mean():.1f} / {arr.std():.1f}"

        details: dict[str, Any] = {
            "Operation": "Histogram Matching",
            "Library": "skimage.exposure.match_histograms",
            "Reference": Path(filename).name,
            "Reference size": f"{reference.width} × {reference.height}",
            "Method": method,
            "Mean/std source": mean_std(source),
            "Mean/std reference": mean_std(reference),
            "Mean/std result": mean_std(output_image),
        }

        return ToolResult(
            image=output_image,
            message=f"Matched histogram to {Path(filename).name} ({method.lower()}).",
            details=details,
        )
