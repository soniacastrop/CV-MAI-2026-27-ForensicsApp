"""Functionality 2: Standard filters from skimage.filters with configurable parameters."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog, ttk
from typing import Any

import numpy as np
from PIL import Image
from skimage import filters
from skimage.morphology import disk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

# Filter key -> label shown in the selection dialog.
FILTERS = {
    "sobel": "Sobel (edges)",
    "prewitt": "Prewitt (edges)",
    "scharr": "Scharr (edges)",
    "gaussian": "Gaussian blur",
    "median": "Median (noise removal)",
    "threshold_otsu": "Otsu threshold (binary)",
}


class FilterDialog(simpledialog.Dialog):
    """Let the user pick one of the available filters."""

    def body(self, master: tk.Frame) -> tk.Widget:
        self.choice = tk.StringVar(value="gaussian")
        ttk.Label(master, text="Filter:").grid(row=0, column=0, sticky="w", pady=(0, 4))
        first: tk.Widget | None = None
        for row, (key, label) in enumerate(FILTERS.items(), start=1):
            button = ttk.Radiobutton(master, text=label, value=key, variable=self.choice)
            button.grid(row=row, column=0, sticky="w")
            first = first or button
        assert first is not None
        return first

    def apply(self) -> None:
        self.result = self.choice.get()


class SkimageFiltersTool(ForensicsTool):
    tool_id = "skimage_filters"
    title = "Filters (skimage.filters)"
    category = "Set 3: Edge Detection"
    description = "Apply Sobel, Prewitt, Scharr, Gaussian, median or Otsu threshold from skimage."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # 1. Ask the user which filter they want to apply
        filter_choice = FilterDialog(parent, title="Filter Selection").result
        if not filter_choice:
            return None

        # 2. Convert active image to float representation in [0, 1]
        gray = document.current.convert("L")
        img_arr = np.asarray(gray, dtype=np.float64) / 255.0

        details: dict[str, Any] = {
            "Library": "skimage.filters",
            "Selection": filter_choice,
        }
        # Edge filters produce arbitrary magnitudes and are min-max normalized for display;
        # filters that return intensities in [0, 1] (median, Otsu) keep their scale.
        normalize = True

        # 3. Apply the selected filter
        if filter_choice == "sobel":
            filtered = filters.sobel(img_arr)
            details["Algorithm"] = "Sobel edge detector"

        elif filter_choice == "prewitt":
            filtered = filters.prewitt(img_arr)
            details["Algorithm"] = "Prewitt edge detector"

        elif filter_choice == "gaussian":
            # Configurable parameter: sigma
            sigma_val = simpledialog.askfloat(
                "Gaussian Parameter",
                "Enter standard deviation (sigma):",
                initialvalue=2.0,
                minvalue=0.1,
                maxvalue=25.0,
                parent=parent,
            )
            if sigma_val is None:
                return None

            filtered = filters.gaussian(img_arr, sigma=sigma_val)
            details["Algorithm"] = "Gaussian blur filter"
            details["Configurable Sigma"] = sigma_val

        elif filter_choice == "median":
            radius = simpledialog.askinteger(
                "Median Parameter",
                "Enter neighbourhood radius in pixels (disk footprint):",
                initialvalue=2,
                minvalue=1,
                maxvalue=50,
                parent=parent,
            )
            if radius is None:
                return None

            filtered = filters.median(np.asarray(gray), footprint=disk(radius)) / 255.0
            normalize = False
            details["Algorithm"] = "Median filter"
            details["Radius"] = radius

        elif filter_choice == "scharr":
            filtered = filters.scharr(img_arr)
            details["Algorithm"] = "Scharr edge detector"

        elif filter_choice == "threshold_otsu":
            threshold = float(filters.threshold_otsu(img_arr))
            filtered = (img_arr > threshold).astype(np.float64)
            normalize = False
            details["Algorithm"] = "Otsu threshold (binary)"
            details["Otsu threshold"] = f"{threshold * 255.0:.1f} / 255"
            details["Pixels above"] = f"{float(filtered.mean()) * 100.0:.2f}%"

        else:
            return ToolResult(
                image=None,
                message=(
                    f"Unknown filter option: '{filter_choice}'. Choose one of: {', '.join(FILTERS)}."
                ),
                details={"Error": "Invalid choice"},
            )

        # 4. Normalize back to 8-bit [0, 255]
        if normalize:
            norm = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered) + 1e-8)
        else:
            norm = filtered
        output_arr = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)
        output_img = Image.fromarray(output_arr, mode="L")

        return ToolResult(
            image=output_img,
            message=f"Successfully applied {details.get('Algorithm', filter_choice)}.",
            details=details,
        )