"""Functionality 2: Standard filters from skimage.filters with configurable sigma."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog
from typing import Any

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class SkimageFiltersTool(ForensicsTool):
    tool_id = "skimage_filters"
    title = "Filters (skimage.filters)"
    category = "Set 3: Edge Detection"
    description = "Apply Sobel, Prewitt, or configurable Gaussian filters from skimage."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # 1. Ask the user which filter they want to apply
        filter_choice = simpledialog.askstring(
            "Filter Selection",
            "Choose a filter:\n1. sobel\n2. prewitt\n3. gaussian",
            initialvalue="gaussian",
            parent=parent,
        )
        if not filter_choice:
            return None

        filter_choice = filter_choice.strip().lower()

        # 2. Convert active image to float representation in [0, 1]
        gray = document.current.convert("L")
        img_arr = np.asarray(gray, dtype=np.float64) / 255.0

        details: dict[str, Any] = {
            "Library": "skimage.filters",
            "Selection": filter_choice,
        }

        # 3. Apply the selected filter
        if filter_choice in ("1", "sobel"):
            filtered = filters.sobel(img_arr)
            details["Algorithm"] = "Sobel edge detector"

        elif filter_choice in ("2", "prewitt"):
            filtered = filters.prewitt(img_arr)
            details["Algorithm"] = "Prewitt edge detector"

        elif filter_choice in ("3", "gaussian"):
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

        else:
            return ToolResult(
                image=None,
                message=f"Unknown filter option: '{filter_choice}'. Choose sobel, prewitt, or gaussian.",
                details={"Error": "Invalid choice"},
            )

        # 4. Normalize back to 8-bit [0, 255]
        norm = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered) + 1e-8)
        output_arr = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)
        output_img = Image.fromarray(output_arr, mode="L")

        return ToolResult(
            image=output_img,
            message=f"Successfully applied {details.get('Algorithm', filter_choice)}.",
            details=details,
        )