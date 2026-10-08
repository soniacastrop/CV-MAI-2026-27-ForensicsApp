"""Functionality 2: Standard filters from skimage.filters with configurable parameters."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog
from typing import Any

import numpy as np
from PIL import Image
from skimage import filters
from skimage.morphology import disk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class SkimageFiltersTool(ForensicsTool):
    tool_id = "skimage_filters"
    title = "Filters (skimage.filters)"
    category = "Set 3: Edge Detection"
    description = "Apply Sobel, Prewitt, Scharr, Gaussian, median, unsharp mask or Otsu threshold from skimage."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # 1. Ask the user which filter they want to apply
        filter_choice = simpledialog.askstring(
            "Filter Selection",
            "Choose a filter (name or number):\n"
            "1. sobel\n2. prewitt\n3. gaussian\n4. median\n5. scharr\n6. unsharp_mask\n7. threshold_otsu",
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
        # Edge filters produce arbitrary magnitudes and are min-max normalized for display;
        # filters that return intensities in [0, 1] (median, unsharp, Otsu) keep their scale.
        normalize = True

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

        elif filter_choice in ("4", "median"):
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

        elif filter_choice in ("5", "scharr"):
            filtered = filters.scharr(img_arr)
            details["Algorithm"] = "Scharr edge detector"

        elif filter_choice in ("6", "unsharp_mask", "unsharp mask", "unsharp"):
            radius_val = simpledialog.askfloat(
                "Unsharp Mask Parameter",
                "Enter blur radius (Gaussian sigma) of the mask:",
                initialvalue=1.0,
                minvalue=0.1,
                maxvalue=25.0,
                parent=parent,
            )
            if radius_val is None:
                return None
            amount = simpledialog.askfloat(
                "Unsharp Mask Parameter",
                "Enter amount (how strongly edges are boosted):",
                initialvalue=1.0,
                minvalue=0.0,
                maxvalue=10.0,
                parent=parent,
            )
            if amount is None:
                return None

            filtered = filters.unsharp_mask(img_arr, radius=radius_val, amount=amount)
            normalize = False
            details["Algorithm"] = "Unsharp mask (sharpening)"
            details["Radius"] = radius_val
            details["Amount"] = amount

        elif filter_choice in ("7", "threshold_otsu", "otsu", "threshold otsu"):
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
                    f"Unknown filter option: '{filter_choice}'. Choose sobel, prewitt, gaussian, "
                    "median, scharr, unsharp_mask or threshold_otsu."
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