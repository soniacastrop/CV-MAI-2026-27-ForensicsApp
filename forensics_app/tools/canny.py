"""Functionality 3: Canny edge detector from skimage.feature with configurable sigma."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog
from typing import Any

import numpy as np
from PIL import Image
from skimage.feature import canny

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class CannyTool(ForensicsTool):
    tool_id = "canny"
    title = "Canny edge detection"
    category = "Set 3: Edge Detection"
    description = "Compute thin, continuous edges using Canny detection with configurable sigma."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # 1. User dialog for configurable sigma
        sigma_val = simpledialog.askfloat(
            "Canny Parameter",
            "Enter Gaussian smoothing standard deviation (sigma):",
            initialvalue=1.5,
            minvalue=0.1,
            maxvalue=10.0,
            parent=parent,
        )
        if sigma_val is None:
            return None

        # 2. Convert active image to 2D grayscale float [0, 1]
        gray = document.current.convert("L")
        img_arr = np.asarray(gray, dtype=np.float64) / 255.0

        # 3. Apply skimage.feature.canny
        edges_bool = canny(img_arr, sigma=sigma_val)

        # 4. Convert boolean edge map to uint8 [0, 255]
        edges_uint8 = (edges_bool * 255).astype(np.uint8)
        output_img = Image.fromarray(edges_uint8, mode="L")

        details: dict[str, Any] = {
            "Operation": "Canny Edge Detection",
            "Library": "skimage.feature.canny",
            "Configured Sigma": sigma_val,
            "Edge Pixels": int(np.sum(edges_bool)),
            "Edge Density": f"{(float(np.mean(edges_bool)) * 100.0):.2f}%",
        }

        return ToolResult(
            image=output_img,
            message=f"Detected edges using Canny with sigma={sigma_val}.",
            details=details,
        )