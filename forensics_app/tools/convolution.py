"""Functionality 1: 2D Convolution with arbitrary kernel and direction."""

from __future__ import annotations

import tkinter as tk
from typing import Any

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class ConvolutionTool(ForensicsTool):
    tool_id = "convolution"
    title = "Convolution (arbitrary kernel)"
    category = "Set 3: Edge Detection"
    description = "Convolve the image using standard directional differential kernels."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None

        # 1. Convert to working grayscale array (float in [0, 1])
        gray = document.current.convert("L")
        img_arr = np.asarray(gray, dtype=np.float64) / 255.0

        # 2. Directional derivative kernels (Horizontal dx and Vertical dy)
        k_horizontal = np.array([[-1.0, 0.0, 1.0]], dtype=np.float64)
        k_vertical = np.array([[-1.0], [0.0], [1.0]], dtype=np.float64)

        # 3. Apply 2D convolution
        conv_h = convolve2d(img_arr, k_horizontal, mode="same", boundary="symm")
        conv_v = convolve2d(img_arr, k_vertical, mode="same", boundary="symm")

        # Combined gradient magnitude
        magnitude = np.hypot(conv_h, conv_v)
        magnitude = np.clip(magnitude * 255.0, 0.0, 255.0).astype(np.uint8)

        output_img = Image.fromarray(magnitude, mode="L")

        details: dict[str, Any] = {
            "Operation": "2D Directional Convolution",
            "Horizontal Kernel": "[-1, 0, 1]",
            "Vertical Kernel": "[-1, 0, 1]^T",
            "Max Gradient": f"{float(np.max(magnitude)):.2f}",
        }

        return ToolResult(
            image=output_img,
            message="Applied horizontal and vertical directional convolution.",
            details=details,
        )