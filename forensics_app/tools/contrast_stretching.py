"""Functionality 5: Linear and percentile-based contrast stretching."""

from __future__ import annotations

import tkinter as tk
from typing import Any

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .modes import split_alpha


class ContrastStretchingTool(ForensicsTool):
    tool_id = "contrast_stretching"
    title = "Contrast stretching"
    category = "Set 2: Image Processing"
    description = "Enhance dynamic range using linear percentile contrast stretching."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by main window

        colour, alpha = split_alpha(document.current)
        img_arr = np.array(colour)
        details: dict[str, Any] = {
            "Operation": "Contrast Stretching",
            "Mode": document.current.mode,
        }

        # Handle Grayscale (2D) vs RGB (3D)
        if img_arr.ndim == 2:
            stretched_arr, p_low, p_high = self._stretch_channel(img_arr)
            details["In Band [p2, p98]"] = f"[{p_low:.1f}, {p_high:.1f}]"
            details["Out Band"] = "[0, 255]"
        else:
            stretched_channels = []
            bands = []
            for c in range(3):
                stretched_ch, p_low, p_high = self._stretch_channel(img_arr[:, :, c])
                stretched_channels.append(stretched_ch)
                bands.append(f"[{p_low:.0f}, {p_high:.0f}]")

            stretched_arr = np.stack(stretched_channels, axis=-1)
            details["R [p2, p98]"] = bands[0]
            details["G [p2, p98]"] = bands[1]
            details["B [p2, p98]"] = bands[2]

        output_image = Image.fromarray(stretched_arr)
        if alpha is not None:
            # Transparency is not intensity: carry it over unchanged
            output_image = output_image.convert("LA" if output_image.mode == "L" else "RGBA")
            output_image.putalpha(alpha)
            details["Alpha"] = "Preserved (not stretched)"

        return ToolResult(
            image=output_image,
            message="Enhanced image dynamic range using 2nd-98th percentile contrast stretching.",
            details=details,
        )

    @staticmethod
    def _stretch_channel(channel: np.ndarray, low_p: float = 2.0, high_p: float = 98.0) -> tuple[np.ndarray, float, float]:
        """Apply linear stretching to a 2D intensity array using percentiles."""
        ch_float = channel.astype(np.float64)
        p_low, p_high = np.percentile(ch_float, (low_p, high_p))

        if p_high > p_low:
            # Formula: BV_out = ((BV_in - min) / (max - min)) * 255
            scaled = (ch_float - p_low) / (p_high - p_low) * 255.0
            stretched = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
        else:
            stretched = channel.copy()

        return stretched, float(p_low), float(p_high)