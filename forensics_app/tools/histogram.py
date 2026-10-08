"""Functionality 4: Histogram visualization for grayscale and RGB images."""

from __future__ import annotations

import tkinter as tk
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class HistogramTool(ForensicsTool):
    tool_id = "histogram"
    title = "Histogram visualization"
    category = "Set 2: Image Processing"
    description = "Compute and inspect the intensity distribution of the image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by main window

        # Convert working PIL image to a NumPy array
        img_arr = np.array(document.current)

        # 1. Create a popup window embedded with a Matplotlib figure
        popup = tk.Toplevel(parent)
        popup.title("Intensity Histogram")
        popup.geometry("640x440")

        fig, ax = plt.subplots(figsize=(6, 4))
        details: dict[str, Any] = {
            "Operation": "Histogram Visualization",
            "Mode": document.current.mode,
            "Dimensions": f"{img_arr.shape[1]}x{img_arr.shape[0]}",
        }

        # 2. Compute and draw histogram (grayscale vs. multichannel RGB)
        if img_arr.ndim == 2:
            # Grayscale: single histogram over [0, 256]
            hist, bin_edges = np.histogram(img_arr.ravel(), bins=256, range=(0, 256))
            ax.plot(bin_edges[:-1], hist, color="black", lw=1.2, label="Grayscale")
            details["Min Intensity"] = int(np.min(img_arr))
            details["Max Intensity"] = int(np.max(img_arr))
            details["Mean Intensity"] = f"{float(np.mean(img_arr)):.2f}"
        else:
            # Color RGB: overlay Red, Green, and Blue distribution curves
            colors = ("red", "green", "blue")
            for i, col in enumerate(colors):
                channel = img_arr[:, :, i]
                hist, bin_edges = np.histogram(channel.ravel(), bins=256, range=(0, 256))
                ax.plot(bin_edges[:-1], hist, color=col, alpha=0.8, lw=1.2, label=col.capitalize())
                details[f"{col.capitalize()} Mean"] = f"{float(np.mean(channel)):.2f}"

            ax.legend(loc="upper right")

        ax.set_title("Pixel Intensity Distribution")
        ax.set_xlabel("Pixel Intensity [0, 255]")
        ax.set_ylabel("Frequency (Pixel Count)")
        ax.set_xlim([0, 255])
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()

        # 3. Mount Matplotlib Canvas into Tkinter Toplevel window
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        return ToolResult(
            image=None,  # Inspection tool: canvas image remains unchanged
            message="Calculated and displayed the image intensity histogram.",
            details=details,
        )