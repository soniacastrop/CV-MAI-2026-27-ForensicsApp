"""Functionality 1: 2D convolution with an arbitrary, user-entered kernel."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Any

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .modes import split_alpha

# How results outside [0, 255] are mapped back to 8-bit (key -> label in the dialog)
OUTPUT_MODES = {
    "clip": "Clip to 0–255",
    "abs": "Absolute value, then clip (keeps negative responses)",
    "rescale": "Rescale min–max to 0–255",
}
GRID_CELL_LIMIT = 12  # larger kernels get a scrollable grid


@dataclass(frozen=True)
class KernelSettings:
    kernel: np.ndarray
    normalize: bool = False
    output: str = "clip"


def max_kernel_size(image: Image.Image) -> int:
    """Largest allowed kernel side: n - 1, with n the image's smaller side."""
    return min(image.width, image.height) - 1


def convolve_channel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """True 2D convolution (kernel flipped), same size as ``channel``, mirrored borders."""
    pad = max(kernel.shape)
    padded = np.pad(channel.astype(np.float64), pad, mode="symmetric")
    return fftconvolve(padded, kernel, mode="same")[pad:-pad, pad:-pad]


def to_uint8(values: np.ndarray, output: str) -> np.ndarray:
    if output == "abs":
        values = np.abs(values)
    elif output == "rescale":
        low, high = values.min(), values.max()
        values = (values - low) / (high - low) * 255.0 if high > low else np.zeros_like(values)
    elif output != "clip":
        raise ValueError(f"Output must be one of {', '.join(OUTPUT_MODES)}.")
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def convolve_image(image: Image.Image, settings: KernelSettings) -> tuple[Image.Image, float, float]:
    """Convolve every colour channel; alpha is kept unchanged. Returns (image, raw min, raw max)."""
    kernel = np.asarray(settings.kernel, dtype=np.float64)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or kernel.size == 0:
        raise ValueError("The kernel must be a non-empty square matrix.")
    if not 1 <= kernel.shape[0] <= max_kernel_size(image):
        raise ValueError(f"Kernel size must be between 1 and {max_kernel_size(image)} for this image.")
    if settings.normalize:
        total = kernel.sum()
        if np.isclose(total, 0.0):
            raise ValueError("Cannot normalize: the kernel values sum to 0.")
        kernel = kernel / total

    colour, alpha = split_alpha(image)
    arr = np.asarray(colour, dtype=np.float64)
    if arr.ndim == 2:
        raw = convolve_channel(arr, kernel)
    else:
        raw = np.stack([convolve_channel(arr[:, :, c], kernel) for c in range(arr.shape[2])], axis=-1)

    output = Image.fromarray(to_uint8(raw, settings.output), mode=colour.mode)
    if alpha is not None:
        output = output.convert("LA" if output.mode == "L" else "RGBA")
        output.putalpha(alpha)
    return output, float(raw.min()), float(raw.max())


def format_kernel(kernel: np.ndarray) -> str:
    return "; ".join(" ".join(f"{value:g}" for value in row) for row in kernel)


class KernelDialog(simpledialog.Dialog):
    """Grid of k x k entries for the kernel values."""

    def __init__(self, parent: tk.Misc, size: int, title: str | None = None) -> None:
        self.size = size
        super().__init__(parent, title=title)

    def body(self, master: tk.Frame) -> tk.Widget:
        ttk.Label(master, text=f"Enter the {self.size} × {self.size} = {self.size ** 2} kernel values:").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        grid_parent = self._grid_container(master)
        centre = self.size // 2
        self.cells: list[list[tk.StringVar]] = []
        first = None
        for r in range(self.size):
            row = []
            for c in range(self.size):
                value = tk.StringVar(value="1" if (r, c) == (centre, centre) else "0")  # identity kernel
                entry = ttk.Entry(grid_parent, textvariable=value, width=6, justify="center")
                entry.grid(row=r, column=c, padx=1, pady=1)
                first = first or entry
                row.append(value)
            self.cells.append(row)

        options = ttk.Frame(master)
        options.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.fill_value = tk.StringVar(value="1")
        ttk.Button(options, text="Fill all with", command=self._fill).grid(row=0, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.fill_value, width=6).grid(row=0, column=1, padx=(4, 0), sticky="w")
        self.normalize = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text="Divide by the sum of the values (e.g. for blur kernels)", variable=self.normalize).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        ttk.Label(options, text="Result outside 0–255:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.output = tk.StringVar(value=OUTPUT_MODES["clip"])
        ttk.Combobox(
            options, textvariable=self.output, values=list(OUTPUT_MODES.values()), state="readonly", width=46
        ).grid(row=3, column=0, columnspan=2, sticky="w")
        return first

    def _grid_container(self, master: tk.Frame) -> tk.Misc:
        """Plain frame for small kernels; scrollable canvas for large ones."""
        if self.size <= GRID_CELL_LIMIT:
            frame = ttk.Frame(master)
            frame.grid(row=1, column=0, sticky="w")
            return frame
        outer = ttk.Frame(master)
        outer.grid(row=1, column=0, sticky="nsew")
        canvas = tk.Canvas(outer, width=560, height=320, highlightthickness=0)
        x_bar = ttk.Scrollbar(outer, orient="horizontal", command=canvas.xview)
        y_bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=x_bar.set, yscrollcommand=y_bar.set)
        canvas.grid(row=0, column=0)
        y_bar.grid(row=0, column=1, sticky="ns")
        x_bar.grid(row=1, column=0, sticky="ew")
        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        return inner

    def _fill(self) -> None:
        for row in self.cells:
            for value in row:
                value.set(self.fill_value.get())

    def validate(self) -> bool:
        try:
            kernel = np.array([[float(value.get()) for value in row] for row in self.cells], dtype=np.float64)
        except ValueError:
            messagebox.showerror("Convolution", "Every kernel value must be a number.", parent=self)
            return False
        if self.normalize.get() and np.isclose(kernel.sum(), 0.0):
            messagebox.showerror("Convolution", "Cannot divide by the sum: the values sum to 0.", parent=self)
            return False
        output = next(key for key, label in OUTPUT_MODES.items() if label == self.output.get())
        self._settings = KernelSettings(kernel=kernel, normalize=self.normalize.get(), output=output)
        return True

    def apply(self) -> None:
        self.result = self._settings


class ConvolutionTool(ForensicsTool):
    tool_id = "convolution"
    title = "Convolution (arbitrary kernel)"
    category = "Set 3: Edge Detection"
    description = "Convolve the image with a square kernel whose size and values you enter."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # 1. Ask the kernel size: 1 .. n-1, n = smaller image side
        largest = max_kernel_size(document.current)
        if largest < 1:
            raise ValueError("The image is too small to convolve (it needs at least 2 × 2 pixels).")
        size = simpledialog.askinteger(
            "Kernel size",
            f"Kernel size k (k × k values), from 1 to {largest}:",
            initialvalue=min(3, largest),
            minvalue=1,
            maxvalue=largest,
            parent=parent,
        )
        if size is None:
            return None

        # 2. Ask the k x k kernel values
        settings = KernelDialog(parent, size, title=f"{size} × {size} kernel").result
        if settings is None:
            return None

        # 3. Convolve each channel
        output_img, raw_min, raw_max = convolve_image(document.current, settings)

        details: dict[str, Any] = {
            "Operation": "2D Convolution",
            "Kernel size": f"{size} × {size}",
            "Kernel": format_kernel(settings.kernel) if size <= 5 else f"{size * size} values",
            "Kernel sum": f"{settings.kernel.sum():g}",
            "Normalized": "Yes" if settings.normalize else "No",
            "Output mapping": OUTPUT_MODES[settings.output],
            "Raw range": f"[{raw_min:.1f}, {raw_max:.1f}]",
        }

        return ToolResult(
            image=output_img,
            message=f"Convolved the image with a {size} × {size} kernel.",
            details=details,
        )
