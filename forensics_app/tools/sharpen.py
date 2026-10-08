"""Image sharpening with skimage.filters.unsharp_mask (configurable radius and amount)."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Any

import numpy as np
from PIL import Image
from skimage.filters import unsharp_mask

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .modes import split_alpha

RADIUS_RANGE = (0.1, 25.0)
AMOUNT_RANGE = (0.0, 10.0)


@dataclass(frozen=True)
class SharpenSettings:
    radius: float = 1.0
    amount: float = 1.0


def sharpen(image: Image.Image, settings: SharpenSettings) -> Image.Image:
    """Unsharp mask: ``out = img + amount * (img - gaussian(img, radius))``, per colour channel.

    Colour images stay in colour and alpha is carried over unchanged.
    """
    colour, alpha = split_alpha(image)
    arr = np.asarray(colour, dtype=np.float64) / 255.0
    sharpened = unsharp_mask(
        arr,
        radius=settings.radius,
        amount=settings.amount,
        # Positive axis on purpose: skimage's slice_at_axis mishandles channel_axis=-1
        # (it slices image rows instead of colour channels, blanking most of the image).
        channel_axis=2 if arr.ndim == 3 else None,
    )
    out = np.clip(np.round(sharpened * 255.0), 0, 255).astype(np.uint8)
    output = Image.fromarray(out, mode=colour.mode)
    if alpha is not None:
        output = output.convert("LA" if output.mode == "L" else "RGBA")
        output.putalpha(alpha)
    return output


class SharpenDialog(simpledialog.Dialog):
    """Ask for the unsharp mask radius and amount."""

    def body(self, master: tk.Frame) -> tk.Widget:
        self.radius = tk.StringVar(value="1.0")
        self.amount = tk.StringVar(value="1.0")

        ttk.Label(master, text="Radius (Gaussian sigma):").grid(row=0, column=0, sticky="w")
        radius_box = ttk.Spinbox(
            master, from_=RADIUS_RANGE[0], to=RADIUS_RANGE[1], increment=0.5, textvariable=self.radius, width=7
        )
        radius_box.grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(master, text="Size of the details that get sharpened.").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        ttk.Label(master, text="Amount:").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(
            master, from_=AMOUNT_RANGE[0], to=AMOUNT_RANGE[1], increment=0.5, textvariable=self.amount, width=7
        ).grid(row=2, column=1, sticky="w", padx=(6, 0))
        ttk.Label(master, text="How strongly edges are boosted (0 = no change).").grid(
            row=3, column=0, columnspan=2, sticky="w"
        )
        return radius_box

    def validate(self) -> bool:
        try:
            radius, amount = float(self.radius.get()), float(self.amount.get())
        except ValueError:
            messagebox.showerror("Sharpen", "Radius and amount must be numbers.", parent=self)
            return False
        if not RADIUS_RANGE[0] <= radius <= RADIUS_RANGE[1]:
            messagebox.showerror("Sharpen", f"Radius must be between {RADIUS_RANGE[0]} and {RADIUS_RANGE[1]:g}.", parent=self)
            return False
        if not AMOUNT_RANGE[0] <= amount <= AMOUNT_RANGE[1]:
            messagebox.showerror("Sharpen", f"Amount must be between {AMOUNT_RANGE[0]:g} and {AMOUNT_RANGE[1]:g}.", parent=self)
            return False
        self._settings = SharpenSettings(radius=radius, amount=amount)
        return True

    def apply(self) -> None:
        self.result = self._settings


class SharpenTool(ForensicsTool):
    tool_id = "sharpen_unsharp_mask"
    title = "Sharpen (unsharp mask)"
    category = "Set 3: Edge Detection"
    description = "Sharpen the image with skimage's unsharp mask, choosing radius and amount."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by main window

        settings = SharpenDialog(parent, title="Sharpen (unsharp mask)").result
        if settings is None:
            return None

        output = sharpen(document.current, settings)

        details: dict[str, Any] = {
            "Operation": "Image Sharpening",
            "Library": "skimage.filters.unsharp_mask",
            "Radius": settings.radius,
            "Amount": settings.amount,
            "Channels": "Luminance" if output.mode in ("L", "LA") else "R, G, B (each)",
        }
        if output.mode in ("LA", "RGBA"):
            details["Alpha"] = "Preserved"

        return ToolResult(
            image=output,
            message=f"Sharpened image with unsharp mask (radius={settings.radius}, amount={settings.amount}).",
            details=details,
        )
