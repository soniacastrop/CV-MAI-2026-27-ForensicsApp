"""Rearrange the colour channels of the working image."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog, ttk

from PIL import Image, ImageChops

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

COLOUR_BANDS = ("R", "G", "B")


def swap_channels(image: Image.Image, order: tuple[str, str, str]) -> Image.Image:
    """Build a new image whose R, G, B channels are taken from ``order``.

    ``order=("B", "G", "R")`` swaps red and blue. Alpha, if present, is kept.
    """
    if any(band not in COLOUR_BANDS for band in order):
        raise ValueError(f"Channels must be chosen from {', '.join(COLOUR_BANDS)}.")
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    source = image.convert("RGBA" if has_alpha else "RGB")
    bands = dict(zip(source.getbands(), source.split()))
    output = [bands[name] for name in order]  #coge los split channels en el orden de order
    if has_alpha:
        return Image.merge("RGBA", (*output, bands["A"])) #los mergea
    return Image.merge("RGB", output)


def has_identical_colour_channels(image: Image.Image) -> bool:
    """True when R, G and B are the same everywhere, so any swap changes nothing.

    Covers grayscale modes (``L``, ``LA``, ``1``...) and RGB images that only look gray.
    """
    red, green, blue = image.convert("RGB").split()
    return ImageChops.difference(red, green).getbbox() is None and ImageChops.difference(green, blue).getbbox() is None


class ChannelOrderDialog(simpledialog.Dialog):
    """Ask which source channel goes into each output channel."""

    def body(self, master: tk.Frame) -> tk.Widget:
        ttk.Label(master, text="Choose the source channel for each output channel:").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        self.choices: list[tk.StringVar] = []
        boxes = []
        for row, output_band in enumerate(COLOUR_BANDS, start=1):
            ttk.Label(master, text=f"Output {output_band} ←").grid(row=row, column=0, sticky="e", padx=(0, 6), pady=2)
            choice = tk.StringVar(value=output_band)
            box = ttk.Combobox(master, textvariable=choice, values=COLOUR_BANDS, state="readonly", width=6)
            box.grid(row=row, column=1, sticky="w", pady=2)
            self.choices.append(choice)
            boxes.append(box)
        return boxes[0]

    def apply(self) -> None: #called when user accepts the dialog
        self.result = tuple(choice.get() for choice in self.choices) #choice.get gets the current selection of that variable in the box


class ChannelSwapTool(ForensicsTool):
    tool_id = "channel_swap"
    title = "Swap image channels"
    category = "Set2"
    description = "Rearrange the R, G and B channels of the image (e.g. swap red and blue)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window
        if has_identical_colour_channels(document.current): #example when a grayscale image is used as input
            # Checked before the dialog: a swap would add an undo step without changing any pixel.
            return ToolResult(
                message="Nothing to swap: the image is grayscale, so R, G and B are identical.",
                details={
                    "Operation": "Channel swap",
                    "Mode": document.current.mode,
                    "Result": "Skipped (grayscale image)",
                },
            )
        order = ChannelOrderDialog(parent, title="Swap channels").result  #get the order from ui
        if order is None:
            return None  # user cancelled
        output = swap_channels(document.current, order)
        return ToolResult(
            image=output,
            message=f"Swapped channels: R←{order[0]}, G←{order[1]}, B←{order[2]}.",
            details={
                "Operation": "Channel swap",
                "Output R": order[0],
                "Output G": order[1],
                "Output B": order[2],
                "Output mode": output.mode,
            },
        )
