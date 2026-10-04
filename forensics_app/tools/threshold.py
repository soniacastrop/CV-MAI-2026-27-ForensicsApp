"""Keep the pixels whose channel value lies in a range and black out the rest."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageChops

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .mask import MaskPreviewDialog, apply_mask, selected_fraction, selected_pixels_detail

CHANNELS = ("Luminance", "R", "G", "B")


@dataclass(frozen=True)
class ThresholdSettings:
    channel: str = "Luminance"
    low: int = 128
    high: int = 255
    invert: bool = False


def threshold_mask(image: Image.Image, settings: ThresholdSettings) -> Image.Image:
    """Return an ``L`` mask: 255 where ``low <= channel value <= high``, else 0 (swapped if inverted).

    ``low=T, high=255`` selects values above T; ``low=0, high=T`` selects values below.
    """
    if settings.channel not in CHANNELS:
        raise ValueError(f"Channel must be one of {', '.join(CHANNELS)}.")
    if not 0 <= settings.low <= settings.high <= 255:
        raise ValueError("Thresholds must satisfy 0 ≤ from ≤ to ≤ 255.")
    if settings.channel == "Luminance":
        band = image.convert("L")
    else:
        band = image.convert("RGB").getchannel(settings.channel)
    mask = band.point([255 if settings.low <= value <= settings.high else 0 for value in range(256)])
    return ImageChops.invert(mask) if settings.invert else mask


class ThresholdDialog(MaskPreviewDialog):
    """Ask for the channel and value range to keep."""

    def build_controls(self, master: ttk.Frame) -> tk.Widget:
        self.channel = tk.StringVar(value=CHANNELS[0])
        self.low = tk.StringVar(value="128")
        self.high = tk.StringVar(value="255")
        self.invert = tk.BooleanVar(value=False)

        ttk.Label(master, text="Channel:").grid(row=0, column=0, sticky="e", padx=(0, 6))
        channel_box = ttk.Combobox(master, textvariable=self.channel, values=CHANNELS, state="readonly", width=10)
        channel_box.grid(row=0, column=1, columnspan=3, sticky="w")
        ttk.Label(master, text="Keep values from").grid(row=1, column=0, sticky="e", padx=(0, 6), pady=4)
        ttk.Spinbox(master, from_=0, to=255, textvariable=self.low, width=6).grid(row=1, column=1, sticky="w")
        ttk.Label(master, text="to").grid(row=1, column=2, padx=6)
        ttk.Spinbox(master, from_=0, to=255, textvariable=self.high, width=6).grid(row=1, column=3, sticky="w")
        ttk.Checkbutton(master, text="Invert (keep values outside the range)", variable=self.invert).grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(4, 8)
        )
        ttk.Label(master, text="Pixels that are not kept become black.").grid(
            row=3, column=0, columnspan=4, sticky="w"
        )
        self.watch(self.channel, self.low, self.high, self.invert)
        return channel_box

    def read_result(self) -> ThresholdSettings:
        try:
            low, high = int(self.low.get()), int(self.high.get())
        except ValueError:
            raise ValueError("Thresholds must be whole numbers.") from None
        if not 0 <= low <= high <= 255:
            raise ValueError("Thresholds must satisfy 0 ≤ from ≤ to ≤ 255.")
        return ThresholdSettings(channel=self.channel.get(), low=low, high=high, invert=self.invert.get())

    def preview_mask(self, image: Image.Image) -> Image.Image:
        return threshold_mask(image, self.read_result())


class ThresholdTool(ForensicsTool):
    tool_id = "threshold"
    title = "Threshold image"
    category = "Set2"
    description = "Keep pixels whose channel value is in a range (or outside it) and black out the rest."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window
        image = document.current
        settings = ThresholdDialog(parent, image, title="Threshold image").result
        if settings is None:
            return None  # user cancelled

        mask = threshold_mask(image, settings)
        output = apply_mask(image, mask)
        selection = f"{settings.channel} in [{settings.low}, {settings.high}]"
        if settings.invert:
            selection = f"NOT ({selection})"
        return ToolResult(
            image=output,
            message=f"Thresholded image: {selected_fraction(mask):.1%} of pixels kept ({selection}).",
            details={
                "Operation": "Threshold",
                "Selection": selection,
                "Kept pixels": selected_pixels_detail(mask),
                "Output mode": output.mode,
            },
        )
