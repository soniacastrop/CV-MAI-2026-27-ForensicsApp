"""Set 4.2: Template matching configs (rotation angle, threshold, and Set 3 superimposition)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .canny import DISPLAYS as CANNY_DISPLAYS, render_edges
from .hog import show_figure
from .ncc import TEMPLATE_TYPES, compute_ncc, ncc_to_image, rotate_template, to_gray_float

ANGLE_RANGE = (-360.0, 360.0)
THRESHOLD_RANGE = (-1.0, 1.0)

# Task 4.2 display modes (includes Set 3 superimpositions + NCC map + 4-panel popup)
CONFIG_DISPLAYS = {
    "red_edges": "Red match pixels on the image",
    "red_overlay": "Transparent red overlay on matches",
    "darken": "Darken everything except matches (as in slide)",
    "contours": "Matches as contours (one colour per match region)",
    "edge_map": "Binary threshold mask (white matches on black)",
    "ncc": "Normalized cross-correlation image",
    "popup": "4-panel comparison: original, template, NCC & thresholded (popup)",
}


@dataclass(frozen=True)
class TemplateMatchingSettings:
    angle: float = 0.0
    threshold: float = 0.7
    pad_input: bool = True
    display: str = "darken"


def threshold_ncc(corr: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of locations where the normalized cross-correlation exceeds ``threshold``."""
    if not THRESHOLD_RANGE[0] <= threshold <= THRESHOLD_RANGE[1]:
        raise ValueError(f"Threshold must be between {THRESHOLD_RANGE[0]:g} and {THRESHOLD_RANGE[1]:g}.")
    return corr >= threshold


def render_matches(image: Image.Image, corr: np.ndarray, threshold: float, display: str) -> Image.Image:
    """Render the template matching result using ``ncc`` or one of the Set 3 superimposition styles."""
    if display == "ncc":
        return ncc_to_image(corr)
    if display not in CANNY_DISPLAYS:
        raise ValueError(f"Display must be one of {', '.join(CONFIG_DISPLAYS)}.")
    matches = threshold_ncc(corr, threshold)
    if matches.shape != (image.height, image.width):
        # When pad_input=False, pad the boolean mask back to image size so superimposition aligns with the image
        padded = np.zeros((image.height, image.width), dtype=bool)
        top = (image.height - matches.shape[0]) // 2
        left = (image.width - matches.shape[1]) // 2
        padded[top : top + matches.shape[0], left : left + matches.shape[1]] = matches
        matches = padded
    return render_edges(image, matches, display)


def config_comparison_figure(
    image: Image.Image,
    tmpl_gray: np.ndarray,
    corr: np.ndarray,
    settings: TemplateMatchingSettings,
) -> Figure:
    """Build the 4-panel figure from Set 4.1/4.2: Original, Template, NCC, and Thresholded image."""
    figure = Figure(figsize=(14, 4.5), layout="constrained")
    ax = figure.subplots(1, 4)
    ax[0].imshow(to_gray_float(image), cmap="gray")
    ax[0].set_title("Original image")
    ax[1].imshow(tmpl_gray, cmap="gray")
    title_tmpl = "Template" if settings.angle == 0.0 else f"Template (angle={settings.angle:g}°)"
    ax[1].set_title(title_tmpl)
    ax[2].imshow(corr, cmap="gray")
    ax[2].set_title("Normalized cross-corr image")
    overlay_style = settings.display if settings.display in CANNY_DISPLAYS else "darken"
    thresholded = render_matches(image, corr, settings.threshold, overlay_style)
    if thresholded.mode == "L":
        ax[3].imshow(np.asarray(thresholded), cmap="gray")
    else:
        ax[3].imshow(np.asarray(thresholded))
    ax[3].set_title(f"Thresholded image (≥ {settings.threshold:g})")
    return figure


class TemplateMatchingDialog(simpledialog.Dialog):
    """Dialog for Set 4.2: Template matching configs (angle, threshold, Set 3 superimposition)."""

    def __init__(self, parent: tk.Misc, title: str, template: Image.Image, max_corr: float | None = None) -> None:
        self.template = template
        self.max_corr = max_corr
        super().__init__(parent, title=title)

    def body(self, master: tk.Frame) -> tk.Widget:
        self.angle = tk.StringVar(value="0.0")
        self.threshold = tk.StringVar(value="0.7")
        self.pad_input = tk.BooleanVar(value=True)
        self.display = tk.StringVar(value="darken")

        info = f"Template size: {self.template.width} × {self.template.height} px"
        if self.max_corr is not None:
            info += f"  (unrotated peak NCC: {self.max_corr:.3f})"
        ttk.Label(master, text=info).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(master, text="Rotation angle (degrees):").grid(row=1, column=0, sticky="w")
        angle_box = ttk.Spinbox(
            master, from_=ANGLE_RANGE[0], to=ANGLE_RANGE[1], increment=5.0, textvariable=self.angle, width=8
        )
        angle_box.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=2)
        ttk.Label(master, text="Rotates template with skimage.transform.rotate(template, a).").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )

        ttk.Label(master, text="Correlation threshold (-1 to 1):").grid(row=3, column=0, sticky="w")
        thresh_box = ttk.Spinbox(
            master, from_=THRESHOLD_RANGE[0], to=THRESHOLD_RANGE[1], increment=0.05, textvariable=self.threshold, width=8
        )
        thresh_box.grid(row=3, column=1, sticky="w", padx=(6, 0), pady=2)
        ttk.Label(master, text="Pixels with NCC ≥ threshold are highlighted.").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )

        ttk.Checkbutton(
            master,
            text="Pad input (pad_input=True)",
            variable=self.pad_input,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(master, text="Superimposition / display style (from Set 3):").grid(
            row=6, column=0, columnspan=2, sticky="w"
        )
        for row, (key, text) in enumerate(CONFIG_DISPLAYS.items(), start=7):
            ttk.Radiobutton(master, text=text, value=key, variable=self.display).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=(12, 0)
            )
        return angle_box

    def validate(self) -> bool:
        try:
            angle = float(self.angle.get())
        except ValueError:
            messagebox.showerror("Template matching", "Rotation angle must be a number.", parent=self)
            return False
        if not ANGLE_RANGE[0] <= angle <= ANGLE_RANGE[1]:
            messagebox.showerror(
                "Template matching",
                f"Rotation angle must be between {ANGLE_RANGE[0]:g} and {ANGLE_RANGE[1]:g} degrees.",
                parent=self,
            )
            return False

        try:
            threshold = float(self.threshold.get())
        except ValueError:
            messagebox.showerror("Template matching", "Threshold must be a number.", parent=self)
            return False
        if not THRESHOLD_RANGE[0] <= threshold <= THRESHOLD_RANGE[1]:
            messagebox.showerror(
                "Template matching",
                f"Threshold must be between {THRESHOLD_RANGE[0]:g} and {THRESHOLD_RANGE[1]:g}.",
                parent=self,
            )
            return False

        self._settings = TemplateMatchingSettings(
            angle=angle,
            threshold=threshold,
            pad_input=self.pad_input.get(),
            display=self.display.get(),
        )
        return True

    def apply(self) -> None:
        self.result = self._settings


class TemplateMatchingConfigTool(ForensicsTool):
    tool_id = "template_matching_configs"
    title = "Template matching configs"
    category = "Set 4: Features & Detection"
    description = (
        "Match a rotatable template with a configurable correlation threshold and superimpose "
        "matches using Set 3 styles."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by main window

        filename = filedialog.askopenfilename(title="Select template image", filetypes=TEMPLATE_TYPES, parent=parent)
        if not filename:
            return None
        with Image.open(filename) as opened:
            template = opened.copy()

        # Pre-compute unrotated peak correlation if template fits, to show a helpful hint in the dialog
        max_corr: float | None = None
        try:
            initial_corr, _ = compute_ncc(document.current, template, angle=0.0, pad_input=True)
            max_corr = float(initial_corr.max())
        except ValueError:
            pass

        settings = TemplateMatchingDialog(
            parent, title="Template matching configs", template=template, max_corr=max_corr
        ).result
        if settings is None:
            return None

        corr, tmpl_gray = compute_ncc(
            document.current, template, angle=settings.angle, pad_input=settings.pad_input
        )
        matches = threshold_ncc(corr, settings.threshold)
        match_count = int(np.sum(matches))
        peak_row, peak_col = np.unravel_index(int(np.argmax(corr)), corr.shape)

        details: dict[str, Any] = {
            "Operation": "Template matching (configurable)",
            "Library": "skimage.feature.match_template, skimage.transform.rotate",
            "Template": Path(filename).name,
            "Template size": f"{template.width} × {template.height}",
            "Rotation angle": f"{settings.angle:g}°",
            "Threshold": f"{settings.threshold:g}",
            "Pad input": str(settings.pad_input),
            "Max correlation": f"{float(corr.max()):.4f}",
            "Peak location (x, y)": f"({int(peak_col)}, {int(peak_row)})",
            "Matched pixels": match_count,
            "Display": CONFIG_DISPLAYS[settings.display],
        }

        if settings.display == "popup":
            figure = config_comparison_figure(document.current, tmpl_gray, corr, settings)
            show_figure(parent, "Template matching comparison", figure)
            return ToolResult(
                message=(
                    f"Opened template matching comparison for {Path(filename).name} "
                    f"(angle={settings.angle:g}°, threshold={settings.threshold:g}, {match_count} matched px)."
                ),
                details=details,
            )

        output = render_matches(document.current, corr, settings.threshold, settings.display)
        return ToolResult(
            image=output,
            message=(
                f"Matched {Path(filename).name} (angle={settings.angle:g}°, threshold≥{settings.threshold:g}): "
                f"{match_count} matched pixel(s)."
            ),
            details=details,
        )


__all__ = [
    "ANGLE_RANGE",
    "CONFIG_DISPLAYS",
    "THRESHOLD_RANGE",
    "TemplateMatchingConfigTool",
    "TemplateMatchingDialog",
    "TemplateMatchingSettings",
    "config_comparison_figure",
    "render_matches",
    "rotate_template",
    "threshold_ncc",
]

__all__ = [
    "ANGLE_RANGE",
    "CONFIG_DISPLAYS",
    "THRESHOLD_RANGE",
    "TemplateMatchingConfigTool",
    "TemplateMatchingDialog",
    "TemplateMatchingSettings",
    "config_comparison_figure",
    "render_matches",
    "rotate_template",
    "threshold_ncc",
]

