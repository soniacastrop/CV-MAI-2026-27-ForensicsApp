"""Set 5.2: Descriptor and keypoint extractor (ORB, SIFT) with configurable parameters and visualization."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Any

from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .corner_detection import draw_keypoints_on_image
from .feature_matching import (
    DESCRIPTORS,
    FAST_THRESHOLD_RANGE,
    MAX_SIDE_RANGE,
    N_KEYPOINTS_RANGE,
    N_OCTAVES_RANGE,
    N_SCALES_RANGE,
    SIGMA_RANGE,
    Features,
    MatchingSettings,
    extract_features,
)
from .hog import show_figure

DESCRIPTOR_DISPLAYS = {
    "keypoints_gray": "Keypoints (red circles) on grayscale image (as in slide)",
    "keypoints_rgb": "Keypoints (red circles) on original colour image",
    "popup": "Keypoints & descriptor matrix visualization (popup)",
}


@dataclass(frozen=True)
class DescriptorSettings:
    descriptor: str = "orb"
    n_keypoints: int = 200  # ORB
    fast_threshold: float = 0.08  # ORB
    n_octaves: int = 5  # SIFT
    n_scales: int = 3  # SIFT
    sigma_min: float = 1.6  # SIFT
    max_side: int = 600
    display: str = "keypoints_gray"

    def to_matching_settings(self) -> MatchingSettings:
        return MatchingSettings(
            descriptor=self.descriptor,
            n_keypoints=self.n_keypoints,
            fast_threshold=self.fast_threshold,
            n_octaves=self.n_octaves,
            n_scales=self.n_scales,
            sigma_min=self.sigma_min,
            max_side=self.max_side,
        )


def extract_descriptors(image: Image.Image, settings: DescriptorSettings) -> Features:
    """Detect keypoints and extract ORB or SIFT descriptors using ``skimage.feature``."""
    return extract_features(image, settings.to_matching_settings())


def render_keypoints(image: Image.Image, features: Features, display: str) -> Image.Image:
    """Visualize detected keypoints as red circles on the grayscale or colour image."""
    if display not in ("keypoints_gray", "keypoints_rgb"):
        raise ValueError(f"Display must be one of {', '.join(k for k in DESCRIPTOR_DISPLAYS if k != 'popup')}.")
    scaled_keypoints = (
        features.keypoints / features.scale if features.scale > 0 and len(features.keypoints) else features.keypoints
    )
    return draw_keypoints_on_image(image, scaled_keypoints, grayscale=(display == "keypoints_gray"))


def descriptor_comparison_figure(
    features: Features,
    settings: DescriptorSettings,
) -> Figure:
    """Build a 2-panel figure showing the detected keypoints on the image and the extracted descriptor matrix."""
    figure = Figure(figsize=(11, 4.5), layout="constrained")
    ax = figure.subplots(1, 2)
    name = settings.descriptor.upper()
    ax[0].imshow(features.gray, cmap="gray")
    if len(features.keypoints):
        ax[0].scatter(features.keypoints[:, 1], features.keypoints[:, 0], facecolors="none", edgecolors="r")
    ax[0].set_title(f"{name} keypoints ({len(features.keypoints)} found)")

    if len(features.descriptors):
        ax[1].imshow(features.descriptors.astype(np.float64), aspect="auto", cmap="viridis")
    else:
        ax[1].imshow(np.zeros((1, features.descriptors.shape[1]), dtype=np.float64), aspect="auto", cmap="viridis")
    ax[1].set_xlabel("Descriptor dimension")
    ax[1].set_ylabel("Keypoint index")
    ax[1].set_title(f"{name} descriptors ({features.descriptors.shape[0]} × {features.descriptors.shape[1]})")
    return figure


class DescriptorDialog(simpledialog.Dialog):
    """Dialog for Set 5.2: ORB / SIFT configurable parameters and keypoint visualization."""

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        defaults: DescriptorSettings = DescriptorSettings(),
    ) -> None:
        self.defaults = defaults
        super().__init__(parent, title=title)

    def body(self, master: tk.Frame) -> tk.Widget:
        d = self.defaults
        self.descriptor = tk.StringVar(value=d.descriptor)
        self.n_keypoints = tk.StringVar(value=str(d.n_keypoints))
        self.fast_threshold = tk.StringVar(value=str(d.fast_threshold))
        self.n_octaves = tk.StringVar(value=str(d.n_octaves))
        self.n_scales = tk.StringVar(value=str(d.n_scales))
        self.sigma_min = tk.StringVar(value=str(d.sigma_min))
        self.max_side = tk.StringVar(value=str(d.max_side))
        self.display = tk.StringVar(value=d.display)

        row = 0
        ttk.Label(master, text="Descriptor extractor:").grid(row=row, column=0, columnspan=2, sticky="w")
        for key, (text, _metric) in DESCRIPTORS.items():
            row += 1
            ttk.Radiobutton(
                master, text=text, value=key, variable=self.descriptor, command=self._toggle
            ).grid(row=row, column=0, columnspan=2, sticky="w", padx=(12, 0))

        self.orb_frame = ttk.LabelFrame(master, text="ORB parameters (skimage.feature.ORB)")
        self.sift_frame = ttk.LabelFrame(master, text="SIFT parameters (skimage.feature.SIFT)")
        row += 1
        self.orb_frame.grid(row=row, column=0, columnspan=2, sticky="we", pady=(6, 0))
        self.sift_frame.grid(row=row, column=0, columnspan=2, sticky="we", pady=(6, 0))
        first = self._spin(self.orb_frame, 0, "Number of keypoints (n_keypoints):", self.n_keypoints, N_KEYPOINTS_RANGE, 50)
        self._spin(self.orb_frame, 1, "FAST threshold:", self.fast_threshold, FAST_THRESHOLD_RANGE, 0.01)
        self._spin(self.sift_frame, 0, "Number of octaves (n_octaves):", self.n_octaves, N_OCTAVES_RANGE, 1)
        self._spin(self.sift_frame, 1, "Scales per octave (n_scales):", self.n_scales, N_SCALES_RANGE, 1)
        self._spin(self.sift_frame, 2, "Sigma min (sigma_min):", self.sigma_min, SIGMA_RANGE, 0.1)

        row += 1
        prep_frame = ttk.LabelFrame(master, text="Preprocessing")
        prep_frame.grid(row=row, column=0, columnspan=2, sticky="we", pady=(6, 0))
        self._spin(prep_frame, 0, "Downscale to max side (px):", self.max_side, MAX_SIDE_RANGE, 100)

        row += 1
        ttk.Label(master, text="Show as:").grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        for key, text in DESCRIPTOR_DISPLAYS.items():
            row += 1
            ttk.Radiobutton(master, text=text, value=key, variable=self.display).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=(12, 0)
            )

        self._toggle()
        return first

    @staticmethod
    def _spin(
        master: tk.Misc,
        row: int,
        text: str,
        variable: tk.StringVar,
        bounds: tuple[float, float],
        step: float,
    ) -> ttk.Spinbox:
        ttk.Label(master, text=text).grid(row=row, column=0, sticky="w", padx=6)
        box = ttk.Spinbox(master, from_=bounds[0], to=bounds[1], increment=step, textvariable=variable, width=8)
        box.grid(row=row, column=1, sticky="w", padx=6, pady=2)
        return box

    def _toggle(self) -> None:
        orb = self.descriptor.get() == "orb"
        (self.sift_frame if orb else self.orb_frame).grid_remove()
        (self.orb_frame if orb else self.sift_frame).grid()

    @staticmethod
    def _number(variable: tk.StringVar, name: str, bounds: tuple[float, float], kind: type = float) -> Any:
        try:
            value = kind(variable.get())
        except ValueError:
            raise ValueError(f"{name} must be {'an integer' if kind is int else 'a number'}.") from None
        if not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"{name} must be between {bounds[0]:g} and {bounds[1]:g}.")
        return value

    def read_settings(self) -> DescriptorSettings:
        return DescriptorSettings(
            descriptor=self.descriptor.get(),
            n_keypoints=self._number(self.n_keypoints, "Number of keypoints", N_KEYPOINTS_RANGE, int),
            fast_threshold=self._number(self.fast_threshold, "FAST threshold", FAST_THRESHOLD_RANGE),
            n_octaves=self._number(self.n_octaves, "Number of octaves", N_OCTAVES_RANGE, int),
            n_scales=self._number(self.n_scales, "Scales per octave", N_SCALES_RANGE, int),
            sigma_min=self._number(self.sigma_min, "Sigma min", SIGMA_RANGE),
            max_side=self._number(self.max_side, "Max side", MAX_SIDE_RANGE, int),
            display=self.display.get(),
        )

    def validate(self) -> bool:
        try:
            self._settings = self.read_settings()
        except ValueError as error:
            messagebox.showerror("Descriptor extractor", str(error), parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self._settings


class DescriptorExtractorTool(ForensicsTool):
    tool_id = "descriptor_extractor"
    title = "Descriptor extractor (ORB / SIFT)"
    category = "Set 5: Keypoints & Matching"
    description = (
        "Extract ORB or SIFT keypoints and descriptors with configurable parameters "
        "(n_keypoints, n_octaves, n_scales, sigma_min) and visualize the detected keypoints."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by main window

        settings = DescriptorDialog(parent, title="Descriptor extractor (ORB / SIFT)").result
        if settings is None:
            return None

        features = extract_descriptors(document.current, settings)
        if settings.descriptor == "orb":
            extractor_desc = f"ORB(n_keypoints={settings.n_keypoints}, fast_threshold={settings.fast_threshold:g})"
            library = "skimage.feature.ORB"
        else:
            extractor_desc = (
                f"SIFT(n_octaves={settings.n_octaves}, n_scales={settings.n_scales}, "
                f"sigma_min={settings.sigma_min:g})"
            )
            library = "skimage.feature.SIFT"

        details: dict[str, Any] = {
            "Operation": "Descriptor & keypoint extraction",
            "Library": library,
            "Descriptor": settings.descriptor.upper(),
            "Extractor": extractor_desc,
            "Max side processed": f"{settings.max_side} px",
            "Keypoints detected": len(features.keypoints),
            "Descriptor shape": f"{features.descriptors.shape[0]} × {features.descriptors.shape[1]}",
            "Descriptor dtype": str(features.descriptors.dtype),
            "Display": DESCRIPTOR_DISPLAYS[settings.display],
        }

        if settings.display == "popup":
            figure = descriptor_comparison_figure(features, settings)
            show_figure(parent, "Keypoints & descriptors", figure)
            return ToolResult(
                message=(
                    f"Opened {settings.descriptor.upper()} keypoint & descriptor visualization "
                    f"({len(features.keypoints)} keypoint(s))."
                ),
                details=details,
            )

        output = render_keypoints(document.current, features, settings.display)
        return ToolResult(
            image=output,
            message=(
                f"Extracted {len(features.keypoints)} {settings.descriptor.upper()} keypoint(s) "
                f"and {features.descriptors.shape[1]}-d descriptors."
            ),
            details=details,
        )


__all__ = [
    "DESCRIPTOR_DISPLAYS",
    "DescriptorDialog",
    "DescriptorExtractorTool",
    "DescriptorSettings",
    "descriptor_comparison_figure",
    "extract_descriptors",
    "render_keypoints",
]

