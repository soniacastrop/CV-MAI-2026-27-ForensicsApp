"""Set 5.1: Corner detection (skimage.feature: FAST and Harris) with modifiable threshold for peaks."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Any

from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import corner_fast, corner_harris, corner_peaks

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialog_fields import read_number, spin_row
from .hog import show_figure
from .modes import split_alpha

# Detector key -> label in the dialog
CORNER_DETECTORS = {
    "fast": "FAST (skimage.feature.corner_fast)",
    "harris": "Harris (skimage.feature.corner_harris)",
}

# Display key -> label in the dialog
CORNER_DISPLAYS = {
    "peaks_gray": "Corner peaks (red circles) on grayscale image (as in slide)",
    "peaks_rgb": "Corner peaks (red circles) on original colour image",
    "response": "Corner response map (corner_fast / corner_harris)",
    "popup": "Side-by-side comparison: response map & corner peaks (popup)",
}

THRESHOLD_REL_RANGE = (0.0, 1.0)
MIN_DISTANCE_RANGE = (1, 100)
FAST_N_RANGE = (1, 16)
FAST_THRESHOLD_RANGE = (0.0, 1.0)
HARRIS_K_RANGE = (0.0, 0.25)
HARRIS_SIGMA_RANGE = (0.1, 10.0)

CORNER_COLOUR = (255, 0, 0)


@dataclass(frozen=True)
class CornerDetectionSettings:
    detector: str = "fast"
    threshold_rel: float = 0.1
    min_distance: int = 1
    fast_n: int = 12
    fast_threshold: float = 0.15
    harris_k: float = 0.05
    harris_sigma: float = 1.0
    display: str = "peaks_gray"


def to_gray_float(image: Image.Image) -> np.ndarray:
    """Convert any PIL image (dropping alpha if present) to a 2-D float64 luminance array in [0, 1]."""
    base, _alpha = split_alpha(image)
    return np.asarray(base.convert("L"), dtype=np.float64) / 255.0


def compute_corner_response(image: Image.Image, settings: CornerDetectionSettings) -> np.ndarray:
    """Compute the FAST or Harris corner response map on the grayscale float image."""
    gray = to_gray_float(image)
    if settings.detector == "fast":
        if not FAST_N_RANGE[0] <= settings.fast_n <= FAST_N_RANGE[1]:
            raise ValueError(f"FAST n must be between {FAST_N_RANGE[0]} and {FAST_N_RANGE[1]}.")
        if not FAST_THRESHOLD_RANGE[0] <= settings.fast_threshold <= FAST_THRESHOLD_RANGE[1]:
            raise ValueError(
                f"FAST threshold must be between {FAST_THRESHOLD_RANGE[0]:g} and {FAST_THRESHOLD_RANGE[1]:g}."
            )
        if gray.max() == gray.min():
            return np.zeros_like(gray)
        return corner_fast(gray, n=settings.fast_n, threshold=settings.fast_threshold)
    if settings.detector == "harris":
        if not HARRIS_K_RANGE[0] <= settings.harris_k <= HARRIS_K_RANGE[1]:
            raise ValueError(f"Harris k must be between {HARRIS_K_RANGE[0]:g} and {HARRIS_K_RANGE[1]:g}.")
        if not HARRIS_SIGMA_RANGE[0] <= settings.harris_sigma <= HARRIS_SIGMA_RANGE[1]:
            raise ValueError(
                f"Harris sigma must be between {HARRIS_SIGMA_RANGE[0]:g} and {HARRIS_SIGMA_RANGE[1]:g}."
            )
        if gray.max() == gray.min():
            return np.zeros_like(gray)
        # corner_harris zero-pads the image, so the frame itself looks like a strong edge/corner and its
        # artificial peaks dominate threshold_rel. Mirror-pad first and crop the response back instead.
        pad = int(np.ceil(4 * settings.harris_sigma)) + 2
        padded = np.pad(gray, pad, mode="reflect")
        response = corner_harris(padded, method="k", k=settings.harris_k, sigma=settings.harris_sigma)
        return response[pad:-pad, pad:-pad]
    raise ValueError(f"Detector must be one of {', '.join(CORNER_DETECTORS)}.")


def detect_corner_peaks(response: np.ndarray, settings: CornerDetectionSettings) -> np.ndarray:
    """Extract corner peak coordinates ``(row, col)`` from ``response`` using ``skimage.feature.corner_peaks``."""
    if not THRESHOLD_REL_RANGE[0] <= settings.threshold_rel <= THRESHOLD_REL_RANGE[1]:
        raise ValueError(
            f"Peak relative threshold must be between {THRESHOLD_REL_RANGE[0]:g} and {THRESHOLD_REL_RANGE[1]:g}."
        )
    if not MIN_DISTANCE_RANGE[0] <= settings.min_distance <= MIN_DISTANCE_RANGE[1]:
        raise ValueError(
            f"Minimum peak distance must be between {MIN_DISTANCE_RANGE[0]} and {MIN_DISTANCE_RANGE[1]}."
        )
    if response.size == 0 or float(response.max()) <= 0.0:
        return np.zeros((0, 2), dtype=np.intp)
    peaks = corner_peaks(
        response,
        min_distance=settings.min_distance,
        threshold_rel=settings.threshold_rel,
    )
    if peaks.size == 0:
        return np.zeros((0, 2), dtype=np.intp)
    return np.asarray(peaks, dtype=np.intp)


def detect_corners(image: Image.Image, settings: CornerDetectionSettings) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(response, corner_points)`` for ``image`` using ``settings``."""
    response = compute_corner_response(image, settings)
    corners = detect_corner_peaks(response, settings)
    return response, corners


def response_to_image(response: np.ndarray) -> Image.Image:
    """Rescale a corner response array from [min, max] to a 0..255 grayscale ``L`` image."""
    low, high = float(response.min()), float(response.max())
    if high > low:
        scaled = (response - low) / (high - low)
    else:
        scaled = np.zeros_like(response, dtype=np.float64)
    return Image.fromarray(np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8), mode="L")


def draw_keypoints_on_image(
    image: Image.Image,
    keypoints: np.ndarray,
    *,
    grayscale: bool = True,
) -> Image.Image:
    """Draw red circles at ``(row, col)`` positions on the grayscale or colour image."""
    base, _alpha = split_alpha(image)
    canvas = (base.convert("L").convert("RGB") if grayscale else base.convert("RGB")).copy()
    draw = ImageDraw.Draw(canvas)
    radius = max(3, round(max(canvas.size) / 140))
    width = max(1, round(max(canvas.size) / 280))
    for row, col in keypoints:
        r, c = float(row), float(col)
        draw.ellipse((c - radius, r - radius, c + radius, r + radius), outline=CORNER_COLOUR, width=width)
    return canvas


def render_corners(
    image: Image.Image,
    response: np.ndarray,
    corners: np.ndarray,
    display: str,
) -> Image.Image:
    """Render the corner detection result as a canvas image according to ``display``."""
    if display == "peaks_gray":
        return draw_keypoints_on_image(image, corners, grayscale=True)
    if display == "peaks_rgb":
        return draw_keypoints_on_image(image, corners, grayscale=False)
    if display == "response":
        return response_to_image(response)
    raise ValueError(f"Display must be one of {', '.join(k for k in CORNER_DISPLAYS if k != 'popup')}.")


def corner_comparison_figure(
    image: Image.Image,
    response: np.ndarray,
    corners: np.ndarray,
    settings: CornerDetectionSettings,
) -> Figure:
    """Build the 2-panel comparison figure from Set 5.1: Corner response map and Corner peaks on grayscale image."""
    figure = Figure(figsize=(10, 4.5), layout="constrained")
    ax = figure.subplots(1, 2)
    detector_label = "FAST" if settings.detector == "fast" else "Harris"
    ax[0].imshow(response, cmap="viridis")
    ax[0].set_title(f"{detector_label} corner response")
    ax[1].imshow(to_gray_float(image), cmap="gray")
    if len(corners):
        ax[1].scatter(corners[:, 1], corners[:, 0], facecolors="none", edgecolors="r")
    ax[1].set_title(f"Corner peaks ({len(corners)} found, threshold_rel={settings.threshold_rel:g})")
    return figure


class CornerDetectionDialog(simpledialog.Dialog):
    """Dialog for Set 5.1: FAST / Harris corner detection and modifiable peak threshold."""

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        defaults: CornerDetectionSettings = CornerDetectionSettings(),
    ) -> None:
        self.defaults = defaults
        super().__init__(parent, title=title)

    def body(self, master: tk.Frame) -> tk.Widget:
        d = self.defaults
        self.detector = tk.StringVar(value=d.detector)
        self.threshold_rel = tk.StringVar(value=str(d.threshold_rel))
        self.min_distance = tk.StringVar(value=str(d.min_distance))
        self.fast_n = tk.StringVar(value=str(d.fast_n))
        self.fast_threshold = tk.StringVar(value=str(d.fast_threshold))
        self.harris_k = tk.StringVar(value=str(d.harris_k))
        self.harris_sigma = tk.StringVar(value=str(d.harris_sigma))
        self.display = tk.StringVar(value=d.display)

        row = 0
        ttk.Label(master, text="Corner detector:").grid(row=row, column=0, columnspan=2, sticky="w")
        for key, text in CORNER_DETECTORS.items():
            row += 1
            ttk.Radiobutton(
                master, text=text, value=key, variable=self.detector, command=self._toggle
            ).grid(row=row, column=0, columnspan=2, sticky="w", padx=(12, 0))

        self.fast_frame = ttk.LabelFrame(master, text="FAST parameters (corner_fast)")
        self.harris_frame = ttk.LabelFrame(master, text="Harris parameters (corner_harris)")
        row += 1
        self.fast_frame.grid(row=row, column=0, columnspan=2, sticky="we", pady=(6, 0))
        self.harris_frame.grid(row=row, column=0, columnspan=2, sticky="we", pady=(6, 0))
        spin_row(self.fast_frame, 0, "Consecutive pixels (n):", self.fast_n, FAST_N_RANGE, 1)
        spin_row(self.fast_frame, 1, "FAST intensity threshold:", self.fast_threshold, FAST_THRESHOLD_RANGE, 0.01)
        # FAST scores are close together (weakest ≈ 20% of strongest), so threshold_rel only bites above ~0.3
        ttk.Label(
            self.fast_frame,
            text="Raise this for fewer corners: FAST scores are close together, so the\n"
            "peak threshold below only removes corners above about 0.3.",
            foreground="gray",
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 4))
        spin_row(self.harris_frame, 0, "Sensitivity factor (k):", self.harris_k, HARRIS_K_RANGE, 0.01)
        spin_row(self.harris_frame, 1, "Gaussian sigma:", self.harris_sigma, HARRIS_SIGMA_RANGE, 0.1)

        row += 1
        peaks_frame = ttk.LabelFrame(master, text="Peak filtering (corner_peaks)")
        peaks_frame.grid(row=row, column=0, columnspan=2, sticky="we", pady=(6, 0))
        first = spin_row(
            peaks_frame, 0, "Peak relative threshold (0–1):", self.threshold_rel, THRESHOLD_REL_RANGE, 0.05
        )
        spin_row(peaks_frame, 1, "Minimum distance between peaks (px):", self.min_distance, MIN_DISTANCE_RANGE, 1)

        row += 1
        ttk.Label(master, text="Show as:").grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        for key, text in CORNER_DISPLAYS.items():
            row += 1
            ttk.Radiobutton(master, text=text, value=key, variable=self.display).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=(12, 0)
            )

        self._toggle()
        return first

    def _toggle(self) -> None:
        is_fast = self.detector.get() == "fast"
        (self.harris_frame if is_fast else self.fast_frame).grid_remove()
        (self.fast_frame if is_fast else self.harris_frame).grid()

    def read_settings(self) -> CornerDetectionSettings:
        return CornerDetectionSettings(
            detector=self.detector.get(),
            threshold_rel=read_number(self.threshold_rel, "Peak relative threshold", THRESHOLD_REL_RANGE),
            min_distance=read_number(self.min_distance, "Minimum peak distance", MIN_DISTANCE_RANGE, int),
            fast_n=read_number(self.fast_n, "FAST n", FAST_N_RANGE, int),
            fast_threshold=read_number(self.fast_threshold, "FAST threshold", FAST_THRESHOLD_RANGE),
            harris_k=read_number(self.harris_k, "Harris k", HARRIS_K_RANGE),
            harris_sigma=read_number(self.harris_sigma, "Harris sigma", HARRIS_SIGMA_RANGE),
            display=self.display.get(),
        )

    def validate(self) -> bool:
        try:
            self._settings = self.read_settings()
        except ValueError as error:
            messagebox.showerror("Corner detection", str(error), parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self._settings


class CornerDetectionTool(ForensicsTool):
    tool_id = "corner_detection"
    title = "Corner detection (FAST / Harris)"
    category = "Set 5: Keypoints & Matching"
    description = (
        "Detect corners with FAST (corner_fast) or Harris (corner_harris) from skimage.feature "
        "and extract corner_peaks with a modifiable peak threshold."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by main window

        settings = CornerDetectionDialog(parent, title="Corner detection (FAST / Harris)").result
        if settings is None:
            return None

        response, corners = detect_corners(document.current, settings)
        if settings.detector == "fast":
            detector_desc = f"corner_fast(n={settings.fast_n}, threshold={settings.fast_threshold:g})"
            library = "skimage.feature.corner_fast, skimage.feature.corner_peaks"
        else:
            detector_desc = f"corner_harris(method='k', k={settings.harris_k:g}, sigma={settings.harris_sigma:g})"
            library = "skimage.feature.corner_harris, skimage.feature.corner_peaks"

        details: dict[str, Any] = {
            "Operation": "Corner detection",
            "Library": library,
            "Detector": detector_desc,
            "Peak relative threshold": f"{settings.threshold_rel:g}",
            "Min peak distance": f"{settings.min_distance} px",
            "Max response": f"{float(response.max()):.4f}",
            "Detected corners": len(corners),
            "Display": CORNER_DISPLAYS[settings.display],
        }

        if settings.display == "popup":
            figure = corner_comparison_figure(document.current, response, corners, settings)
            show_figure(parent, "Corner detection comparison", figure)
            return ToolResult(
                message=(
                    f"Opened {settings.detector.upper()} corner comparison "
                    f"({len(corners)} corner peak(s), threshold_rel={settings.threshold_rel:g})."
                ),
                details=details,
            )

        output = render_corners(document.current, response, corners, settings.display)
        return ToolResult(
            image=output,
            message=(
                f"Detected {len(corners)} {settings.detector.upper()} corner peak(s) "
                f"(threshold_rel={settings.threshold_rel:g}, min_distance={settings.min_distance})."
            ),
            details=details,
        )


__all__ = [
    "CORNER_COLOUR",
    "CORNER_DETECTORS",
    "CORNER_DISPLAYS",
    "CornerDetectionDialog",
    "CornerDetectionSettings",
    "CornerDetectionTool",
    "FAST_N_RANGE",
    "FAST_THRESHOLD_RANGE",
    "HARRIS_K_RANGE",
    "HARRIS_SIGMA_RANGE",
    "MIN_DISTANCE_RANGE",
    "THRESHOLD_REL_RANGE",
    "compute_corner_response",
    "corner_comparison_figure",
    "detect_corner_peaks",
    "detect_corners",
    "draw_keypoints_on_image",
    "render_corners",
    "response_to_image",
    "to_gray_float",
]

