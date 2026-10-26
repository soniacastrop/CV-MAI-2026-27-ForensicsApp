"""Set 5.3: Match ORB / SIFT descriptors of the current image with a second image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import ORB, SIFT, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialog_fields import read_number, spin_row
from .modes import split_alpha

IMAGE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]

# Descriptor -> (label in the dialog, metric used by match_descriptors)
DESCRIPTORS = {
    "orb": ("ORB (binary descriptor, Hamming distance)", "hamming"),
    "sift": ("SIFT (128-d float descriptor, Euclidean distance)", "euclidean"),
}
N_KEYPOINTS_RANGE = (1, 5000)
FAST_THRESHOLD_RANGE = (0.0, 1.0)
N_OCTAVES_RANGE = (1, 10)
N_SCALES_RANGE = (1, 10)
SIGMA_RANGE = (0.1, 10.0)
MAX_SIDE_RANGE = (64, 4000)
RATIO_RANGE = (0.0, 1.0)  # max_ratio must be in (0, 1]
RESIDUAL_RANGE = (0.1, 50.0)
RANSAC_MIN_SAMPLES = 3  # an affine transform needs three point pairs
RANSAC_TRIALS = 500

KEYPOINT_COLOUR = (255, 0, 0)
LINE_COLOURS = (  # matplotlib's tab10, so neighbouring matches are easy to tell apart
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
)


@dataclass(frozen=True)
class MatchingSettings:
    descriptor: str = "orb"
    n_keypoints: int = 200  # ORB
    fast_threshold: float = 0.08  # ORB
    n_octaves: int = 5  # SIFT
    n_scales: int = 3  # SIFT
    sigma_min: float = 1.6  # SIFT
    max_side: int = 600  # images are downscaled to this before extraction
    cross_check: bool = True
    max_ratio: float = 0.8
    use_ransac: bool = False
    residual_threshold: float = 3.0
    show_keypoints: bool = True

    @property
    def metric(self) -> str:
        return DESCRIPTORS[self.descriptor][1]


@dataclass(frozen=True)
class Features:
    """Keypoints ``(row, col)`` and descriptors of an image downscaled by ``scale``."""

    gray: np.ndarray
    keypoints: np.ndarray
    descriptors: np.ndarray
    scale: float


def prepare_gray(image: Image.Image, max_side: int) -> tuple[np.ndarray, float]:
    """Gray float image in [0, 1], downscaled so its longest side is at most ``max_side``."""
    base, _alpha = split_alpha(image)
    gray = base.convert("L")
    scale = min(1.0, max_side / max(gray.size))
    if scale < 1.0:
        size = (max(1, round(gray.width * scale)), max(1, round(gray.height * scale)))
        gray = gray.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(gray, dtype=np.float64) / 255.0, scale


def extract_features(image: Image.Image, settings: MatchingSettings) -> Features:
    """Detect keypoints and extract ORB or SIFT descriptors; no features gives empty arrays."""
    gray, scale = prepare_gray(image, settings.max_side)
    if settings.descriptor == "orb":
        extractor: ORB | SIFT = ORB(n_keypoints=settings.n_keypoints, fast_threshold=settings.fast_threshold)
        empty = np.zeros((0, 256), dtype=bool)
    elif settings.descriptor == "sift":
        extractor = SIFT(n_octaves=settings.n_octaves, n_scales=settings.n_scales, sigma_min=settings.sigma_min)
        empty = np.zeros((0, 128), dtype=np.uint8)
    else:
        raise ValueError(f"Descriptor must be one of {', '.join(DESCRIPTORS)}.")
    try:
        extractor.detect_and_extract(gray)
    except (RuntimeError, ValueError, IndexError):
        # Flat or tiny images: skimage raises instead of returning zero keypoints
        return Features(gray, np.zeros((0, 2)), empty, scale)
    return Features(gray, np.asarray(extractor.keypoints, dtype=np.float64), extractor.descriptors, scale)


def ransac_inliers(
    keypoints1: np.ndarray, keypoints2: np.ndarray, matches: np.ndarray, residual_threshold: float
) -> np.ndarray:
    """Boolean mask of the matches consistent with one affine transform (all ``False`` if too few)."""
    if len(matches) <= RANSAC_MIN_SAMPLES:
        return np.zeros(len(matches), dtype=bool)
    src = keypoints1[matches[:, 0]][:, ::-1]  # (row, col) -> (x, y)
    dst = keypoints2[matches[:, 1]][:, ::-1]
    _model, inliers = ransac(
        (src, dst),
        AffineTransform,
        min_samples=RANSAC_MIN_SAMPLES,
        residual_threshold=residual_threshold,
        max_trials=RANSAC_TRIALS,
        rng=0,
    )
    return np.zeros(len(matches), dtype=bool) if inliers is None else inliers


def match_features(features1: Features, features2: Features, settings: MatchingSettings) -> np.ndarray:
    """``(K, 2)`` index pairs into the two keypoint arrays, optionally kept to RANSAC inliers."""
    if len(features1.descriptors) == 0 or len(features2.descriptors) == 0:
        return np.zeros((0, 2), dtype=np.intp)
    matches = match_descriptors(
        features1.descriptors,
        features2.descriptors,
        metric=settings.metric,
        cross_check=settings.cross_check,
        max_ratio=settings.max_ratio,
    )
    if settings.use_ransac:
        matches = matches[ransac_inliers(features1.keypoints, features2.keypoints, matches, settings.residual_threshold)]
    return matches


def draw_matches(features1: Features, features2: Features, matches: np.ndarray, show_keypoints: bool = True) -> Image.Image:
    """Both (processed) gray images side by side with keypoints as red circles and coloured match lines."""
    gray1, gray2 = (np.round(f.gray * 255).astype(np.uint8) for f in (features1, features2))
    height = max(gray1.shape[0], gray2.shape[0])
    offset = gray1.shape[1]
    canvas = Image.new("RGB", (offset + gray2.shape[1], height), (0, 0, 0))
    canvas.paste(Image.fromarray(gray1, mode="L").convert("RGB"), (0, 0))
    canvas.paste(Image.fromarray(gray2, mode="L").convert("RGB"), (offset, 0))

    draw = ImageDraw.Draw(canvas)
    radius = max(2, round(height / 150))
    width = max(1, round(height / 300))

    def circle(row: float, col: float) -> None:
        draw.ellipse((col - radius, row - radius, col + radius, row + radius), outline=KEYPOINT_COLOUR, width=width)

    if show_keypoints:
        for row, col in features1.keypoints:
            circle(row, col)
        for row, col in features2.keypoints:
            circle(row, col + offset)
    for index, (i, j) in enumerate(matches):
        (r1, c1), (r2, c2) = features1.keypoints[i], features2.keypoints[j]
        colour = LINE_COLOURS[index % len(LINE_COLOURS)]
        draw.line((c1, r1, c2 + offset, r2), fill=colour, width=width)
        if not show_keypoints:  # still mark the matched endpoints
            circle(r1, c1)
            circle(r2, c2 + offset)
    return canvas


def describe_settings(settings: MatchingSettings) -> dict[str, Any]:
    """Settings rows shared by the matching and lookup tools' details panels."""
    if settings.descriptor == "orb":
        extractor = f"ORB(n_keypoints={settings.n_keypoints}, fast_threshold={settings.fast_threshold:g})"
    else:
        extractor = (
            f"SIFT(n_octaves={settings.n_octaves}, n_scales={settings.n_scales}, "
            f"sigma_min={settings.sigma_min:g})"
        )
    return {
        "Extractor": extractor,
        "Max side processed": f"{settings.max_side} px",
        "Metric": settings.metric,
        "Cross check": str(settings.cross_check),
        "Max ratio": f"{settings.max_ratio:g}",
        "RANSAC (affine)": f"residual ≤ {settings.residual_threshold:g} px" if settings.use_ransac else "Off",
    }


class MatchingDialog(simpledialog.Dialog):
    """Descriptor (ORB / SIFT) parameters and match_descriptors options."""

    def __init__(self, parent: tk.Misc, title: str, defaults: MatchingSettings = MatchingSettings()) -> None:
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
        self.cross_check = tk.BooleanVar(value=d.cross_check)
        self.max_ratio = tk.StringVar(value=str(d.max_ratio))
        self.use_ransac = tk.BooleanVar(value=d.use_ransac)
        self.residual = tk.StringVar(value=str(d.residual_threshold))
        self.show_keypoints = tk.BooleanVar(value=d.show_keypoints)

        row = 0
        ttk.Label(master, text="Descriptor:").grid(row=row, column=0, columnspan=2, sticky="w")
        for key, (text, _metric) in DESCRIPTORS.items():
            row += 1
            ttk.Radiobutton(
                master, text=text, value=key, variable=self.descriptor, command=self._toggle
            ).grid(row=row, column=0, columnspan=2, sticky="w", padx=(12, 0))

        self.orb_frame = ttk.LabelFrame(master, text="ORB")
        self.sift_frame = ttk.LabelFrame(master, text="SIFT")
        row += 1
        self.orb_frame.grid(row=row, column=0, columnspan=2, sticky="we", pady=(6, 0))
        self.sift_frame.grid(row=row, column=0, columnspan=2, sticky="we", pady=(6, 0))
        first = spin_row(self.orb_frame, 0, "Number of keypoints:", self.n_keypoints, N_KEYPOINTS_RANGE, 50)
        spin_row(self.orb_frame, 1, "FAST threshold:", self.fast_threshold, FAST_THRESHOLD_RANGE, 0.01)
        spin_row(self.sift_frame, 0, "Number of octaves:", self.n_octaves, N_OCTAVES_RANGE, 1)
        spin_row(self.sift_frame, 1, "Scales per octave:", self.n_scales, N_SCALES_RANGE, 1)
        spin_row(self.sift_frame, 2, "Sigma min:", self.sigma_min, SIGMA_RANGE, 0.1)

        row += 1
        common = ttk.LabelFrame(master, text="Matching")
        common.grid(row=row, column=0, columnspan=2, sticky="we", pady=(6, 0))
        spin_row(common, 0, "Downscale to max side (px):", self.max_side, MAX_SIDE_RANGE, 100)
        ttk.Checkbutton(common, text="Cross check (mutual nearest neighbours)", variable=self.cross_check).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=6
        )
        spin_row(common, 2, "Max ratio (Lowe's test, 0–1]:", self.max_ratio, RATIO_RANGE, 0.05)
        ttk.Checkbutton(
            common, text="Keep only RANSAC inliers (affine geometric check)", variable=self.use_ransac
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=6)
        spin_row(common, 4, "RANSAC residual threshold (px):", self.residual, RESIDUAL_RANGE, 0.5)
        ttk.Checkbutton(common, text="Draw unmatched keypoints too", variable=self.show_keypoints).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=6
        )

        self.extra_body(master, row + 1)
        self._toggle()
        return first

    def extra_body(self, master: tk.Frame, row: int) -> None:
        """Hook for subclasses to add rows below the matching options."""

    def _toggle(self) -> None:
        """Only the chosen descriptor's parameters are shown."""
        orb = self.descriptor.get() == "orb"
        (self.sift_frame if orb else self.orb_frame).grid_remove()
        (self.orb_frame if orb else self.sift_frame).grid()

    def read_settings(self) -> MatchingSettings:
        """Parse the widgets; raises ``ValueError`` with a user-facing message."""
        max_ratio = read_number(self.max_ratio, "Max ratio", RATIO_RANGE)
        if max_ratio == 0:
            raise ValueError("Max ratio must be greater than 0.")
        return MatchingSettings(
            descriptor=self.descriptor.get(),
            n_keypoints=read_number(self.n_keypoints, "Number of keypoints", N_KEYPOINTS_RANGE, int),
            fast_threshold=read_number(self.fast_threshold, "FAST threshold", FAST_THRESHOLD_RANGE),
            n_octaves=read_number(self.n_octaves, "Number of octaves", N_OCTAVES_RANGE, int),
            n_scales=read_number(self.n_scales, "Scales per octave", N_SCALES_RANGE, int),
            sigma_min=read_number(self.sigma_min, "Sigma min", SIGMA_RANGE),
            max_side=read_number(self.max_side, "Max side", MAX_SIDE_RANGE, int),
            cross_check=self.cross_check.get(),
            max_ratio=max_ratio,
            use_ransac=self.use_ransac.get(),
            residual_threshold=read_number(self.residual, "RANSAC residual threshold", RESIDUAL_RANGE),
            show_keypoints=self.show_keypoints.get(),
        )

    def validate(self) -> bool:
        try:
            self._settings = self.read_settings()
        except ValueError as error:
            messagebox.showerror(self.title(), str(error), parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self._settings


class FeatureMatchingTool(ForensicsTool):
    tool_id = "feature_matching"
    title = "Feature matching"
    category = "Set 5: Keypoints & Matching"
    description = (
        "Match ORB or SIFT descriptors of the image with a second image (cross check, ratio test, "
        "optional RANSAC) and draw the matches side by side."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by main window

        filename = filedialog.askopenfilename(title="Select second image", filetypes=IMAGE_TYPES, parent=parent)
        if not filename:
            return None
        with Image.open(filename) as opened:
            second = opened.copy()

        settings = MatchingDialog(parent, title="Feature matching").result
        if settings is None:
            return None

        features1 = extract_features(document.current, settings)
        features2 = extract_features(second, settings)
        matches = match_features(features1, features2, settings)
        name = Path(filename).name

        details: dict[str, Any] = {
            "Operation": "Descriptor matching",
            "Library": "skimage.feature.match_descriptors" + (", skimage.measure.ransac" if settings.use_ransac else ""),
            "Second image": name,
            **describe_settings(settings),
            "Keypoints (current)": len(features1.keypoints),
            f"Keypoints ({name})": len(features2.keypoints),
            "Matches": len(matches),
        }
        return ToolResult(
            image=draw_matches(features1, features2, matches, settings.show_keypoints),
            message=f"{len(matches)} {settings.descriptor.upper()} match(es) with {name}.",
            details=details,
        )


__all__ = [
    "DESCRIPTORS",
    "FeatureMatchingTool",
    "Features",
    "MatchingDialog",
    "MatchingSettings",
    "describe_settings",
    "draw_matches",
    "extract_features",
    "match_features",
    "prepare_gray",
    "ransac_inliers",
]
