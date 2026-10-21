"""Set 5.4: Look up the image in a folder that best matches the current image."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .feature_matching import (
    Features,
    MatchingDialog,
    MatchingSettings,
    describe_settings,
    draw_matches,
    extract_features,
    match_features,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SKIPPED_DIRS = {"__MACOSX"}  # macOS zip metadata: "._name.jpg" files that are not images

# How candidates are ranked (key -> label in the dialog)
SCORES = {
    "matches": "Number of matches (simplest)",
    "normalized": "Matches ÷ keypoints of the smaller image (fairer to small images)",
}
MAX_RANKED = 10  # rows of the ranking shown in the details panel


@dataclass(frozen=True)
class LookupSettings:
    # On the Set 5 images SIFT ranked by plain match count found the right image for every reference;
    # ORB, the affine RANSAC check and the normalized score each picked wrong images for some.
    matching: MatchingSettings = MatchingSettings(descriptor="sift")
    score: str = "matches"
    recursive: bool = True


@dataclass(frozen=True)
class Candidate:
    path: Path
    features: Features
    matches: np.ndarray
    score: float


def list_images(folder: Path, recursive: bool = True, exclude: Path | None = None) -> list[Path]:
    """Image files in ``folder`` sorted by path, skipping hidden files and ``exclude``."""
    pattern = "**/*" if recursive else "*"
    excluded = exclude.resolve() if exclude is not None else None
    return sorted(
        path
        for path in folder.glob(pattern)
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not path.name.startswith(".")
        and not SKIPPED_DIRS.intersection(path.relative_to(folder).parts)
        and path.resolve() != excluded
    )


def score_matches(score: str, matches: np.ndarray, reference: Features, candidate: Features) -> float:
    if score == "matches":
        return float(len(matches))
    if score == "normalized":
        fewest = min(len(reference.keypoints), len(candidate.keypoints))
        return len(matches) / fewest if fewest else 0.0
    raise ValueError(f"Score must be one of {', '.join(SCORES)}.")


def rank_images(
    reference: Features,
    paths: list[Path],
    settings: LookupSettings,
    progress: Callable[[int, int, Path], None] | None = None,
) -> tuple[list[Candidate], list[Path]]:
    """Match every image against ``reference``; return candidates best first and unreadable paths."""
    candidates: list[Candidate] = []
    failed: list[Path] = []
    for index, path in enumerate(paths):
        if progress is not None:
            progress(index, len(paths), path)
        try:
            with Image.open(path) as opened:
                image = opened.copy()
        except (OSError, UnidentifiedImageError):
            failed.append(path)
            continue
        features = extract_features(image, settings.matching)
        matches = match_features(reference, features, settings.matching)
        score = score_matches(settings.score, matches, reference, features)
        candidates.append(Candidate(path, features, matches, score))
    # Stable sort: ties keep the alphabetical order of the paths
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates, failed


def format_score(score: str, candidate: Candidate) -> str:
    if score == "normalized":
        return f"{candidate.score:.3f} ({len(candidate.matches)} matches)"
    return f"{len(candidate.matches)} matches"


class LookupDialog(MatchingDialog):
    """Matching options plus how the folder is searched and ranked."""

    def __init__(self, parent: tk.Misc, title: str, folder: Path) -> None:
        self.folder = folder
        super().__init__(parent, title, defaults=LookupSettings().matching)

    def extra_body(self, master: tk.Frame, row: int) -> None:
        self.score = tk.StringVar(value=LookupSettings.score)
        self.recursive = tk.BooleanVar(value=LookupSettings.recursive)
        lookup = ttk.LabelFrame(master, text=f"Lookup in {self.folder.name or self.folder}")
        lookup.grid(row=row, column=0, columnspan=2, sticky="we", pady=(6, 0))
        ttk.Checkbutton(lookup, text="Search subfolders too", variable=self.recursive).grid(
            row=0, column=0, sticky="w", padx=6
        )
        ttk.Label(lookup, text="Rank candidates by:").grid(row=1, column=0, sticky="w", padx=6)
        for offset, (key, text) in enumerate(SCORES.items(), start=2):
            ttk.Radiobutton(lookup, text=text, value=key, variable=self.score).grid(
                row=offset, column=0, sticky="w", padx=(18, 6)
            )

    def apply(self) -> None:
        self.result = LookupSettings(matching=self._settings, score=self.score.get(), recursive=self.recursive.get())


class ImageLookupTool(ForensicsTool):
    tool_id = "image_lookup"
    title = "Closest image in folder"
    category = "Set 5: Keypoints & Matching"
    description = (
        "Find the image in a folder that best matches the current image (most ORB/SIFT matches, "
        "optionally RANSAC-verified) and draw its matches."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by main window

        directory = filedialog.askdirectory(title="Select folder to search", parent=parent, mustexist=True)
        if not directory:
            return None
        folder = Path(directory)

        settings = LookupDialog(parent, "Closest image in folder", folder).result
        if settings is None:
            return None

        # Skip the file the current image was loaded from, otherwise it always wins
        paths = list_images(folder, settings.recursive, exclude=document.path)
        if not paths:
            return ToolResult(message=f"No images found in {folder}.")

        # Matching a whole folder takes a few seconds: show progress in the window title
        window = parent.winfo_toplevel()
        previous_title = window.title()

        def progress(index: int, total: int, path: Path) -> None:
            window.title(f"Matching {index + 1}/{total}: {path.name}")
            window.update_idletasks()

        window.configure(cursor="watch")
        try:
            reference = extract_features(document.current, settings.matching)
            candidates, failed = rank_images(reference, paths, settings, progress)
        finally:
            window.configure(cursor="")
            window.title(previous_title)

        if not candidates:
            return ToolResult(message=f"None of the {len(paths)} file(s) in {folder} could be opened.")
        best = candidates[0]
        best_name = best.path.relative_to(folder).as_posix()

        details: dict[str, Any] = {
            "Operation": "Closest image lookup",
            "Library": "skimage.feature.match_descriptors"
            + (", skimage.measure.ransac" if settings.matching.use_ransac else ""),
            "Folder": str(folder),
            "Images searched": len(candidates),
            **describe_settings(settings.matching),
            "Ranked by": SCORES[settings.score],
            "Reference keypoints": len(reference.keypoints),
            "Best match": best_name,
        }
        for rank, candidate in enumerate(candidates[:MAX_RANKED], start=1):
            details[f"#{rank} {candidate.path.relative_to(folder).as_posix()}"] = format_score(settings.score, candidate)
        if failed:
            details["Unreadable files"] = ", ".join(path.name for path in failed)

        return ToolResult(
            image=draw_matches(reference, best.features, best.matches, settings.matching.show_keypoints),
            message=f"Best: {best_name} ({format_score(settings.score, best)}) out of {len(candidates)} image(s).",
            details=details,
        )


__all__ = [
    "Candidate",
    "ImageLookupTool",
    "LookupDialog",
    "LookupSettings",
    "SCORES",
    "list_images",
    "rank_images",
    "score_matches",
]
