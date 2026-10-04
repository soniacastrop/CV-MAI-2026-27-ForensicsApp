"""Apply a mask image: keep the current image where the mask is white, fill the rest.

Also holds the helpers shared with the threshold tool: mask compositing and the
dialog base class that previews the mask and the result side by side.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from PIL import Image, ImageChops, ImageTk, UnidentifiedImageError

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

FILL_BLACK = "black"
FILL_IMAGE = "image"

PREVIEW_SIZE = (220, 220)  # per panel; the dialog shows the mask and the result side by side
IMAGE_FILETYPES = [("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif"), ("All files", "*.*")]


def binarize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Turn any image into a 0/255 ``L`` mask (bright = selected), resized to ``size`` if needed.

    Resizing happens before binarizing so upscaled edges stay smooth, not blocky.
    """
    gray = mask.convert("L")
    if gray.size != size:
        gray = gray.resize(size, Image.Resampling.BILINEAR)
    return gray.point([255 if value >= 128 else 0 for value in range(256)])


def load_image_file(path: str | Path) -> Image.Image:
    """Open an image and detach it from the file handle."""
    with Image.open(path) as image:
        image.load()
        return image.copy()


def apply_mask(image: Image.Image, mask: Image.Image, fill: Image.Image | None = None) -> Image.Image:
    """Keep pixels where the 0/255 ``L`` mask is white; take the rest from ``fill``.

    ``fill`` is resized to the image size; without it the rest is painted opaque black.
    """
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    base = image.convert("RGBA" if has_alpha else "RGB")
    if fill is None:
        background = Image.new(base.mode, base.size, "black")
    else:
        background = fill.convert(base.mode)
        if background.size != base.size:
            background = background.resize(base.size, Image.Resampling.LANCZOS)
    return Image.composite(base, background, mask)


def selected_fraction(mask: Image.Image) -> float:
    return mask.histogram()[255] / (mask.width * mask.height)


def selected_pixels_detail(mask: Image.Image) -> str:
    total = mask.width * mask.height
    return f"{mask.histogram()[255]} / {total} ({selected_fraction(mask):.2%})"


def resize_note(source: Image.Image, target: Image.Image) -> str | None:
    if source.size == target.size:
        return None
    return f"{source.width}×{source.height} → {target.width}×{target.height}"


class MaskPreviewDialog(simpledialog.Dialog):
    """Base dialog: subclass controls on the left, live mask and result previews on the right.

    Subclasses implement ``build_controls``, ``preview_mask`` and ``read_result``; the
    latter two raise ``ValueError`` while the input is incomplete or invalid.
    """

    def __init__(self, parent: tk.Misc, image: Image.Image, title: str | None = None) -> None:
        # The preview works on a thumbnail so changing a value stays responsive.
        self._preview_image = image.copy()
        self._preview_image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        self._loaded: dict[str, Image.Image | None] = {}  # path -> image (None if unreadable)
        self._fill_previews: dict[int, Image.Image] = {}  # id(fill image) -> preview-sized copy
        super().__init__(parent, title=title)

    # --- subclass hooks -------------------------------------------------------------
    def build_controls(self, master: ttk.Frame) -> tk.Widget:
        raise NotImplementedError

    def preview_mask(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError

    def preview_fill(self) -> Image.Image | None:
        return None  # black

    def read_result(self) -> Any:
        raise NotImplementedError

    # --- shared helpers -------------------------------------------------------------
    def watch(self, *variables: tk.Variable) -> None:
        for variable in variables:
            variable.trace_add("write", lambda *_: self.refresh_preview())

    def path_row(self, master: ttk.Frame, row: int, variable: tk.StringVar, title: str, indent: int = 20):
        frame = ttk.Frame(master, padding=(indent, 2, 0, 6))
        frame.grid(row=row, column=0, columnspan=4, sticky="w")
        entry = ttk.Entry(frame, textvariable=variable, width=32)
        entry.grid(row=0, column=0, sticky="w")
        browse = ttk.Button(frame, text="Browse…", command=lambda: self._browse(variable, title))
        browse.grid(row=0, column=1, padx=(6, 0))
        return entry, browse

    def _browse(self, variable: tk.StringVar, title: str) -> None:
        path = filedialog.askopenfilename(parent=self, title=title, filetypes=IMAGE_FILETYPES)
        if path:
            variable.set(path)

    def load(self, path: str, what: str) -> Image.Image:
        """Load ``path`` once per distinct path; raise ``ValueError`` if unreadable."""
        if path not in self._loaded:
            try:
                self._loaded[path] = load_image_file(path)
            except (OSError, UnidentifiedImageError):
                self._loaded[path] = None
        image = self._loaded[path]
        if image is None:
            raise ValueError(f"Cannot open the {what}:\n{path}")
        return image

    # --- dialog plumbing ------------------------------------------------------------
    def body(self, master: tk.Frame) -> tk.Widget:
        controls = ttk.Frame(master)
        controls.grid(row=0, column=0, sticky="nw")
        focus = self.build_controls(controls)

        previews = ttk.Frame(master)
        previews.grid(row=0, column=1, sticky="n", padx=(12, 0))
        mask_frame = ttk.LabelFrame(previews, text="Mask (white = kept)", padding=6)
        mask_frame.grid(row=0, column=0, sticky="n")
        self._mask_label = ttk.Label(mask_frame, anchor="center", wraplength=PREVIEW_SIZE[0])
        self._mask_label.pack()
        self._mask_stats = ttk.Label(mask_frame, text="")
        self._mask_stats.pack(pady=(4, 0))
        result_frame = ttk.LabelFrame(previews, text="Result", padding=6)
        result_frame.grid(row=0, column=1, sticky="n", padx=(8, 0))
        self._result_label = ttk.Label(result_frame, anchor="center", wraplength=PREVIEW_SIZE[0])
        self._result_label.pack()

        self.refresh_preview()
        return focus

    def _show(self, label: ttk.Label, attribute: str, image: Image.Image | None, text: str = "") -> None:
        photo = ImageTk.PhotoImage(image) if image is not None else None
        setattr(self, attribute, photo)  # keep a reference or Tk shows nothing
        label.configure(image=photo or "", text=text)

    def refresh_preview(self) -> None:
        if not hasattr(self, "_result_label"):
            return  # a trace fired while the controls were still being built
        try:
            mask = self.preview_mask(self._preview_image)
        except ValueError as error:
            message = str(error).splitlines()[0]
            self._show(self._mask_label, "_mask_photo", None, message)
            self._show(self._result_label, "_result_photo", None, message)
            self._mask_stats.configure(text="")
            return
        self._show(self._mask_label, "_mask_photo", mask)
        self._mask_stats.configure(text=f"≈ {selected_fraction(mask):.1%} of pixels kept")
        try:
            fill = self.preview_fill()
            if fill is not None:
                # Shrink a large fill photo once, not on every change.
                if id(fill) not in self._fill_previews:
                    self._fill_previews[id(fill)] = fill.resize(self._preview_image.size, Image.Resampling.LANCZOS)
                fill = self._fill_previews[id(fill)]
        except ValueError as error:
            self._show(self._result_label, "_result_photo", None, str(error).splitlines()[0])
            return
        self._show(self._result_label, "_result_photo", apply_mask(self._preview_image, mask, fill))

    def validate(self) -> bool:
        try:
            self._result = self.read_result()
        except ValueError as error:
            messagebox.showerror(self.title() or "Mask", str(error), parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self._result


@dataclass(frozen=True)
class MaskSettings:
    mask_path: str
    mask_image: Image.Image
    invert: bool = False
    fill_path: str | None = None  # None means fill with black
    fill_image: Image.Image | None = None


def build_mask(settings: MaskSettings, size: tuple[int, int]) -> Image.Image:
    """Return the 0/255 ``L`` mask of the given size described by ``settings``."""
    mask = binarize_mask(settings.mask_image, size)
    return ImageChops.invert(mask) if settings.invert else mask


class MaskDialog(MaskPreviewDialog):
    """Ask for the mask image and what fills its black areas."""

    def build_controls(self, master: ttk.Frame) -> tk.Widget:
        self.mask_path = tk.StringVar()
        self.invert = tk.BooleanVar(value=False)
        self.fill = tk.StringVar(value=FILL_BLACK)
        self.fill_path = tk.StringVar()

        ttk.Label(master, text="Mask image (white = keep current image):").grid(row=0, column=0, sticky="w")
        mask_entry, _ = self.path_row(master, 1, self.mask_path, "Choose mask image")
        ttk.Checkbutton(master, text="Invert mask", variable=self.invert).grid(
            row=2, column=0, sticky="w", pady=(4, 8)
        )
        ttk.Label(master, text="Fill the black areas of the mask with:").grid(row=3, column=0, sticky="w")
        ttk.Radiobutton(master, text="Black", variable=self.fill, value=FILL_BLACK, command=self._sync).grid(
            row=4, column=0, sticky="w", padx=(20, 0)
        )
        ttk.Radiobutton(
            master, text="Another image (resized to fit)", variable=self.fill, value=FILL_IMAGE, command=self._sync
        ).grid(row=5, column=0, sticky="w", padx=(20, 0))
        self._fill_widgets = self.path_row(master, 6, self.fill_path, "Choose fill image", indent=40)

        self.watch(self.mask_path, self.invert, self.fill, self.fill_path)
        self._sync()
        return mask_entry

    def _sync(self) -> None:
        for widget in self._fill_widgets:
            widget.configure(state="normal" if self.fill.get() == FILL_IMAGE else "disabled")

    def _mask_settings(self) -> MaskSettings:
        path = self.mask_path.get().strip()
        if not path:
            raise ValueError("Choose a mask image.")
        return MaskSettings(mask_path=path, mask_image=self.load(path, "mask image"), invert=self.invert.get())

    def preview_mask(self, image: Image.Image) -> Image.Image:
        return build_mask(self._mask_settings(), image.size)

    def preview_fill(self) -> Image.Image | None:
        if self.fill.get() != FILL_IMAGE:
            return None
        path = self.fill_path.get().strip()
        if not path:
            raise ValueError("Choose the image that fills the black areas.")
        return self.load(path, "fill image")

    def read_result(self) -> MaskSettings:
        settings = self._mask_settings()
        fill = self.preview_fill()
        if fill is None:
            return settings
        return MaskSettings(
            mask_path=settings.mask_path,
            mask_image=settings.mask_image,
            invert=settings.invert,
            fill_path=self.fill_path.get().strip(),
            fill_image=fill,
        )


class MaskTool(ForensicsTool):
    tool_id = "mask"
    title = "Apply mask"
    category = "Set2"
    description = "Keep the image where a mask is white; fill the rest with black or another image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window
        image = document.current
        settings = MaskDialog(parent, image, title="Apply mask").result
        if settings is None:
            return None  # user cancelled

        mask = build_mask(settings, image.size)
        output = apply_mask(image, mask, settings.fill_image)
        mask_name = Path(settings.mask_path).name
        fill = "Black" if settings.fill_image is None else f"Image {Path(settings.fill_path).name}"

        details = {
            "Operation": "Apply mask",
            "Mask": f"NOT {mask_name}" if settings.invert else mask_name,
            "Rest filled with": fill,
            "Kept pixels": selected_pixels_detail(mask),
        }
        if note := resize_note(settings.mask_image, image):
            details["Mask resized"] = note
        if settings.fill_image is not None and (note := resize_note(settings.fill_image, image)):
            details["Fill image resized"] = note
        details["Output mode"] = output.mode
        return ToolResult(
            image=output,
            message=f"Applied mask {mask_name}: {selected_fraction(mask):.1%} kept, rest filled with {fill}.",
            details=details,
        )
