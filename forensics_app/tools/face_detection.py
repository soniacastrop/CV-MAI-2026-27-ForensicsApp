"""Set 6.1: Face detection with OpenCV's YuNet CNN (not Haar), on the current image or a webcam stream."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any
import urllib.request

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialog_fields import read_number, spin_row
from .modes import split_alpha

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/" + MODEL_NAME
)
# Where the model is looked for (first existing wins); downloads go to the first entry
MODEL_LOCATIONS = (
    PROJECT_ROOT / "models" / MODEL_NAME,
    PROJECT_ROOT / "images" / "Set6_materials" / MODEL_NAME,
)
MODEL_TYPES = [("ONNX model", "*.onnx"), ("All files", "*.*")]

SOURCES = {
    "image": "Current image",
    "webcam": "Webcam stream (live)",
}
OUTPUTS = {
    "side": "Original and detection side by side (as in the slide)",
    "annotate": "Boxes, scores and the 5 landmarks",
    "crop": "Crop of the most confident face (e.g. to feed Face recognition)",
}
LANDMARK_NAMES = ("Right eye", "Left eye", "Nose", "Right mouth corner", "Left mouth corner")

SCORE_RANGE = (0.0, 1.0)
NMS_RANGE = (0.0, 1.0)
TOP_K_RANGE = (1, 10000)
CAMERA_RANGE = (0, 10)
CROP_MARGIN_RANGE = (0.0, 1.0)

BOX_COLOUR = (0, 255, 0)  # as in the notebook: green box and score, blue landmarks
LANDMARK_COLOUR = (0, 0, 255)
WEBCAM_DELAY_MS = 15  # between frames; detection itself takes most of the frame time


@dataclass(frozen=True)
class FaceDetectionSettings:
    # Defaults from the tutorial notebook: FaceDetectorYN.create(model, "", size, 0.6, 0.3, 5000)
    model_path: str = ""
    score_threshold: float = 0.6
    nms_threshold: float = 0.3
    top_k: int = 5000
    source: str = "image"
    output: str = "side"
    show_landmarks: bool = True
    crop_margin: float = 0.2  # crop: extra border around the box, as a fraction of its size
    camera: int = 0
    mirror: bool = True  # webcam: show it like a mirror


@dataclass(frozen=True)
class Face:
    """One YuNet detection in pixel coordinates."""

    box: tuple[float, float, float, float]  # x, y, width, height
    landmarks: np.ndarray  # (5, 2) x, y in LANDMARK_NAMES order
    score: float


def default_model_path() -> Path | None:
    return next((path for path in MODEL_LOCATIONS if path.is_file()), None)


def download_model(target: Path = MODEL_LOCATIONS[0]) -> Path:
    """Fetch the pretrained YuNet weights from the OpenCV model zoo (as the notebook does)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, str(target))
    if target.stat().st_size < 100_000:
        target.unlink()
        raise RuntimeError("YuNet download failed or the file is incomplete.")
    return target


def create_detector(settings: FaceDetectionSettings, size: tuple[int, int] = (320, 320)) -> Any:
    """``cv2.FaceDetectorYN``; ``size`` is the (width, height) of the images it will see."""
    path = Path(settings.model_path)
    if not path.is_file():
        raise ValueError(f"YuNet model not found: {path}")
    return cv2.FaceDetectorYN.create(
        str(path), "", size, settings.score_threshold, settings.nms_threshold, settings.top_k
    )


def to_bgr(image: Image.Image) -> np.ndarray:
    """OpenCV's 3-channel BGR uint8 layout; alpha is dropped and gray is repeated."""
    base, _alpha = split_alpha(image)
    rgb = np.asarray(base.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def detect_faces(detector: Any, bgr: np.ndarray) -> list[Face]:
    """Run YuNet on one BGR frame; faces sorted by score, best first."""
    height, width = bgr.shape[:2]
    detector.setInputSize((width, height))
    _retval, raw = detector.detect(bgr)
    if raw is None:
        return []
    faces = [
        Face(box=tuple(float(v) for v in row[:4]), landmarks=row[4:14].reshape(5, 2).astype(np.float64), score=float(row[14]))
        for row in raw
    ]
    return sorted(faces, key=lambda face: face.score, reverse=True)


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def draw_faces(image: Image.Image, faces: list[Face], show_landmarks: bool = True) -> Image.Image:
    """Green boxes labelled ``Face: score`` and blue landmark dots, scaled to the image size."""
    base, _alpha = split_alpha(image)
    canvas = base.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    side = max(canvas.size)
    width = max(2, round(side / 250))
    radius = max(2, round(side / 130))
    font = _font(max(12, round(side / 32)))
    for face in faces:
        x, y, w, h = face.box
        draw.rectangle((x, y, x + w, y + h), outline=BOX_COLOUR, width=width)
        label = f"Face: {face.score:.2f}"
        height = draw.textbbox((0, 0), label, font=font, anchor="lt")[3]
        # Above the box, with its bottom ("ld" = left/descender) clear of the line; inside if no room
        draw.text((x, max(height, y - 2 * width)), label, fill=BOX_COLOUR, font=font, anchor="ld")
        if show_landmarks:
            for px, py in face.landmarks:
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=LANDMARK_COLOUR)
    return canvas


def side_by_side(image: Image.Image, detected: Image.Image, title: str = "Original image") -> Image.Image:
    """The notebook's figure: original on the left, YuNet detections on the right."""
    figure = Figure(figsize=(12, 6), layout="constrained")
    for ax, picture, name in zip(
        figure.subplots(1, 2), (split_alpha(image)[0].convert("RGB"), detected),
        (title, f"YuNet Face Detection (OpenCV {cv2.__version__.split('.')[0]})"),
    ):
        ax.imshow(picture)
        ax.set_title(name)
        ax.axis("off")
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())


def crop_face(image: Image.Image, face: Face, margin: float) -> Image.Image:
    """The face box grown by ``margin`` of its size on each side, clipped to the image."""
    base, _alpha = split_alpha(image)
    x, y, w, h = face.box
    left = max(0, round(x - margin * w))
    top = max(0, round(y - margin * h))
    right = min(base.width, round(x + w + margin * w))
    bottom = min(base.height, round(y + h + margin * h))
    return base.crop((left, top, right, bottom))


def face_details(faces: list[Face], limit: int = 10) -> dict[str, Any]:
    """Results-panel rows: box, score and landmarks of the best ``limit`` faces."""
    details: dict[str, Any] = {}
    for index, face in enumerate(faces[:limit], start=1):
        x, y, w, h = (round(v) for v in face.box)
        details[f"Face {index}"] = f"score {face.score:.3f}, box ({x}, {y}) {w}×{h}"
        for name, (px, py) in zip(LANDMARK_NAMES, face.landmarks):
            details[f"  {name}"] = f"({round(px)}, {round(py)})"
    if len(faces) > limit:
        details["…"] = f"{len(faces) - limit} more face(s)"
    return details


class WebcamSession:
    """Grab, detect and draw webcam frames; kept free of Tk so it can be tested."""

    def __init__(self, settings: FaceDetectionSettings, capture: Any | None = None) -> None:
        self.settings = settings
        if capture is None:
            # DirectShow opens much faster than the default backend on Windows
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            capture = cv2.VideoCapture(settings.camera, backend)
        self.capture = capture
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open webcam {settings.camera}.")
        self.detector = create_detector(settings)
        self.frames = 0
        self.fps = 0.0
        self._last = time.perf_counter()
        self.last_frame: Image.Image | None = None
        self.last_faces: list[Face] = []

    def next_frame(self) -> Image.Image | None:
        """Annotated RGB frame, or ``None`` when the camera returns nothing."""
        ok, bgr = self.capture.read()
        if not ok or bgr is None:
            return None
        if self.settings.mirror:
            bgr = cv2.flip(bgr, 1)
        self.last_faces = detect_faces(self.detector, bgr)
        self.last_frame = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
        now = time.perf_counter()
        elapsed, self._last = now - self._last, now
        if elapsed > 0:  # smoothed so the number is readable
            self.fps = 1 / elapsed if self.frames == 0 else 0.9 * self.fps + 0.1 / elapsed
        self.frames += 1
        return draw_faces(self.last_frame, self.last_faces, self.settings.show_landmarks)

    def close(self) -> None:
        self.capture.release()


class WebcamWindow:
    """Modal window that shows the live detections; "Capture" returns the current frame."""

    def __init__(self, parent: tk.Misc, session: WebcamSession) -> None:
        self.session = session
        self.captured: Image.Image | None = None
        self._job: str | None = None
        self.window = tk.Toplevel(parent)
        self.window.title("YuNet face detection — webcam")
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.view = ttk.Label(self.window)
        self.view.pack(padx=6, pady=6)
        self.status = tk.StringVar(value="Starting camera…")
        ttk.Label(self.window, textvariable=self.status).pack(anchor="w", padx=6)
        buttons = ttk.Frame(self.window)
        buttons.pack(fill="x", padx=6, pady=6)
        ttk.Button(buttons, text="Capture frame", command=self.capture).pack(side="left")
        ttk.Button(buttons, text="Close", command=self.close).pack(side="right")
        self.window.bind("<space>", lambda _event: self.capture())
        self.window.bind("<Escape>", lambda _event: self.close())
        self._tick()

    def _tick(self) -> None:
        frame = self.session.next_frame()
        if frame is not None:
            self._photo = ImageTk.PhotoImage(frame)  # keep a reference or Tk shows nothing
            self.view.configure(image=self._photo)
            self.status.set(
                f"{len(self.session.last_faces)} face(s) · {self.session.fps:.1f} fps · "
                "Space = capture, Esc = close"
            )
        self._job = self.window.after(WEBCAM_DELAY_MS, self._tick)

    def capture(self) -> None:
        if self.session.last_frame is not None:
            self.captured = self.session.last_frame.copy()
        self.close()

    def close(self) -> None:
        if self._job is not None:
            self.window.after_cancel(self._job)
            self._job = None
        self.session.close()
        self.window.destroy()


class FaceDetectionDialog(simpledialog.Dialog):
    """Model file, YuNet thresholds, source (image / webcam) and output."""

    def __init__(self, parent: tk.Misc, has_image: bool) -> None:
        self.has_image = has_image
        super().__init__(parent, title="Face detection (YuNet)")

    def body(self, master: tk.Frame) -> tk.Widget:
        d = FaceDetectionSettings()
        found = default_model_path()
        self.model_path = tk.StringVar(value=str(found) if found else "")
        self.score = tk.StringVar(value=str(d.score_threshold))
        self.nms = tk.StringVar(value=str(d.nms_threshold))
        self.top_k = tk.StringVar(value=str(d.top_k))
        self.source = tk.StringVar(value="image" if self.has_image else "webcam")
        self.output = tk.StringVar(value=d.output)
        self.landmarks = tk.BooleanVar(value=d.show_landmarks)
        self.margin = tk.StringVar(value=str(d.crop_margin))
        self.camera = tk.StringVar(value=str(d.camera))
        self.mirror = tk.BooleanVar(value=d.mirror)

        model = ttk.LabelFrame(master, text="YuNet model (ONNX)")
        model.grid(row=0, column=0, columnspan=2, sticky="we")
        entry = ttk.Entry(model, textvariable=self.model_path, width=46)
        entry.grid(row=0, column=0, padx=6, pady=2)
        ttk.Button(model, text="Browse…", command=self._browse).grid(row=0, column=1, padx=(0, 6))
        ttk.Label(model, text="Missing? OK offers to download it from the OpenCV model zoo.").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=6
        )

        params = ttk.LabelFrame(master, text="Detector")
        params.grid(row=1, column=0, columnspan=2, sticky="we", pady=(6, 0))
        first = spin_row(params, 0, "Score threshold:", self.score, SCORE_RANGE, 0.05)
        spin_row(params, 1, "NMS threshold:", self.nms, NMS_RANGE, 0.05)
        spin_row(params, 2, "Max candidates (top k):", self.top_k, TOP_K_RANGE, 100)
        ttk.Checkbutton(params, text="Draw the 5 facial landmarks", variable=self.landmarks).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=6
        )

        source = ttk.LabelFrame(master, text="Source")
        source.grid(row=2, column=0, columnspan=2, sticky="we", pady=(6, 0))
        for row, (key, text) in enumerate(SOURCES.items()):
            button = ttk.Radiobutton(source, text=text, value=key, variable=self.source, command=self._toggle)
            button.grid(row=row, column=0, columnspan=2, sticky="w", padx=6)
            if key == "image" and not self.has_image:
                button.state(["disabled"])
        self.webcam_frame = ttk.Frame(source)
        self.webcam_frame.grid(row=len(SOURCES), column=0, columnspan=2, sticky="w", padx=(18, 6))
        spin_row(self.webcam_frame, 0, "Camera index:", self.camera, CAMERA_RANGE, 1)
        ttk.Checkbutton(self.webcam_frame, text="Mirror the picture", variable=self.mirror).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=6
        )

        output = ttk.LabelFrame(master, text="Output")
        output.grid(row=3, column=0, columnspan=2, sticky="we", pady=(6, 0))
        for row, (key, text) in enumerate(OUTPUTS.items()):
            ttk.Radiobutton(output, text=text, value=key, variable=self.output).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=6
            )
        spin_row(output, len(OUTPUTS), "Crop margin (fraction of box):", self.margin, CROP_MARGIN_RANGE, 0.05)

        self._toggle()
        return first

    def _browse(self) -> None:
        filename = filedialog.askopenfilename(title="YuNet model", filetypes=MODEL_TYPES, parent=self)
        if filename:
            self.model_path.set(filename)

    def _toggle(self) -> None:
        if self.source.get() == "webcam":
            self.webcam_frame.grid()
        else:
            self.webcam_frame.grid_remove()

    def validate(self) -> bool:
        try:
            settings = FaceDetectionSettings(
                model_path=self.model_path.get().strip(),
                score_threshold=read_number(self.score, "Score threshold", SCORE_RANGE),
                nms_threshold=read_number(self.nms, "NMS threshold", NMS_RANGE),
                top_k=read_number(self.top_k, "Max candidates", TOP_K_RANGE, int),
                source=self.source.get(),
                output=self.output.get(),
                show_landmarks=self.landmarks.get(),
                crop_margin=read_number(self.margin, "Crop margin", CROP_MARGIN_RANGE),
                camera=read_number(self.camera, "Camera index", CAMERA_RANGE, int),
                mirror=self.mirror.get(),
            )
        except ValueError as error:
            messagebox.showerror(self.title(), str(error), parent=self)
            return False
        if not settings.model_path or not Path(settings.model_path).is_file():
            if not messagebox.askyesno(
                self.title(),
                f"No YuNet model at '{settings.model_path or '—'}'.\n\n"
                f"Download {MODEL_NAME} (~230 KB) from the OpenCV model zoo into the models/ folder?",
                parent=self,
            ):
                return False
            try:
                path = download_model()
            except (OSError, RuntimeError) as error:
                messagebox.showerror(self.title(), f"Download failed: {error}", parent=self)
                return False
            self.model_path.set(str(path))
            settings = replace(settings, model_path=str(path))
        self._settings = settings
        return True

    def apply(self) -> None:
        self.result = self._settings


def describe_settings(settings: FaceDetectionSettings) -> dict[str, Any]:
    return {
        "Operation": "Face detection",
        "Library": "cv2.FaceDetectorYN (YuNet CNN)",
        "Model": Path(settings.model_path).name,
        "Score threshold": f"{settings.score_threshold:g}",
        "NMS threshold": f"{settings.nms_threshold:g}",
        "Top k": settings.top_k,
        "Source": SOURCES[settings.source],
    }


class FaceDetectionTool(ForensicsTool):
    tool_id = "face_detection"
    title = "Face detection (YuNet)"
    category = "Set 6: Faces"
    description = (
        "Detect faces with OpenCV's YuNet CNN: boxes, confidence and 5 landmarks, on the image or a live "
        "webcam stream; optionally crop the best face."
    )
    requires_image = False  # the webcam works without an open image

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        settings = FaceDetectionDialog(parent, has_image=document.current is not None).result
        if settings is None:
            return None
        if settings.source == "webcam":
            return self._run_webcam(parent, settings)
        if document.current is None:
            raise ValueError("Open an image first, or choose the webcam as source.")
        return self._run_image(document.current, settings)

    def _run_image(self, image: Image.Image, settings: FaceDetectionSettings) -> ToolResult:
        detector = create_detector(settings, image.size)
        faces = detect_faces(detector, to_bgr(image))
        return self._result(image, faces, settings)

    def _run_webcam(self, parent: tk.Misc, settings: FaceDetectionSettings) -> ToolResult:
        session = WebcamSession(settings)
        window = WebcamWindow(parent, session)
        parent.wait_window(window.window)
        if window.captured is None:
            return ToolResult(
                message=f"Webcam closed after {session.frames} frame(s) ({session.fps:.1f} fps).",
                details={**describe_settings(settings), "Frames processed": session.frames},
            )
        result = self._result(window.captured, session.last_faces, settings)
        details = {**result.details, "Frames processed": session.frames, "Speed": f"{session.fps:.1f} fps"}
        return ToolResult(message="Captured webcam frame: " + result.message, image=result.image, details=details)

    @staticmethod
    def _result(image: Image.Image, faces: list[Face], settings: FaceDetectionSettings) -> ToolResult:
        details = {**describe_settings(settings), "Faces found": len(faces), **face_details(faces)}
        if settings.output == "crop":
            if not faces:
                return ToolResult(message="No face detected: nothing to crop.", details=details)
            crop = crop_face(image, faces[0], settings.crop_margin)
            details["Crop size"] = f"{crop.width} × {crop.height}"
            return ToolResult(
                image=crop, message=f"Cropped the best of {len(faces)} face(s) (score {faces[0].score:.2f}).", details=details
            )
        message = f"Detected {len(faces)} face(s)." if faces else "No faces detected."
        detected = draw_faces(image, faces, settings.show_landmarks)
        if settings.output == "side":
            detected = side_by_side(image, detected)
        return ToolResult(image=detected, message=message, details=details)


__all__ = [
    "Face",
    "FaceDetectionDialog",
    "FaceDetectionSettings",
    "FaceDetectionTool",
    "WebcamSession",
    "WebcamWindow",
    "create_detector",
    "crop_face",
    "default_model_path",
    "detect_faces",
    "draw_faces",
    "side_by_side",
    "to_bgr",
]
