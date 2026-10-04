"""Rotate the working image by 90 degrees."""

from __future__ import annotations

import tkinter as tk

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class Rotate90Tool(ForensicsTool):
    tool_id = "rotate_90"
    title = "Rotate 90° clockwise"
    category = "Starter tools"
    description = "Rotate the working image 90 degrees clockwise (lossless)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by the main window
        output = document.current.transpose(Image.Transpose.ROTATE_270)
        return ToolResult(
            image=output,
            message="Rotated the image 90° clockwise.",
            details={"Operation": "Rotate 90° clockwise", "Output size": f"{output.width} x {output.height}"},
        )
