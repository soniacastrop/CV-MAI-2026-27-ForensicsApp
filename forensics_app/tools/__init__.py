"""Register course functionality here so it appears in the sidebar."""

from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .contrast_stretching import ContrastStretchingTool
from .histogram import HistogramTool
from .canny import CannyTool
from .skimage_filters import SkimageFiltersTool
from .convolution import ConvolutionTool

def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            HistogramTool(),
            ContrastStretchingTool(),
            ConvolutionTool(),
        SkimageFiltersTool(),
        CannyTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
