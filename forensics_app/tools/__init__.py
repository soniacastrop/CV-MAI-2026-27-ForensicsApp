"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .mask import MaskTool
from .registry import ToolRegistry
from .rotate import Rotate90Tool
from .threshold import ThresholdTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ChannelSplitTool(),
            ChannelSwapTool(),
            ThresholdTool(),
            MaskTool(),
            Rotate90Tool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
