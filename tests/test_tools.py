import unittest

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.channel_split import ChannelSplitTool, split_channels
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.registry import ToolRegistry
from forensics_app.tools.rotate import Rotate90Tool


class ToolTests(unittest.TestCase):
    def test_grayscale_returns_image_without_mutating_document(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "red")
        result = GrayscaleTool().run(None, document)  # parent is unused by this tool
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(document.current.mode, "RGB")

    def test_split_channels_returns_one_band_per_channel(self) -> None:
        channels = split_channels(Image.new("RGB", (4, 3), (255, 0, 128)))
        self.assertEqual([name for name, _ in channels], ["R", "G", "B"])
        self.assertEqual([band.getpixel((0, 0)) for _, band in channels], [255, 0, 128])
        self.assertTrue(all(band.mode == "L" for _, band in channels))

    def test_split_channels_converts_palette_images(self) -> None:
        channels = split_channels(Image.new("RGB", (4, 3), "red").convert("P"))
        self.assertEqual([name for name, _ in channels], ["R", "G", "B"])

    def test_split_channels_drops_fully_opaque_alpha(self) -> None:
        opaque = split_channels(Image.new("RGBA", (4, 3), (255, 0, 0, 255)))
        self.assertEqual([name for name, _ in opaque], ["R", "G", "B"])
        transparent = split_channels(Image.new("RGBA", (4, 3), (255, 0, 0, 128)))
        self.assertEqual([name for name, _ in transparent], ["R", "G", "B", "A"])

    def test_channel_split_does_not_resplit_its_own_plot(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "red")
        tool = ChannelSplitTool()
        first = tool.run(None, document)
        document.apply(first.image)
        plot_size = document.current.size
        second = tool.run(None, document)
        self.assertIsNone(second.image)
        self.assertEqual(second.details, first.details)
        self.assertEqual(document.current.size, plot_size)

    def test_rotate_90_turns_image_clockwise(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "black")
        document.current.putpixel((0, 0), (255, 0, 0))  # top-left corner
        result = Rotate90Tool().run(None, document)  # parent is unused by this tool
        self.assertEqual(result.image.size, (3, 4))
        self.assertEqual(result.image.getpixel((2, 0)), (255, 0, 0))  # now top-right
        self.assertEqual(document.current.size, (4, 3))

    def test_registry_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            ToolRegistry([GrayscaleTool(), GrayscaleTool()])


if __name__ == "__main__":
    unittest.main()
