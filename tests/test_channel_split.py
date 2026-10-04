import unittest

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.channel_split import ChannelSplitTool, plot_channels, split_channels


class SplitChannelsTests(unittest.TestCase):
    def test_returns_one_band_per_channel(self) -> None:
        channels = split_channels(Image.new("RGB", (4, 3), (255, 0, 128)))
        self.assertEqual([name for name, _ in channels], ["R", "G", "B"])
        self.assertEqual([band.getpixel((0, 0)) for _, band in channels], [255, 0, 128])
        self.assertTrue(all(band.mode == "L" for _, band in channels))

    def test_grayscale_image_has_single_channel(self) -> None:
        channels = split_channels(Image.new("L", (4, 3), 77))
        self.assertEqual([name for name, _ in channels], ["L"])

    def test_converts_palette_images(self) -> None:
        channels = split_channels(Image.new("RGB", (4, 3), "red").convert("P"))
        self.assertEqual([name for name, _ in channels], ["R", "G", "B"])

    def test_drops_fully_opaque_alpha(self) -> None:
        channels = split_channels(Image.new("RGBA", (4, 3), (255, 0, 0, 255)))
        self.assertEqual([name for name, _ in channels], ["R", "G", "B"])

    def test_keeps_alpha_when_some_pixel_is_transparent(self) -> None:
        image = Image.new("RGBA", (4, 3), (255, 0, 0, 255))
        image.putpixel((0, 0), (255, 0, 0, 0))
        channels = split_channels(image)
        self.assertEqual([name for name, _ in channels], ["R", "G", "B", "A"])

    def test_does_not_mutate_input(self) -> None:
        image = Image.new("RGBA", (4, 3), (1, 2, 3, 255))
        split_channels(image)
        self.assertEqual(image.mode, "RGBA")
        self.assertEqual(image.getpixel((0, 0)), (1, 2, 3, 255))


class PlotChannelsTests(unittest.TestCase):
    def test_plot_is_rgb_and_wider_with_more_channels(self) -> None:
        one = plot_channels(split_channels(Image.new("L", (4, 3))))
        three = plot_channels(split_channels(Image.new("RGB", (4, 3))))
        self.assertEqual(three.mode, "RGB")
        self.assertGreater(three.width, one.width)

    def test_large_channels_are_downscaled_to_panel_size(self) -> None:
        plot = plot_channels(split_channels(Image.new("RGB", (2000, 1000))))
        self.assertLess(plot.height, 1000)


class ChannelSplitToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        self.document.current = Image.new("RGB", (4, 3), "red")
        self.tool = ChannelSplitTool()

    def test_returns_plot_and_details_without_mutating_document(self) -> None:
        result = self.tool.run(None, self.document)  # parent is unused by this tool
        self.assertIsNotNone(result.image)
        self.assertEqual(result.details["Channels"], "R, G, B")
        self.assertEqual(result.details["Mode"], "RGB")
        self.assertEqual(self.document.current.size, (4, 3))

    def test_does_not_resplit_its_own_plot(self) -> None:
        first = self.tool.run(None, self.document)
        self.document.apply(first.image)
        plot_size = self.document.current.size
        second = self.tool.run(None, self.document)
        self.assertIsNone(second.image)
        self.assertEqual(second.details, first.details)
        self.assertEqual(self.document.current.size, plot_size)

    def test_splits_again_after_undo(self) -> None:
        self.document.apply(self.tool.run(None, self.document).image)
        self.document.undo()
        self.assertIsNotNone(self.tool.run(None, self.document).image)


if __name__ == "__main__":
    unittest.main()
