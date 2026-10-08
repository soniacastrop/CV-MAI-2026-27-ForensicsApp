from pathlib import Path
import unittest

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.image_info import ImageInfoTool
from forensics_app.tools.registry import ToolRegistry
from forensics_app.tools.rotate import Rotate90Tool


class ToolTests(unittest.TestCase):
    def test_grayscale_returns_image_without_mutating_document(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "red")
        result = GrayscaleTool().run(None, document)  # parent is unused by this tool
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(document.current.mode, "RGB")

    def test_rotate_90_turns_image_clockwise(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "black")
        document.current.putpixel((0, 0), (255, 0, 0))  # top-left corner
        result = Rotate90Tool().run(None, document)  # parent is unused by this tool
        self.assertEqual(result.image.size, (3, 4))
        self.assertEqual(result.image.getpixel((2, 0)), (255, 0, 0))  # now top-right
        self.assertEqual(document.current.size, (4, 3))

    def test_image_info_reports_properties_without_image(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGBA", (7, 5))
        document.path = Path("evidence.jpeg")
        result = ImageInfoTool().run(None, document)
        self.assertIsNone(result.image)
        self.assertEqual(result.details["Width"], "7 px")
        self.assertEqual(result.details["Height"], "5 px")
        self.assertEqual(result.details["Mode"], "RGBA")
        self.assertEqual(result.details["Format"], "JPEG")

    def test_image_info_unknown_format_without_path(self) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (1, 1))
        self.assertEqual(ImageInfoTool().run(None, document).details["Format"], "Unknown")


class RegistryTests(unittest.TestCase):
    def test_registry_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            ToolRegistry([GrayscaleTool(), GrayscaleTool()])

    def test_categories_keep_registration_order(self) -> None:
        registry = ToolRegistry([GrayscaleTool(), ImageInfoTool(), Rotate90Tool()])
        names = [name for name, _tools in registry.categories()]
        self.assertEqual(len(names), len(set(names)))
        all_grouped = [tool for _name, tools in registry.categories() for tool in tools]
        self.assertEqual(set(all_grouped), set(registry.all()))

    def test_application_registry_is_complete_and_described(self) -> None:
        registry = build_tool_registry()
        ids = {tool.tool_id for tool in registry.all()}
        for expected in (
            "histogram",
            "contrast_stretching",
            "histogram_equalization",
            "histogram_equalization_clahe",
            "histogram_matching",
            "convolution",
            "skimage_filters",
            "canny",
        ):
            self.assertIn(expected, ids)
        for tool in registry.all():
            with self.subTest(tool=tool.tool_id):
                self.assertNotEqual(tool.title, "Unnamed tool")
                self.assertTrue(tool.description)

    def test_set2_tools_are_grouped_together(self) -> None:
        set2 = dict(build_tool_registry().categories())["Set 2: Image Processing"]
        self.assertEqual(
            [tool.tool_id for tool in set2],
            [
                "histogram",
                "contrast_stretching",
                "histogram_equalization",
                "histogram_equalization_clahe",
                "histogram_matching",
            ],
        )


if __name__ == "__main__":
    unittest.main()
