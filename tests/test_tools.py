import unittest

from PIL import Image

from forensics_app.core import ImageDocument
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
