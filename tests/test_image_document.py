from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from forensics_app.core import ImageDocument


class ImageDocumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.source = Path(self.temporary_directory.name) / "source.png"
        Image.new("RGB", (8, 6), "red").save(self.source)
        self.document = ImageDocument()
        self.document.load(self.source)

    def test_load_keeps_original_and_current_images(self) -> None:
        self.assertTrue(self.document.is_loaded)
        self.assertEqual(self.document.current.size, (8, 6))
        self.assertEqual(self.document.original.getpixel((0, 0)), (255, 0, 0))

    def test_apply_undo_and_redo(self) -> None:
        self.document.apply(Image.new("RGB", (8, 6), "blue"))
        self.assertEqual(self.document.current.getpixel((0, 0)), (0, 0, 255))
        self.assertTrue(self.document.undo())
        self.assertEqual(self.document.current.getpixel((0, 0)), (255, 0, 0))
        self.assertTrue(self.document.redo())
        self.assertEqual(self.document.current.getpixel((0, 0)), (0, 0, 255))

    def test_reset_is_undoable(self) -> None:
        self.document.apply(Image.new("RGB", (8, 6), "blue"))
        self.assertTrue(self.document.reset())
        self.assertEqual(self.document.current.getpixel((0, 0)), (255, 0, 0))
        self.assertTrue(self.document.undo())
        self.assertEqual(self.document.current.getpixel((0, 0)), (0, 0, 255))

    def test_save_writes_current_image(self) -> None:
        target = Path(self.temporary_directory.name) / "result.png"
        self.document.save(target)
        self.assertTrue(target.exists())

    def test_new_edit_clears_redo_history(self) -> None:
        self.document.apply(Image.new("RGB", (8, 6), "blue"))
        self.document.undo()
        self.document.apply(Image.new("RGB", (8, 6), "green"))
        self.assertFalse(self.document.can_redo)
        self.assertFalse(self.document.redo())

    def test_undo_and_redo_report_false_when_history_is_empty(self) -> None:
        self.assertFalse(self.document.undo())
        self.assertFalse(self.document.redo())
        self.assertFalse(self.document.is_modified)

    def test_apply_stores_a_copy(self) -> None:
        result = Image.new("RGB", (8, 6), "blue")
        self.document.apply(result)
        result.putpixel((0, 0), (0, 255, 0))
        self.assertEqual(self.document.current.getpixel((0, 0)), (0, 0, 255))

    def test_loading_clears_history(self) -> None:
        self.document.apply(Image.new("RGB", (8, 6), "blue"))
        self.document.load(self.source)
        self.assertFalse(self.document.can_undo)
        self.assertEqual(self.document.path, self.source)

    def test_operations_without_image(self) -> None:
        empty = ImageDocument()
        self.assertFalse(empty.is_loaded)
        self.assertFalse(empty.undo())
        self.assertFalse(empty.reset())
        with self.assertRaises(RuntimeError):
            empty.apply(Image.new("RGB", (1, 1)))
        with self.assertRaises(RuntimeError):
            empty.save(Path(self.temporary_directory.name) / "nothing.png")


if __name__ == "__main__":
    unittest.main()
