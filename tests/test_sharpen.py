import tkinter as tk
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.sharpen import SharpenDialog, SharpenSettings, SharpenTool, sharpen

DIALOG = "forensics_app.tools.sharpen.SharpenDialog"


def _step(mode: str = "L") -> Image.Image:
    """Vertical step edge 80 -> 170, as grayscale or a bluish RGB image."""
    arr = np.zeros((20, 40), dtype=np.uint8)
    arr[:, :20], arr[:, 20:] = 80, 170
    gray = Image.fromarray(arr, mode="L")
    if mode == "L":
        return gray
    return Image.merge("RGB", (gray.point(lambda v: v // 2), gray, Image.new("L", gray.size, 200)))


class SharpenFunctionTests(unittest.TestCase):
    def test_creates_halos_at_edges_and_keeps_flat_areas(self) -> None:
        arr = np.asarray(sharpen(_step(), SharpenSettings(radius=2.0, amount=1.5))).astype(int)
        self.assertLess(arr[10, 19], 80)  # dark side undershoots
        self.assertGreater(arr[10, 20], 170)  # bright side overshoots
        self.assertEqual(arr[10, 2], 80)
        self.assertEqual(arr[10, 38], 170)

    def test_zero_amount_leaves_image_unchanged(self) -> None:
        image = _step("RGB")
        out = sharpen(image, SharpenSettings(radius=3.0, amount=0.0))
        np.testing.assert_array_equal(np.asarray(out), np.asarray(image))

    def test_larger_amount_sharpens_more(self) -> None:
        weak = np.asarray(sharpen(_step(), SharpenSettings(radius=2.0, amount=0.5))).astype(int)
        strong = np.asarray(sharpen(_step(), SharpenSettings(radius=2.0, amount=2.0))).astype(int)
        self.assertGreater(strong[10, 20], weak[10, 20])
        self.assertLess(strong[10, 19], weak[10, 19])

    def test_larger_radius_widens_the_halo(self) -> None:
        narrow = np.asarray(sharpen(_step(), SharpenSettings(radius=1.0, amount=1.0))).astype(int)
        wide = np.asarray(sharpen(_step(), SharpenSettings(radius=4.0, amount=1.0))).astype(int)
        self.assertEqual(narrow[10, 15], 80)  # 5 px from the edge: outside a narrow halo
        self.assertLess(wide[10, 15], 80)  # ...but inside a wide one

    def test_colour_is_sharpened_per_channel(self) -> None:
        out = sharpen(_step("RGB"), SharpenSettings(radius=2.0, amount=1.5))
        self.assertEqual(out.mode, "RGB")
        r, g, b = (np.asarray(c).astype(int) for c in out.split())
        self.assertGreater(g[10, 20], 170)  # the green step was sharpened
        self.assertTrue(np.all(b == 200))  # the flat blue channel stays flat

    def test_alpha_is_preserved(self) -> None:
        image = _step("RGB").convert("RGBA")
        image.putalpha(90)
        out = sharpen(image, SharpenSettings())
        self.assertEqual(out.mode, "RGBA")
        self.assertTrue(np.all(np.asarray(out.getchannel("A")) == 90))

    def test_other_modes(self) -> None:
        for image in (_step().convert("LA"), _step("RGB").convert("P"), _step().convert("1")):
            with self.subTest(mode=image.mode):
                self.assertEqual(sharpen(image, SharpenSettings()).size, (40, 20))


class SharpenToolTests(unittest.TestCase):
    def _run(self, settings, image=None):
        document = ImageDocument()
        document.current = image or _step()
        with mock.patch(DIALOG) as dialog:
            dialog.return_value.result = settings
            return SharpenTool().run(None, document), document

    def test_uses_dialog_radius_and_amount(self) -> None:
        result, document = self._run(SharpenSettings(radius=2.5, amount=3.0))
        self.assertEqual(result.details["Radius"], 2.5)
        self.assertEqual(result.details["Amount"], 3.0)
        expected = sharpen(document.current, SharpenSettings(radius=2.5, amount=3.0))
        np.testing.assert_array_equal(np.asarray(result.image), np.asarray(expected))
        self.assertEqual(np.asarray(document.current).max(), 170)  # document untouched

    def test_cancel_returns_none(self) -> None:
        self.assertIsNone(self._run(None)[0])

    def test_registered_in_sidebar(self) -> None:
        from forensics_app.tools import build_tool_registry

        self.assertIn("sharpen_unsharp_mask", {tool.tool_id for tool in build_tool_registry().all()})


class SharpenDialogTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.root = tk.Tk()
        except tk.TclError as error:  # no display available
            raise unittest.SkipTest(f"Tk unavailable: {error}")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    def _submit(self, radius: str, amount: str):
        def press():
            dialog = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)][-1]
            dialog.radius.set(radius)
            dialog.amount.set(amount)
            dialog.ok()
            if dialog.winfo_exists():  # invalid input keeps the dialog open
                dialog.cancel()

        self.root.after(200, press)
        return SharpenDialog(self.root, title="Sharpen").result

    def test_returns_entered_values(self) -> None:
        self.assertEqual(self._submit("3.5", "2"), SharpenSettings(radius=3.5, amount=2.0))

    def test_rejects_out_of_range_or_text(self) -> None:
        with mock.patch("forensics_app.tools.sharpen.messagebox.showerror") as showerror:
            self.assertIsNone(self._submit("0", "1"))
            self.assertIsNone(self._submit("1", "20"))
            self.assertIsNone(self._submit("abc", "1"))
        self.assertEqual(showerror.call_count, 3)


if __name__ == "__main__":
    unittest.main()
