import tkinter as tk
from tkinter import ttk
import unittest
from unittest import mock

from PIL import Image

from forensics_app.tools import build_tool_registry
from forensics_app.tools.base import ForensicsTool, ToolResult
from forensics_app.tools.registry import ToolRegistry
from forensics_app.ui.main_window import MainWindow

UI = "forensics_app.ui.main_window"


class _InvertTool(ForensicsTool):
    tool_id = "invert"
    title = "Invert"
    category = "Test"

    def run(self, parent, document):
        return ToolResult(message="inverted", image=Image.new("L", (4, 4), 255), details={"Done": "yes"})


class _CancelTool(ForensicsTool):
    tool_id = "cancel"
    title = "Cancel"
    category = "Test"

    def run(self, parent, document):
        return None


class _BrokenTool(ForensicsTool):
    tool_id = "broken"
    title = "Broken"
    category = "Test"

    def run(self, parent, document):
        raise ValueError("boom")


def _descendants(widget: tk.Misc):
    for child in widget.winfo_children():
        yield child
        yield from _descendants(child)


class MainWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.root = tk.Tk()
        except tk.TclError as error:  # no display available
            raise unittest.SkipTest(f"Tk unavailable: {error}")
        self.addCleanup(self.root.destroy)

    def _window(self, registry: ToolRegistry) -> MainWindow:
        window = MainWindow(self.root, registry)
        window.document.current = Image.new("L", (4, 4), 0)
        window.document.original = window.document.current.copy()
        return window

    def _sidebar_canvas(self) -> tk.Canvas:
        canvases = [w for w in _descendants(self.root) if isinstance(w, tk.Canvas) and "imageview" not in str(w)]
        return canvases[0]

    def test_sidebar_has_a_button_for_every_registered_tool(self) -> None:
        registry = build_tool_registry()
        MainWindow(self.root, registry)
        labels = {w.cget("text") for w in _descendants(self.root) if isinstance(w, ttk.Button)}
        for tool in registry.all():
            self.assertIn(tool.title, labels)

    def test_sidebar_scrolls_when_tools_do_not_fit(self) -> None:
        MainWindow(self.root, build_tool_registry())
        self.root.geometry("900x300")
        self.root.update()
        canvas = self._sidebar_canvas()
        self.assertEqual(canvas.yview()[0], 0.0)
        self.assertLess(canvas.yview()[1], 1.0)  # bottom tools are hidden...
        canvas.yview_moveto(1.0)
        self.root.update()
        self.assertEqual(canvas.yview()[1], 1.0)  # ...but reachable by scrolling

    def test_run_tool_applies_result_and_enables_undo(self) -> None:
        window = self._window(ToolRegistry([_InvertTool()]))
        window.run_tool(_InvertTool())
        self.assertEqual(window.document.current.getpixel((0, 0)), 255)
        self.assertEqual(window.status.get(), "inverted")
        self.assertEqual(str(window.undo_button.cget("state")), "normal")
        window.undo()
        self.assertEqual(window.document.current.getpixel((0, 0)), 0)

    def test_cancelled_tool_leaves_image_unchanged(self) -> None:
        window = self._window(ToolRegistry([_CancelTool()]))
        window.run_tool(_CancelTool())
        self.assertEqual(window.status.get(), "Cancelled Cancel.")
        self.assertFalse(window.document.can_undo)

    def test_failing_tool_shows_error_instead_of_crashing(self) -> None:
        window = self._window(ToolRegistry([_BrokenTool()]))
        with mock.patch(f"{UI}.messagebox.showerror") as showerror:
            window.run_tool(_BrokenTool())
        showerror.assert_called_once()
        self.assertEqual(window.status.get(), "Error in Broken.")
        self.assertFalse(window.document.can_undo)

    def test_tool_needing_image_is_blocked_without_one(self) -> None:
        window = MainWindow(self.root, ToolRegistry([_InvertTool()]))
        with mock.patch(f"{UI}.messagebox.showinfo") as showinfo:
            window.run_tool(_InvertTool())
        showinfo.assert_called_once()
        self.assertFalse(window.document.is_loaded)


if __name__ == "__main__":
    unittest.main()
