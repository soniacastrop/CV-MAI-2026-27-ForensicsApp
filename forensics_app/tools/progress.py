"""A small "please wait" popup with a progress bar for tools that take a few seconds."""

from __future__ import annotations

from collections.abc import Callable
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Any, TypeVar

T = TypeVar("T")

POLL_SECONDS = 0.03  # how often the popup is redrawn while a background job runs


class ProgressPopup:
    """Popup with a message and a progress bar, used as a context manager.

    ``step(i, total, text)`` shows a determinate bar (call it from the Tk thread, e.g. inside a loop);
    ``run(text, func)`` runs ``func`` in a worker thread with a moving bar so the window never looks frozen.
    Without a real Tk parent (unit tests with mocks) it does nothing and ``run`` just calls ``func``.
    """

    def __init__(self, parent: Any, title: str, text: str = "Working…") -> None:
        self.window: tk.Toplevel | None = None
        if not isinstance(parent, tk.Misc):
            return
        window = tk.Toplevel(parent)
        window.title(title)
        window.transient(parent.winfo_toplevel())
        window.resizable(False, False)
        window.protocol("WM_DELETE_WINDOW", lambda: None)  # cannot be closed mid-task
        self.label = ttk.Label(window, text=text, width=55, anchor="w")
        self.label.pack(padx=14, pady=(12, 6), fill="x")
        self.bar = ttk.Progressbar(window, length=360, mode="determinate", maximum=1)
        self.bar.pack(padx=14, pady=(0, 6), fill="x")
        self.count = ttk.Label(window, text="", anchor="e")
        self.count.pack(padx=14, pady=(0, 10), fill="x")
        window.update_idletasks()
        top = parent.winfo_toplevel()  # centre over the window that opened it
        x = top.winfo_rootx() + (top.winfo_width() - window.winfo_reqwidth()) // 2
        y = top.winfo_rooty() + (top.winfo_height() - window.winfo_reqheight()) // 3
        window.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        window.update()
        self.window = window

    def __enter__(self) -> ProgressPopup:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def step(self, index: int, total: int, text: str | None = None) -> None:
        """Item ``index`` (0-based) of ``total`` is being processed."""
        if self.window is None:
            return
        self.bar.stop()
        self.bar.configure(mode="determinate", maximum=max(total, 1), value=index + 1)
        if text is not None:
            self.label.configure(text=text)
        self.count.configure(text=f"{index + 1} / {total}")
        self.window.update()

    def run(self, text: str, func: Callable[[], T]) -> T:
        """``func()`` in a worker thread (it must not touch Tk) while the bar keeps moving."""
        if self.window is None:
            return func()
        self.label.configure(text=text)
        self.count.configure(text="")
        self.bar.configure(mode="indeterminate", maximum=100, value=0)
        self.bar.start(15)
        outcome: dict[str, Any] = {}

        def work() -> None:
            try:
                outcome["value"] = func()
            except BaseException as error:  # re-raised in the Tk thread below
                outcome["error"] = error

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        while worker.is_alive():
            self.window.update()
            time.sleep(POLL_SECONDS)
        self.bar.stop()
        if "error" in outcome:
            raise outcome["error"]
        return outcome["value"]

    def close(self) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None


__all__ = ["ProgressPopup"]
