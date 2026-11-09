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

    ``step(i, total, text)`` shows a determinate bar with elapsed / remaining time (call it from the Tk thread);
    ``run(text, func)`` runs ``func`` in a worker thread with a moving bar so the window never looks frozen;
    ``run_step(i, total, text, func)`` does both: item ``i`` of a loop computed in a worker thread.
    Without a real Tk parent (unit tests with mocks) it does nothing and ``run`` just calls ``func``.
    """

    def __init__(self, parent: Any, title: str, text: str = "Working…") -> None:
        self.window: tk.Toplevel | None = None
        self._loop_start: float | None = None  # time of the first step(), for the remaining-time estimate
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
        self.bar.configure(mode="determinate", maximum=max(total, 1), value=index)
        if text is not None:
            self.label.configure(text=text)
        self.count.configure(text=f"{index + 1} / {total}{self._timing(index, total)}")
        self.window.update()

    def _timing(self, index: int, total: int) -> str:
        """``  ·  12 s elapsed, ~30 s left`` once a few items are done (index = items already finished)."""
        now = time.monotonic()
        if self._loop_start is None or index == 0:
            self._loop_start = now
            return ""
        elapsed = now - self._loop_start
        left = elapsed / index * (total - index)
        return f"  ·  {format_seconds(elapsed)} elapsed, ~{format_seconds(left)} left"

    def run_step(self, index: int, total: int, text: str, func: Callable[[], T]) -> T:
        """Item ``index`` of ``total``: update the bar, then compute ``func()`` in a worker thread."""
        self.step(index, total, text)
        return self._in_thread(func)

    def run(self, text: str, func: Callable[[], T]) -> T:
        """``func()`` in a worker thread (it must not touch Tk) while the bar keeps moving."""
        if self.window is None:
            return func()
        self.label.configure(text=text)
        self.count.configure(text="")
        self.bar.configure(mode="indeterminate", maximum=100, value=0)
        self.bar.start(15)
        try:
            return self._in_thread(func)
        finally:
            self.bar.stop()

    def _in_thread(self, func: Callable[[], T]) -> T:
        """``func()`` in a worker thread while the popup keeps redrawing; exceptions are re-raised here."""
        if self.window is None:
            return func()
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
        if "error" in outcome:
            raise outcome["error"]
        return outcome["value"]

    def close(self) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None


def format_seconds(seconds: float) -> str:
    """``42 s`` or ``3 min 05 s``."""
    seconds = int(round(seconds))
    return f"{seconds} s" if seconds < 60 else f"{seconds // 60} min {seconds % 60:02d} s"


__all__ = ["ProgressPopup", "format_seconds"]
