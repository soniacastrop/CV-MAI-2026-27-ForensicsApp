"""Numeric fields shared by the parameter dialogs."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any


def spin_row(
    master: tk.Misc, row: int, text: str, variable: tk.StringVar, bounds: tuple[float, float], step: float
) -> ttk.Spinbox:
    """A label and a spinbox on grid ``row`` of ``master``."""
    ttk.Label(master, text=text).grid(row=row, column=0, sticky="w", padx=6)
    box = ttk.Spinbox(master, from_=bounds[0], to=bounds[1], increment=step, textvariable=variable, width=8)
    box.grid(row=row, column=1, sticky="w", padx=6, pady=2)
    return box


def read_number(variable: tk.StringVar, name: str, bounds: tuple[float, float], kind: type = float) -> Any:
    """Parse ``variable`` as ``kind`` within ``bounds``; raises ``ValueError`` with a user-facing message."""
    try:
        value = kind(variable.get())
    except ValueError:
        raise ValueError(f"{name} must be {'an integer' if kind is int else 'a number'}.") from None
    if not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"{name} must be between {bounds[0]:g} and {bounds[1]:g}.")
    return value
