"""Helpers for normalizing the many PIL image modes to what intensity tools expect."""

from __future__ import annotations

from PIL import Image


def split_alpha(image: Image.Image) -> tuple[Image.Image, Image.Image | None]:
    """Return an ``L`` or ``RGB`` image of intensities plus its alpha band (or ``None``)."""
    if image.mode == "P":
        # Palette pixels are indices into a colour table, not intensities
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")
    if image.mode in ("LA", "La", "PA"):
        image = image.convert("LA")
        return image.getchannel("L"), image.getchannel("A")
    if image.mode in ("RGBA", "RGBa"):
        return image.convert("RGB"), image.getchannel("A")
    if image.mode in ("L", "RGB"):
        return image, None
    # 1, I, F -> 8-bit gray; CMYK, YCbCr, HSV, ... -> RGB
    return image.convert("L" if len(image.getbands()) == 1 else "RGB"), None
