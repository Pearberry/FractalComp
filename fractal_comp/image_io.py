"""
Image loading, resizing, normalization, and built-in sample test image generation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale_image(path: str | Path, size: int = 64) -> np.ndarray:
    """
    Load any image file, convert it to grayscale ('L'), resize it to (size, size)
    using bicubic resampling, and normalize pixel intensities to floats in [0.0, 1.0].
    """
    img_path = Path(path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    with Image.open(img_path) as img:
        gray = img.convert("L")
        resized = gray.resize((size, size), resample=Image.Resampling.BICUBIC)
        arr = np.asarray(resized, dtype=np.float64) / 255.0

    return np.clip(arr, 0.0, 1.0)


def create_sample_image(size: int = 64, kind: str = "ascent") -> np.ndarray:
    """
    Provide a built-in grayscale test image normalized to [0.0, 1.0] of shape (size, size).

    Parameters
    ----------
    size : int
        Target width and height (default: 64).
    kind : str
        - "ascent": Classic standard test image from scipy.datasets (if available),
          falling back to "geometric".
        - "geometric": Synthetic multi-scale self-similar geometric & gradient test pattern.
    """
    if kind == "ascent":
        try:
            import scipy.datasets

            raw = scipy.datasets.ascent()
            pil_img = Image.fromarray(np.uint8(raw)).convert("L")
            pil_img = pil_img.resize((size, size), resample=Image.Resampling.BICUBIC)
            return np.clip(np.asarray(pil_img, dtype=np.float64) / 255.0, 0.0, 1.0)
        except Exception:
            pass

    # Multi-scale geometric + smooth gradient test image with rich self-similar edges
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) / max(size - 1, 1)
    canvas = 0.15 + 0.35 * (0.6 * x + 0.4 * y)

    # Central bright circle
    r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
    canvas[r2 < 0.10] = 0.85

    # Inner dark ring
    canvas[(r2 >= 0.03) & (r2 < 0.05)] = 0.20

    # Diagonal bar & corner squares (self-similar across 2x scales)
    canvas[(np.abs(x - y) < 0.08) & (x > 0.15) & (x < 0.85)] = 0.75
    canvas[(x > 0.12) & (x < 0.32) & (y > 0.65) & (y < 0.85)] = 0.90
    canvas[(x > 0.68) & (x < 0.88) & (y > 0.12) & (y < 0.32)] = 0.10

    return np.clip(canvas, 0.0, 1.0)
