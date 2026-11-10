"""
Fractal Image Compression using Partitioned Iterated Function Systems (PIFS).
"""

from .isometries import (
    ISOMETRY_NAMES,
    apply_isometry,
    all_isometries,
    downsample_2x2,
)
from .pifs import (
    PIFSTransform,
    FractalCompressor,
    decode_step,
    compute_rmse,
    compute_psnr,
)
from .image_io import (
    load_grayscale_image,
    create_sample_image,
)

__all__ = [
    "ISOMETRY_NAMES",
    "apply_isometry",
    "all_isometries",
    "downsample_2x2",
    "PIFSTransform",
    "FractalCompressor",
    "decode_step",
    "compute_rmse",
    "compute_psnr",
    "load_grayscale_image",
    "create_sample_image",
]

