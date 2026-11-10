"""
Dihedral group isometries (8 symmetries of the square) and 2x2 block averaging downsampling.
"""

from __future__ import annotations

import numpy as np

# Human-readable labels for the 8 dihedral isometries:
# 0..3: rotations by 0°, 90°, 180°, 270° counter-clockwise
# 4..7: horizontal flip followed by rotations by 0°, 90°, 180°, 270° CCW
ISOMETRY_NAMES: tuple[str, ...] = (
    "rot_0",
    "rot_90",
    "rot_180",
    "rot_270",
    "flip_h_rot_0",
    "flip_h_rot_90",
    "flip_h_rot_180",
    "flip_h_rot_270",
)


def downsample_2x2(block: np.ndarray) -> np.ndarray:
    """
    Downsample a 2D block of shape (2*R, 2*R) or a batch of 2D blocks of shape (..., 2*R, 2*R)
    to (..., R, R) via 2x2 local block averaging.
    """
    *leading, h, w = block.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"Block dimensions must be even for 2x2 downsampling, got ({h}, {w}).")
    r_h, r_w = h // 2, w // 2
    reshaped = block.reshape(*leading, r_h, 2, r_w, 2)
    return reshaped.mean(axis=(-3, -1))


def apply_isometry(block: np.ndarray, isometry_index: int) -> np.ndarray:
    """
    Apply one of the 8 dihedral group isometries (0 <= isometry_index < 8) to a 2D array
    (or batch of 2D arrays along the last two axes).

    Index mapping:
      0: Rotate 0°
      1: Rotate 90° CCW
      2: Rotate 180°
      3: Rotate 270° CCW
      4: Horizontal flip + Rotate 0°
      5: Horizontal flip + Rotate 90° CCW
      6: Horizontal flip + Rotate 180°
      7: Horizontal flip + Rotate 270° CCW
    """
    if not (0 <= isometry_index < 8):
        raise ValueError(f"isometry_index must be in 0..7, got {isometry_index}")

    out = block
    if isometry_index >= 4:
        # Horizontal flip (reverse last axis)
        out = np.flip(out, axis=-1)
    k = isometry_index % 4
    if k > 0:
        out = np.rot90(out, k=k, axes=(-2, -1))
    return np.ascontiguousarray(out)


def all_isometries(block: np.ndarray) -> np.ndarray:
    """
    Generate all 8 dihedral isometries for a 2D block of shape (R, R) or a batch (N, R, R).
    Returns:
      - If input is (R, R): shape (8, R, R)
      - If input is (N, R, R): shape (N, 8, R, R)
    """
    if block.ndim == 2:
        return np.stack([apply_isometry(block, idx) for idx in range(8)], axis=0)
    elif block.ndim == 3:
        return np.stack([apply_isometry(block, idx) for idx in range(8)], axis=1)
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {block.shape}")

