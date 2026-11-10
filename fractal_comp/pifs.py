"""
Core Partitioned Iterated Function System (PIFS) engine:
- PIFSTransform dataclass
- Vectorized domain codebook construction & closed-form least-squares encoding
- Iterative fixed-point decoding (decode_step)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .isometries import ISOMETRY_NAMES, all_isometries, apply_isometry, downsample_2x2


@dataclass(frozen=True)
class PIFSTransform:
    """
    Stores the optimal affine contractive transformation mapping a D x D domain block
    to an R x R range block.

    Coordinate convention:
      - x is the horizontal pixel coordinate (column index in 2D array: image[:, x])
      - y is the vertical pixel coordinate (row index in 2D array: image[y, :])
    """

    range_x: int
    range_y: int
    best_domain_x: int
    best_domain_y: int
    isometry_index: int
    s: float
    o: float
    rmse_error: float
    range_size: int = 8

    @property
    def domain_size(self) -> int:
        return 2 * self.range_size

    @property
    def isometry_name(self) -> str:
        return ISOMETRY_NAMES[self.isometry_index]

    def to_tuple(self) -> tuple[int, int, int, int, int, float, float, float]:
        """Return the 8-tuple specified in the design doc."""
        return (
            self.range_x,
            self.range_y,
            self.best_domain_x,
            self.best_domain_y,
            self.isometry_index,
            self.s,
            self.o,
            self.rmse_error,
        )


def decode_step(
    current_canvas: np.ndarray,
    transforms: Sequence[PIFSTransform],
    range_size: int | None = None,
) -> np.ndarray:
    """
    Perform a single deterministic PIFS iteration:
      next_canvas = W(current_canvas)

    For each range block:
      1. Extract the mapped D x D domain block from `current_canvas`
      2. Downsample to R x R via 2x2 local block averaging
      3. Apply the recorded dihedral isometry
      4. Scale by contrast `s`, add offset `o`, and clamp to [0.0, 1.0]
      5. Place into the range location on `next_canvas`
    """
    next_canvas = np.zeros_like(current_canvas, dtype=np.float64)

    for t in transforms:
        r = range_size if range_size is not None else t.range_size
        d = 2 * r
        domain_block = current_canvas[
            t.best_domain_y : t.best_domain_y + d,
            t.best_domain_x : t.best_domain_x + d,
        ]
        downsampled = downsample_2x2(domain_block)
        transformed = apply_isometry(downsampled, t.isometry_index)
        reconstructed = np.clip(t.s * transformed + t.o, 0.0, 1.0)
        next_canvas[t.range_y : t.range_y + r, t.range_x : t.range_x + r] = reconstructed

    return next_canvas


def compute_rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    """Compute Root Mean Squared Error between two normalized float images."""
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def compute_psnr(target: np.ndarray, prediction: np.ndarray, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) between two normalized float images."""
    mse = float(np.mean((target - prediction) ** 2))
    if mse <= 1e-15:
        return float("inf")
    return float(10.0 * np.log10((max_val ** 2) / mse))


class FractalCompressor:
    """
    Partitioned Iterated Function System (PIFS) Fractal Image Compressor.

    Parameters
    ----------
    range_size : int
        Size R of non-overlapping R x R range blocks (default: 8).
        Domain blocks have size D = 2 * R (default: 16).
    domain_step : int
        Spatial stride for candidate domain block extraction (default: 8; supports 4, 2, 1).
    max_contrast : float
        Strict upper bound on |s| to guarantee Banach Fixed-Point contractivity (default: 0.75).
    """

    def __init__(
        self,
        range_size: int = 8,
        domain_step: int = 8,
        max_contrast: float = 0.75,
    ) -> None:
        if range_size <= 0:
            raise ValueError("range_size must be positive.")
        if domain_step <= 0:
            raise ValueError("domain_step must be positive.")
        if not (0.0 < max_contrast < 1.0):
            raise ValueError("max_contrast must be in (0.0, 1.0) for contractivity.")

        self.range_size = range_size
        self.domain_size = 2 * range_size
        self.domain_step = domain_step
        self.max_contrast = max_contrast
        self.transforms: list[PIFSTransform] = []

    def _build_domain_codebook(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract all candidate D x D domain blocks, downsample to R x R, and precompute
        all 8 dihedral isometries in a vectorized codebook.

        Returns
        -------
        flat_codebook : np.ndarray of shape (M, R*R)
            Flattened transformed domain blocks (where M = num_domains * 8).
        domain_means : np.ndarray of shape (M,)
            Mean intensity of each candidate entry.
        domain_centered : np.ndarray of shape (M, R*R)
            Zero-mean candidate vectors (D_k - mean(D_k)).
        meta_coords : np.ndarray of shape (M, 3)
            Integer array of (domain_x, domain_y, isometry_index) for each entry.
        """
        h, w = image.shape
        d = self.domain_size
        step = self.domain_step

        ys = list(range(0, h - d + 1, step))
        xs = list(range(0, w - d + 1, step))
        if not ys or not xs:
            raise ValueError(
                f"Image shape {image.shape} is smaller than domain size ({d}x{d})."
            )

        raw_domains = []
        coords = []
        for y in ys:
            for x in xs:
                raw_domains.append(image[y : y + d, x : x + d])
                coords.append((x, y))

        # Shape: (num_domains, D, D) -> downsample to (num_domains, R, R)
        domains_arr = np.stack(raw_domains, axis=0)
        downsampled = downsample_2x2(domains_arr)

        # Generate all 8 isometries: shape (num_domains, 8, R, R)
        iso_blocks = all_isometries(downsampled)

        num_domains = len(coords)
        num_pixels = self.range_size * self.range_size
        flat_codebook = iso_blocks.reshape(num_domains * 8, num_pixels)

        # Build metadata table (domain_x, domain_y, isometry_index)
        meta_coords = np.empty((num_domains, 8, 3), dtype=np.int32)
        coords_arr = np.array(coords, dtype=np.int32)
        meta_coords[:, :, 0] = coords_arr[:, 0:1]
        meta_coords[:, :, 1] = coords_arr[:, 1:2]
        meta_coords[:, :, 2] = np.arange(8, dtype=np.int32)[None, :]
        meta_coords = meta_coords.reshape(num_domains * 8, 3)

        domain_means = flat_codebook.mean(axis=1)
        domain_centered = flat_codebook - domain_means[:, None]

        return flat_codebook, domain_means, domain_centered, meta_coords

    def compress(self, image: np.ndarray) -> list[PIFSTransform]:
        """
        Compress a normalized 2D grayscale image in [0.0, 1.0] into a list of PIFSTransform objects.
        Also stores the result in `self.transforms`.
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")
        h, w = image.shape
        r = self.range_size
        if h % r != 0 or w % r != 0:
            raise ValueError(
                f"Image dimensions ({h}x{w}) must be divisible by range_size ({r})."
            )

        flat_codebook, domain_means, domain_centered, meta_coords = (
            self._build_domain_codebook(image)
        )
        # Sum of squared deviations for each candidate domain block: sum((D_k - mean(D_k))^2)
        domain_ss = np.sum(domain_centered ** 2, axis=1)
        valid_var_mask = domain_ss > 1e-12

        transforms: list[PIFSTransform] = []
        num_pixels = r * r

        for ry in range(0, h, r):
            for rx in range(0, w, r):
                r_block = image[ry : ry + r, rx : rx + r].reshape(num_pixels)
                r_mean = float(np.mean(r_block))
                r_centered = r_block - r_mean

                # Unconstrained least-squares slope: s = sum((D - mean(D))*(R - mean(R))) / sum((D - mean(D))^2)
                numerator = domain_centered @ r_centered
                s_vec = np.zeros_like(domain_means)
                s_vec[valid_var_mask] = numerator[valid_var_mask] / domain_ss[valid_var_mask]

                # Strictly clamp contrast |s| <= max_contrast (0.75) for Banach Fixed-Point condition
                s_vec = np.clip(s_vec, -self.max_contrast, self.max_contrast)

                # Optimal brightness offset given (possibly clamped) s: o = mean(R) - s * mean(D)
                o_vec = r_mean - s_vec * domain_means

                # Residuals across all candidate domain-isometry pairs: s * D_k + o - R_i
                residuals = (
                    s_vec[:, None] * flat_codebook + o_vec[:, None] - r_block[None, :]
                )
                mse_vec = np.mean(residuals ** 2, axis=1)
                best_idx = int(np.argmin(mse_vec))

                best_dx, best_dy, best_iso = meta_coords[best_idx]
                best_s = float(s_vec[best_idx])
                best_o = float(o_vec[best_idx])
                best_rmse = float(np.sqrt(mse_vec[best_idx]))

                transforms.append(
                    PIFSTransform(
                        range_x=int(rx),
                        range_y=int(ry),
                        best_domain_x=int(best_dx),
                        best_domain_y=int(best_dy),
                        isometry_index=int(best_iso),
                        s=best_s,
                        o=best_o,
                        rmse_error=best_rmse,
                        range_size=r,
                    )
                )

        self.transforms = transforms
        return transforms

    def decode(
        self,
        transforms: Sequence[PIFSTransform] | None = None,
        shape: tuple[int, int] = (64, 64),
        iterations: int = 8,
        initial_value: float = 0.5,
    ) -> list[np.ndarray]:
        """
        Iteratively reconstruct the image starting from a uniform canvas I_0 = initial_value (default: 0.5).

        Returns
        -------
        history : list[np.ndarray]
            List of canvases [I_0, I_1, ..., I_iterations] of length `iterations + 1`.
        """
        active_transforms = transforms if transforms is not None else self.transforms
        if not active_transforms:
            raise ValueError("No transforms provided or stored in compressor.")

        current = np.full(shape, initial_value, dtype=np.float64)
        history = [current.copy()]

        for _ in range(iterations):
            current = decode_step(current, active_transforms, range_size=self.range_size)
            history.append(current.copy())

        return history

    def find_transform_at_pixel(self, x: int, y: int) -> PIFSTransform | None:
        """
        Helper for interactive visualizers: given a pixel coordinate (x, y),
        return the PIFSTransform covering that range block.
        """
        r = self.range_size
        for t in self.transforms:
            if t.range_x <= x < t.range_x + r and t.range_y <= y < t.range_y + r:
                return t
        return None

