"""
Unit tests for the PIFS Fractal Image Compression engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from fractal_comp import (
    FractalCompressor,
    PIFSTransform,
    all_isometries,
    apply_isometry,
    compute_rmse,
    create_sample_image,
    decode_step,
    downsample_2x2,
)


def test_downsample_2x2_averages_2x2_blocks() -> None:
    block = np.array(
        [
            [0.0, 0.2, 0.4, 0.6],
            [0.2, 0.4, 0.6, 0.8],
            [0.1, 0.1, 0.9, 0.9],
            [0.3, 0.3, 0.7, 0.7],
        ],
        dtype=np.float64,
    )
    down = downsample_2x2(block)
    expected = np.array(
        [
            [0.2, 0.6],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(down, expected, atol=1e-12)


def test_eight_dihedral_isometries_are_distinct_on_asymmetric_block() -> None:
    rng = np.random.default_rng(42)
    asymmetric = rng.random((8, 8))
    iso_stack = all_isometries(asymmetric)
    assert iso_stack.shape == (8, 8, 8)

    # All 8 transformations of an asymmetric block must be pairwise distinct
    for i in range(8):
        np.testing.assert_allclose(iso_stack[i], apply_isometry(asymmetric, i))
        for j in range(i + 1, 8):
            assert not np.allclose(iso_stack[i], iso_stack[j])


def test_contrast_clamping_and_tuple_format() -> None:
    image = create_sample_image(size=64, kind="geometric")
    compressor = FractalCompressor(range_size=8, domain_step=8, max_contrast=0.75)
    transforms = compressor.compress(image)

    assert len(transforms) == 64  # 8x8 grid of range blocks for 64x64 image
    for t in transforms:
        assert isinstance(t, PIFSTransform)
        assert abs(t.s) <= 0.75 + 1e-12
        assert 0 <= t.isometry_index < 8
        assert len(t.to_tuple()) == 8
        assert t.rmse_error >= 0.0


def test_flat_image_handles_zero_variance_domains() -> None:
    flat_img = np.full((64, 64), 0.37, dtype=np.float64)
    compressor = FractalCompressor(range_size=8, domain_step=8)
    transforms = compressor.compress(flat_img)

    for t in transforms:
        assert t.s == pytest.approx(0.0, abs=1e-12)
        assert t.o == pytest.approx(0.37, abs=1e-12)
        assert t.rmse_error == pytest.approx(0.0, abs=1e-12)

    history = compressor.decode(transforms, shape=flat_img.shape, iterations=4)
    np.testing.assert_allclose(history[1], flat_img, atol=1e-12)


def test_iterative_reconstruction_converges_from_gray() -> None:
    image = create_sample_image(size=64, kind="ascent")
    compressor = FractalCompressor(range_size=8, domain_step=8, max_contrast=0.75)
    transforms = compressor.compress(image)

    history = compressor.decode(transforms, shape=image.shape, iterations=10, initial_value=0.5)
    assert len(history) == 11

    # Iteration 0 is uniform 0.5 gray
    np.testing.assert_allclose(history[0], 0.5)

    # Error between successive iterations ||I_{k+1} - I_k|| must shrink toward 0 (Banach Fixed-Point)
    step_diffs = [compute_rmse(history[k], history[k + 1]) for k in range(len(history) - 1)]
    assert step_diffs[-1] < 1e-3
    assert step_diffs[-1] < step_diffs[0]

    # Final reconstruction RMSE against target image must be substantially better than initial gray canvas
    initial_rmse = compute_rmse(image, history[0])
    final_rmse = compute_rmse(image, history[-1])
    assert final_rmse < initial_rmse * 0.5


def test_finer_domain_step_improves_or_matches_encoding_error() -> None:
    image = create_sample_image(size=64, kind="geometric")
    comp_coarse = FractalCompressor(range_size=8, domain_step=8)
    comp_fine = FractalCompressor(range_size=8, domain_step=4)

    t_coarse = comp_coarse.compress(image)
    t_fine = comp_fine.compress(image)

    # Because step=4 is a superset of step=8 domain positions, every range block's fit RMSE
    # with step=4 must be <= its fit RMSE with step=8.
    for tc, tf in zip(t_coarse, t_fine):
        assert tf.rmse_error <= tc.rmse_error + 1e-12

