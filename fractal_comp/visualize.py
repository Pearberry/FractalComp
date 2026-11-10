"""
Visualization tools for PIFS fractal image compression:
1. Iterative reconstruction progression strip (Original | Iter 0 | Iter 1 | Iter 2 | Iter 4 | Iter 8)
2. Static & interactive Domain -> Range block mapping inspector with bounding boxes and arrows
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .isometries import apply_isometry, downsample_2x2
from .pifs import PIFSTransform, compute_psnr, compute_rmse


def plot_reconstruction_progression(
    original: np.ndarray,
    history: Sequence[np.ndarray],
    displayed_iterations: Sequence[int] = (0, 1, 2, 4, 8),
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot the iterative PIFS reconstruction progression:
      Original Image | Iteration 0 (Gray) | Iteration 1 | Iteration 2 | Iteration 4 | Iteration 8
    """
    valid_iters = [i for i in displayed_iterations if i < len(history)]
    num_cols = 1 + len(valid_iters)

    fig, axes = plt.subplots(1, num_cols, figsize=(3.0 * num_cols, 3.6), squeeze=False)
    ax_row = axes[0]

    # Panel 0: Original Image
    ax_row[0].imshow(original, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_row[0].set_title(
        f"Original Image\n({original.shape[1]}x{original.shape[0]})", fontsize=10, fontweight="bold"
    )
    ax_row[0].axis("off")

    for col_idx, it in enumerate(valid_iters, start=1):
        canvas = history[it]
        rmse = compute_rmse(original, canvas)
        psnr = compute_psnr(original, canvas)
        label = f"Iteration {it} (Gray)" if it == 0 else f"Iteration {it}"
        subtitle = f"{label}\nRMSE: {rmse:.4f} | {psnr:.1f} dB"

        ax_row[col_idx].imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_row[col_idx].set_title(subtitle, fontsize=9.5)
        ax_row[col_idx].axis("off")

    fig.suptitle(
        "PIFS Fractal Image Compression — Fixed-Point Attractor Convergence",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_block_mapping(
    image: np.ndarray,
    transforms: Sequence[PIFSTransform],
    selected_index: int = 0,
    reconstructed: np.ndarray | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize a single PIFSTransform (or allow interactive click selection across all range blocks).
    Draws bounding boxes around the source Domain block (orange) and target Range block (cyan)
    connected by an arrow, plus a breakdown of the transformation steps.

    Click anywhere on the left image panel in an interactive window to inspect another range block.
    """
    if not transforms:
        raise ValueError("transforms list cannot be empty.")

    canvas_to_show = reconstructed if reconstructed is not None else image
    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[2.2, 1, 1, 1], hspace=0.38, wspace=0.3)

    ax_main = fig.add_subplot(gs[:, 0])
    ax_dom = fig.add_subplot(gs[0, 1])
    ax_iso = fig.add_subplot(gs[0, 2])
    ax_pred = fig.add_subplot(gs[0, 3])
    ax_rng = fig.add_subplot(gs[1, 1])
    ax_err = fig.add_subplot(gs[1, 2])
    ax_info = fig.add_subplot(gs[1, 3])

    def _render_transform(t: PIFSTransform) -> None:
        for ax in (ax_main, ax_dom, ax_iso, ax_pred, ax_rng, ax_err, ax_info):
            ax.clear()

        r = t.range_size
        d = t.domain_size

        ax_main.imshow(canvas_to_show, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_main.set_title("Click any Range block to inspect", fontsize=10, fontweight="bold")

        # Draw subtle range grid
        h, w = canvas_to_show.shape
        for gx in range(0, w + 1, r):
            ax_main.axvline(gx - 0.5, color="white", alpha=0.15, linewidth=0.6)
        for gy in range(0, h + 1, r):
            ax_main.axhline(gy - 0.5, color="white", alpha=0.15, linewidth=0.6)

        # Domain bounding box (orange)
        dom_rect = patches.Rectangle(
            (t.best_domain_x - 0.5, t.best_domain_y - 0.5),
            d,
            d,
            linewidth=2.2,
            edgecolor="#ff7f0e",
            facecolor="#ff7f0e",
            alpha=0.22,
            label=f"Domain ({d}x{d})",
        )
        dom_border = patches.Rectangle(
            (t.best_domain_x - 0.5, t.best_domain_y - 0.5),
            d,
            d,
            linewidth=2.2,
            edgecolor="#ff7f0e",
            facecolor="none",
        )
        ax_main.add_patch(dom_rect)
        ax_main.add_patch(dom_border)

        # Range bounding box (cyan)
        rng_rect = patches.Rectangle(
            (t.range_x - 0.5, t.range_y - 0.5),
            r,
            r,
            linewidth=2.2,
            edgecolor="#00e5ff",
            facecolor="#00e5ff",
            alpha=0.25,
            label=f"Range ({r}x{r})",
        )
        rng_border = patches.Rectangle(
            (t.range_x - 0.5, t.range_y - 0.5),
            r,
            r,
            linewidth=2.2,
            edgecolor="#00e5ff",
            facecolor="none",
        )
        ax_main.add_patch(rng_rect)
        ax_main.add_patch(rng_border)

        # Arrow from domain center to range center
        dom_cx = t.best_domain_x + d / 2.0 - 0.5
        dom_cy = t.best_domain_y + d / 2.0 - 0.5
        rng_cx = t.range_x + r / 2.0 - 0.5
        rng_cy = t.range_y + r / 2.0 - 0.5

        ax_main.annotate(
            "",
            xy=(rng_cx, rng_cy),
            xytext=(dom_cx, dom_cy),
            arrowprops=dict(
                arrowstyle="->",
                color="#ffeb3b",
                lw=2.0,
                mutation_scale=16,
                connectionstyle="arc3,rad=0.15",
            ),
        )
        ax_main.legend(loc="upper right", fontsize=8, framealpha=0.85)

        # Extract sub-blocks from original image to illustrate the transform pipeline
        dom_block = image[t.best_domain_y : t.best_domain_y + d, t.best_domain_x : t.best_domain_x + d]
        down_block = downsample_2x2(dom_block)
        iso_block = apply_isometry(down_block, t.isometry_index)
        pred_block = np.clip(t.s * iso_block + t.o, 0.0, 1.0)
        rng_block = image[t.range_y : t.range_y + r, t.range_x : t.range_x + r]
        abs_err = np.abs(rng_block - pred_block)

        ax_dom.imshow(dom_block, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_dom.set_title(f"1. Domain ({d}x{d})\n@ ({t.best_domain_x}, {t.best_domain_y})", fontsize=9)
        ax_dom.axis("off")

        ax_iso.imshow(iso_block, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_iso.set_title(f"2. Downsample + Iso #{t.isometry_index}\n({t.isometry_name})", fontsize=9)
        ax_iso.axis("off")

        ax_pred.imshow(pred_block, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_pred.set_title(f"3. Affine Fit (s*D + o)\ns={t.s:+.3f}, o={t.o:+.3f}", fontsize=9)
        ax_pred.axis("off")

        ax_rng.imshow(rng_block, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_rng.set_title(f"Target Range ({r}x{r})\n@ ({t.range_x}, {t.range_y})", fontsize=9)
        ax_rng.axis("off")

        im_err = ax_err.imshow(abs_err, cmap="magma", vmin=0.0, vmax=0.25, interpolation="nearest")
        ax_err.set_title(f"|Range - Fit|\nRMSE = {t.rmse_error:.4f}", fontsize=9)
        ax_err.axis("off")

        ax_info.axis("off")
        info_text = (
            f"PIFSTransform Details\n"
            f"─────────────────────\n"
            f"Range (x, y):  ({t.range_x}, {t.range_y})\n"
            f"Domain (x, y): ({t.best_domain_x}, {t.best_domain_y})\n"
            f"Isometry:      {t.isometry_index} ({t.isometry_name})\n"
            f"Contrast (s):  {t.s:+.4f}\n"
            f"Offset (o):    {t.o:+.4f}\n"
            f"Block RMSE:    {t.rmse_error:.5f}"
        )
        ax_info.text(
            0.02,
            0.95,
            info_text,
            va="top",
            ha="left",
            fontsize=8.8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f6f8", edgecolor="#cbd5e1"),
        )
        fig.canvas.draw_idle()

    initial_idx = max(0, min(selected_index, len(transforms) - 1))
    _render_transform(transforms[initial_idx])

    def _on_click(event) -> None:
        if event.inaxes != ax_main or event.xdata is None or event.ydata is None:
            return
        px, py = int(round(event.xdata)), int(round(event.ydata))
        for t in transforms:
            r = t.range_size
            if t.range_x <= px < t.range_x + r and t.range_y <= py < t.range_y + r:
                _render_transform(t)
                break

    fig.canvas.mpl_connect("button_press_event", _on_click)

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()

    return fig
