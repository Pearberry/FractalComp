"""
Command-line entry point for the PIFS Fractal Image Compression experiment.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from fractal_comp import (
    FractalCompressor,
    compute_psnr,
    compute_rmse,
    create_sample_image,
    load_grayscale_image,
)
from fractal_comp.visualize import plot_block_mapping, plot_reconstruction_progression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partitioned Iterated Function System (PIFS) Fractal Image Compression Explorer"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an input image (grayscale or color). Uses built-in test image if omitted.",
    )
    parser.add_argument(
        "--sample-kind",
        type=str,
        default="ascent",
        choices=["ascent", "geometric"],
        help="Built-in sample image kind when --image is not specified (default: ascent).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="Target N x N image resolution (default: 64).",
    )
    parser.add_argument(
        "--range-size",
        type=int,
        default=8,
        help="Range block size R (default: 8, giving D = 2*R = 16).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=8,
        help="Domain extraction stride step (default: 8; try 4 or 2 for higher quality).",
    )
    parser.add_argument(
        "--max-contrast",
        type=float,
        default=0.75,
        help="Maximum absolute contrast |s| clamp for contractivity (default: 0.75).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="Number of decoding iterations from neutral gray I_0 = 0.5 (default: 8).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="output",
        help="Directory to save generated plots (default: output).",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Also open/save the Domain -> Range block mapping inspector.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive Matplotlib windows (save figures only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.image:
        print(f"Loading image '{args.image}' resized to {args.size}x{args.size}...")
        image = load_grayscale_image(args.image, size=args.size)
    else:
        print(
            f"Using built-in '{args.sample_kind}' test image resized to {args.size}x{args.size}..."
        )
        image = create_sample_image(size=args.size, kind=args.sample_kind)

    compressor = FractalCompressor(
        range_size=args.range_size,
        domain_step=args.step,
        max_contrast=args.max_contrast,
    )

    t0 = time.perf_counter()
    transforms = compressor.compress(image)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    num_domains_1d = (args.size - 2 * args.range_size) // args.step + 1
    total_candidates = (num_domains_1d ** 2) * 8
    print(
        f"Encoded {len(transforms)} range blocks ({args.range_size}x{args.range_size}) "
        f"against {total_candidates} domain-isometry candidates in {elapsed_ms:.1f} ms."
    )

    history = compressor.decode(
        transforms=transforms,
        shape=image.shape,
        iterations=args.iterations,
        initial_value=0.5,
    )

    print("\nIterative Reconstruction Metrics:")
    print(f"{'Iter':>4} | {'RMSE':>10} | {'PSNR (dB)':>10}")
    print("-" * 32)
    for it, canvas in enumerate(history):
        rmse = compute_rmse(image, canvas)
        psnr = compute_psnr(image, canvas)
        print(f"{it:>4d} | {rmse:>10.5f} | {psnr:>10.2f}")

    # Pick an interesting high-contrast range block for the sample summary / inspector
    interesting_idx = max(
        range(len(transforms)),
        key=lambda idx: abs(transforms[idx].s),
    )
    sample_t = transforms[interesting_idx]
    print(
        f"\nSample PIFSTransform [block #{interesting_idx}]:\n"
        f"  tuple = {sample_t.to_tuple()}"
    )

    save_dir = Path(args.save_dir)
    prog_path = save_dir / "pifs_progression.png"
    plot_reconstruction_progression(
        original=image,
        history=history,
        displayed_iterations=(0, 1, 2, 4, args.iterations),
        save_path=prog_path,
        show=not args.no_show,
    )
    print(f"\nSaved progression figure to: {prog_path}")

    if args.inspect or args.no_show:
        map_path = save_dir / "pifs_block_mapping.png"
        plot_block_mapping(
            image=image,
            transforms=transforms,
            selected_index=interesting_idx,
            reconstructed=history[-1],
            save_path=map_path,
            show=args.inspect and not args.no_show,
        )
        print(f"Saved block mapping inspector figure to: {map_path}")


if __name__ == "__main__":
    main()

