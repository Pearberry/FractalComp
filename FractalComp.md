I want to build an educational Partitioned Iterated Function System (PIFS) fractal image compression experiment in Python.

### Core Architecture & Constraints:
1. Input:
   - Accept any grayscale image and resize it to an initial N x N test size (default: 64x64, using Pillow or OpenCV).
   - Normalize pixel intensities to floats in [0.0, 1.0].

2. Partitioning:
   - Range blocks: Non-overlapping R x R blocks tiling the canvas (default: R = 8, yielding an 8x8 grid of 64 range blocks for a 64x64 image).
   - Domain blocks: D x D blocks where D = 2 * R (default: 16x16).
   - Domain extraction: Extract candidate domain blocks across the image using a configurable step stride (default: step = 8 for fast evaluation; allow step = 4 or step = 2).

3. Block Transformation:
   - For every candidate domain block, downsample it to R x R by 2x2 local block averaging.
   - Test all 8 dihedral isometries (4 rotations: 0°, 90°, 180°, 270°; each with/without horizontal flip).
   - Use closed-form linear regression to solve for optimal contrast (s) and brightness offset (o):
     Minimize sum((s * D_k + o - R_i)^2).
   - Strictly clamp contrast: |s| <= 0.75 to ensure the Banach Fixed-Point contractive condition holds.
   - Record the transformation tuple for each range block:
     (range_x, range_y, best_domain_x, best_domain_y, isometry_index, s, o, rmse_error).

4. Iterative Reconstruction & Visualization:
   - Start with an initial canvas I_0 initialized entirely to solid neutral gray (0.5).
   - Implement a deterministic step generator `decode_step(current_canvas, transforms) -> next_canvas`:
     For each range block, extract the mapped domain block from `current_canvas`, downsample, apply the recorded isometry, scale by s, add o, clamp to [0.0, 1.0], and place it into the range location.
   - Run 8–10 iterations.
   - Plot the progression using Matplotlib showing: Original Image | Iteration 0 (Gray) | Iteration 1 | Iteration 2 | Iteration 4 | Iteration 8.

5. Code Style & Future Hook:
   - Keep the code modular using clean classes or dataclasses (e.g., `PIFSTransform`, `FractalCompressor`).
   - Store the computed transforms in an accessible list so that I can later build an interactive visualizer that draws an arrow or bounding box between a selected Range block and its source Domain block.
   - Vectorize domain downsampling or isometric transforms where practical using NumPy to keep execution fast.