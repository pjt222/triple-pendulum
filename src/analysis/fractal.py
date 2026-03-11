"""Fractal dimension estimation for 3D chaos boundary surfaces.

Provides box-counting fractal dimension analysis for the boundary
surfaces extracted from the triple pendulum's 3D flip-time voxel grid.
The fractal dimension quantifies the geometric complexity of the
boundary between regular and chaotic regions in initial-condition space.

**Box-counting method**: Covers the boundary with boxes of decreasing
size *s* and counts how many boxes contain at least one boundary voxel.
The fractal dimension *D* is the slope of ``log(count)`` vs
``log(1/s)``.  For a smooth 2D surface embedded in 3D, *D* = 2; for a
fractal surface, *D* > 2.

**Convergence analysis**: Computes fractal dimension at multiple grid
resolutions to verify that the estimate stabilises as resolution
increases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import numpy.typing as npt
from scipy.stats import linregress

from src.visualization.volume_render import extract_boundary_mask

# Use non-interactive backend so plotting works in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Box-counting fractal dimension
# ---------------------------------------------------------------------------


def box_counting_dimension(
    binary_mask_3d: npt.NDArray[np.bool_],
    min_box_size: int = 1,
    max_box_size: int | None = None,
) -> dict[str, Any]:
    """Estimate the fractal dimension of a 3D binary mask via box counting.

    Divides the volume into non-overlapping cubic boxes of side length
    *s* (powers of 2) and counts how many boxes contain at least one
    ``True`` voxel.  The fractal dimension is the slope of the
    least-squares fit of ``log(count)`` vs ``log(1/s)``.

    Parameters
    ----------
    binary_mask_3d : np.ndarray
        Boolean array of shape ``(n, n, n)`` where ``True`` marks
        boundary voxels.
    min_box_size : int
        Smallest box side length to use (default 1).  Must be a power
        of 2.
    max_box_size : int or None
        Largest box side length.  If ``None``, defaults to the largest
        power of 2 that is <= half the grid size (so at least 2x2x2
        boxes tile the volume).

    Returns
    -------
    dict
        ``"dimension"`` : float -- estimated fractal dimension.

        ``"std_error"`` : float -- standard error of the slope.

        ``"box_sizes"`` : list of int -- box sizes used.

        ``"counts"`` : list of int -- number of occupied boxes at each
        size.

        ``"r_squared"`` : float -- coefficient of determination of the
        linear fit.

    Raises
    ------
    ValueError
        If the mask contains no ``True`` voxels, or if fewer than two
        valid box sizes are available for the fit.
    """
    binary_mask_3d = np.asarray(binary_mask_3d, dtype=bool)

    if binary_mask_3d.ndim != 3:
        raise ValueError(
            f"Expected a 3D array, got shape {binary_mask_3d.shape}"
        )

    if not np.any(binary_mask_3d):
        raise ValueError(
            "The binary mask contains no True voxels.  Cannot compute "
            "fractal dimension on an empty set."
        )

    grid_size = binary_mask_3d.shape[0]

    # Determine the range of box sizes (powers of 2).
    if max_box_size is None:
        max_box_size = _largest_power_of_2_le(grid_size // 2)
        if max_box_size < 1:
            max_box_size = 1

    box_sizes: list[int] = []
    current_size = min_box_size
    while current_size <= max_box_size:
        box_sizes.append(current_size)
        current_size *= 2

    if len(box_sizes) < 2:
        raise ValueError(
            f"Need at least 2 box sizes for a fit, but only "
            f"{len(box_sizes)} size(s) available in range "
            f"[{min_box_size}, {max_box_size}].  Increase the grid "
            f"resolution or widen the box-size range."
        )

    # Count occupied boxes at each scale.
    counts: list[int] = []
    for box_side_length in box_sizes:
        occupied_count = _count_occupied_boxes(
            binary_mask_3d, box_side_length
        )
        counts.append(occupied_count)

    # Linear regression: log(count) vs log(1/s) = -log(s).
    log_inverse_sizes = np.log(1.0 / np.array(box_sizes, dtype=np.float64))
    log_counts = np.log(np.array(counts, dtype=np.float64))

    regression_result = linregress(log_inverse_sizes, log_counts)
    fractal_dimension = float(regression_result.slope)
    standard_error = float(regression_result.stderr)
    r_squared = float(regression_result.rvalue ** 2)

    return {
        "dimension": fractal_dimension,
        "std_error": standard_error,
        "box_sizes": box_sizes,
        "counts": counts,
        "r_squared": r_squared,
    }


# ---------------------------------------------------------------------------
# Boundary dimension estimation
# ---------------------------------------------------------------------------


def estimate_boundary_dimension(
    flip_times_3d: npt.NDArray[np.float64],
    threshold: float | None = None,
) -> dict[str, Any]:
    """Extract the chaos boundary and estimate its fractal dimension.

    Combines boundary extraction (from
    :func:`src.visualization.volume_render.extract_boundary_mask`) with
    box-counting fractal dimension estimation.

    Parameters
    ----------
    flip_times_3d : np.ndarray
        3-D array of shape ``(n, n, n)`` with flip-time values.
        ``np.nan`` marks voxels that never flipped.
    threshold : float or None
        Gradient magnitude threshold for boundary detection.  If
        ``None``, an adaptive threshold is used (see
        :func:`extract_boundary_mask`).

    Returns
    -------
    dict
        All fields from :func:`box_counting_dimension`, plus:

        ``"num_boundary_voxels"`` : int -- total number of boundary
        voxels detected.
    """
    boundary_mask = extract_boundary_mask(
        flip_times_3d, threshold=threshold
    )

    num_boundary_voxels = int(np.count_nonzero(boundary_mask))

    if num_boundary_voxels == 0:
        raise ValueError(
            "No boundary voxels detected.  Try adjusting the threshold "
            "or using a higher-resolution grid."
        )

    box_counting_result = box_counting_dimension(boundary_mask)
    box_counting_result["num_boundary_voxels"] = num_boundary_voxels

    return box_counting_result


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------


def dimension_convergence(
    flip_times_3d_list: list[npt.NDArray[np.float64]],
    grid_sizes: list[int],
) -> dict[str, Any]:
    """Compute fractal dimension at multiple grid resolutions.

    Used to verify that the estimated fractal dimension converges as
    grid resolution increases.  A converging estimate indicates that
    the box-counting method is capturing a genuine fractal property
    rather than a resolution artefact.

    Parameters
    ----------
    flip_times_3d_list : list of np.ndarray
        Flip-time volumes at increasing resolutions.  Each array has
        shape ``(n_i, n_i, n_i)`` where ``n_i`` is the corresponding
        entry in *grid_sizes*.
    grid_sizes : list of int
        Grid resolution for each volume.  Must have the same length as
        *flip_times_3d_list*.

    Returns
    -------
    dict
        ``"grid_sizes"`` : np.ndarray of int -- the input grid sizes.

        ``"dimensions"`` : np.ndarray of float -- fractal dimension at
        each resolution.

        ``"std_errors"`` : np.ndarray of float -- standard error of
        each dimension estimate.

    Raises
    ------
    ValueError
        If the two input lists have different lengths.
    """
    if len(flip_times_3d_list) != len(grid_sizes):
        raise ValueError(
            f"flip_times_3d_list has {len(flip_times_3d_list)} entries "
            f"but grid_sizes has {len(grid_sizes)} entries.  They must "
            f"match."
        )

    dimensions: list[float] = []
    standard_errors: list[float] = []

    for flip_times_volume, resolution in zip(
        flip_times_3d_list, grid_sizes
    ):
        result = estimate_boundary_dimension(flip_times_volume)
        dimensions.append(result["dimension"])
        standard_errors.append(result["std_error"])

    return {
        "grid_sizes": np.array(grid_sizes, dtype=int),
        "dimensions": np.array(dimensions, dtype=np.float64),
        "std_errors": np.array(standard_errors, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_box_counting(
    result: dict[str, Any],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Create a log-log plot of box count vs inverse box size.

    Plots the empirical data points and the fitted line whose slope is
    the estimated fractal dimension.

    Parameters
    ----------
    result : dict
        Output from :func:`box_counting_dimension` or
        :func:`estimate_boundary_dimension`.  Must contain keys
        ``"box_sizes"``, ``"counts"``, ``"dimension"``, ``"std_error"``,
        and ``"r_squared"``.
    output_path : str, Path, or None
        If given, save the figure to this path (PNG, PDF, etc.).
        Parent directories are created if needed.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    box_sizes = np.array(result["box_sizes"], dtype=np.float64)
    counts = np.array(result["counts"], dtype=np.float64)
    fractal_dimension = result["dimension"]
    standard_error = result["std_error"]
    r_squared = result["r_squared"]

    inverse_sizes = 1.0 / box_sizes
    log_inverse_sizes = np.log(inverse_sizes)
    log_counts = np.log(counts)

    # Reconstruct the fitted line from the regression.
    regression_result = linregress(log_inverse_sizes, log_counts)
    fitted_log_counts = (
        regression_result.slope * log_inverse_sizes
        + regression_result.intercept
    )

    figure, axes = plt.subplots(figsize=(8, 6))

    axes.plot(
        log_inverse_sizes,
        log_counts,
        "o",
        markersize=8,
        color="#2196F3",
        label="Data",
        zorder=3,
    )

    axes.plot(
        log_inverse_sizes,
        fitted_log_counts,
        "-",
        linewidth=2,
        color="#FF5722",
        label=(
            f"Fit: D = {fractal_dimension:.3f} "
            f"+/- {standard_error:.3f} "
            f"(R² = {r_squared:.4f})"
        ),
        zorder=2,
    )

    axes.set_xlabel("log(1 / box size)", fontsize=12)
    axes.set_ylabel("log(box count)", fontsize=12)
    axes.set_title("Box-Counting Fractal Dimension", fontsize=14)
    axes.legend(fontsize=11, loc="upper left")
    axes.grid(True, alpha=0.3)

    figure.tight_layout()

    if output_path is not None:
        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return figure


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _largest_power_of_2_le(value: int) -> int:
    """Return the largest power of 2 that is <= *value*.

    Parameters
    ----------
    value : int
        Upper bound (inclusive).

    Returns
    -------
    int
        Largest power of 2 <= *value*, or 1 if *value* < 1.
    """
    if value < 1:
        return 1
    power = 1
    while power * 2 <= value:
        power *= 2
    return power


def _count_occupied_boxes(
    binary_mask_3d: npt.NDArray[np.bool_],
    box_side_length: int,
) -> int:
    """Count how many boxes of the given size contain a True voxel.

    The volume is tiled with non-overlapping cubic boxes of side
    *box_side_length*.  Voxels beyond the grid boundary (when the grid
    size is not an exact multiple of the box size) are included in a
    partial box at the far edge of each axis.

    Parameters
    ----------
    binary_mask_3d : np.ndarray
        Boolean 3D array.
    box_side_length : int
        Side length of each cubic box.

    Returns
    -------
    int
        Number of boxes containing at least one ``True`` voxel.
    """
    shape = binary_mask_3d.shape

    if box_side_length == 1:
        # Fast path: each voxel is its own box.
        return int(np.count_nonzero(binary_mask_3d))

    # Pad the array to the next multiple of box_side_length along each
    # axis so that reshaping works cleanly.
    padded_shape = tuple(
        _ceil_to_multiple(dimension_size, box_side_length)
        for dimension_size in shape
    )

    if padded_shape != shape:
        padded_mask = np.zeros(padded_shape, dtype=bool)
        padded_mask[: shape[0], : shape[1], : shape[2]] = binary_mask_3d
    else:
        padded_mask = binary_mask_3d

    # Reshape into (num_boxes_x, s, num_boxes_y, s, num_boxes_z, s)
    # then reduce over the three box-interior axes.
    num_boxes_x = padded_shape[0] // box_side_length
    num_boxes_y = padded_shape[1] // box_side_length
    num_boxes_z = padded_shape[2] // box_side_length

    reshaped = padded_mask.reshape(
        num_boxes_x, box_side_length,
        num_boxes_y, box_side_length,
        num_boxes_z, box_side_length,
    )

    # any() over the interior axes (1, 3, 5) tells us which boxes are
    # occupied.
    occupied = reshaped.any(axis=(1, 3, 5))
    return int(np.count_nonzero(occupied))


def _ceil_to_multiple(value: int, multiple: int) -> int:
    """Round *value* up to the nearest multiple of *multiple*.

    Parameters
    ----------
    value : int
        The value to round up.
    multiple : int
        The multiple to round to.

    Returns
    -------
    int
        Smallest integer >= *value* that is divisible by *multiple*.
    """
    return ((value + multiple - 1) // multiple) * multiple
