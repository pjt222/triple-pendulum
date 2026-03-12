"""Initial condition grid construction for triple pendulum simulations.

Builds uniform grids of (theta1, theta2, theta3) initial angles spanning
the configured range. The default range of +/-170 degrees avoids the
singularity at +/-180 degrees where the mass matrix becomes degenerate.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def make_grid(
    points_per_axis: int,
    theta_min: float = -170.0,
    theta_max: float = 170.0,
) -> npt.NDArray[np.float64]:
    """Create a flat array of all initial-angle triplets on a uniform grid.

    Parameters
    ----------
    points_per_axis : int
        Number of uniformly spaced sample points along each theta axis.
        Total number of grid points is ``points_per_axis ** 3``.
    theta_min : float
        Lower bound of the angle range in degrees (default -170).
    theta_max : float
        Upper bound of the angle range in degrees (default 170).

    Returns
    -------
    np.ndarray
        Array of shape ``(points_per_axis**3, 3)`` where each row holds
        ``(theta1, theta2, theta3)`` in degrees.
    """
    axis_values = np.linspace(theta_min, theta_max, points_per_axis)
    theta1_grid, theta2_grid, theta3_grid = np.meshgrid(
        axis_values, axis_values, axis_values, indexing="ij"
    )
    initial_conditions = np.column_stack(
        [theta1_grid.ravel(), theta2_grid.ravel(), theta3_grid.ravel()]
    )
    return initial_conditions


def make_grid_chunks(
    points_per_axis: int,
    chunk_size: int = 1_000_000,
    theta_min: float = -170.0,
    theta_max: float = 170.0,
):
    """Yield chunks of initial-angle triplets without materializing the full grid.

    This generator produces the same results as :func:`make_grid` but yields
    manageable slices, enabling simulation of grids that exceed available RAM
    (e.g. 1000^3 = 1 billion points ~ 24 GB).

    Parameters
    ----------
    points_per_axis : int
        Number of uniformly spaced sample points along each theta axis.
    chunk_size : int
        Maximum number of grid points per yielded chunk (default 1M).
    theta_min : float
        Lower bound of the angle range in degrees (default -170).
    theta_max : float
        Upper bound of the angle range in degrees (default 170).

    Yields
    ------
    tuple of (np.ndarray, int, int)
        ``(chunk_thetas, start_idx, end_idx)`` where *chunk_thetas* has shape
        ``(end_idx - start_idx, 3)`` with angles in degrees, and *start_idx* /
        *end_idx* are flat indices into the full ``points_per_axis**3`` grid.
    """
    axis_values = np.linspace(theta_min, theta_max, points_per_axis)
    total = points_per_axis ** 3
    n = points_per_axis

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        flat_indices = np.arange(start, end)
        i = flat_indices // (n * n)
        j = (flat_indices // n) % n
        k = flat_indices % n
        chunk_thetas = np.column_stack([axis_values[i], axis_values[j], axis_values[k]])
        yield chunk_thetas, start, end


def grid_to_indices(
    thetas: npt.NDArray[np.float64],
    points_per_axis: int,
    theta_min: float = -170.0,
    theta_max: float = 170.0,
) -> npt.NDArray[np.intp]:
    """Map angle values back to integer grid indices.

    Each angle is snapped to the nearest grid index along its axis using
    the same linspace that :func:`make_grid` produces.

    Parameters
    ----------
    thetas : np.ndarray
        Array of shape ``(M, 3)`` with angle triplets in degrees.
    points_per_axis : int
        Number of points per axis (must match the grid that produced *thetas*).
    theta_min : float
        Lower bound of the angle range in degrees (default -170).
    theta_max : float
        Upper bound of the angle range in degrees (default 170).

    Returns
    -------
    np.ndarray
        Integer index array of shape ``(M, 3)`` with values in
        ``[0, points_per_axis - 1]``.
    """
    normalized = (np.asarray(thetas, dtype=np.float64) - theta_min) / (theta_max - theta_min)
    fractional_indices = normalized * (points_per_axis - 1)
    integer_indices = np.rint(fractional_indices).astype(np.intp)
    integer_indices = np.clip(integer_indices, 0, points_per_axis - 1)
    return integer_indices


def make_grid_3d(
    points_per_axis: int,
    theta_min: float = -170.0,
    theta_max: float = 170.0,
) -> npt.NDArray[np.float64]:
    """Create a 3-D meshgrid array of initial-angle triplets.

    Unlike :func:`make_grid` which returns a flat ``(N**3, 3)`` array,
    this function returns the angles arranged in a volumetric
    ``(n, n, n, 3)`` array suitable for direct indexing and volumetric
    rendering operations.

    Parameters
    ----------
    points_per_axis : int
        Number of uniformly spaced sample points along each theta axis.
    theta_min : float
        Lower bound of the angle range in degrees (default -170).
    theta_max : float
        Upper bound of the angle range in degrees (default 170).

    Returns
    -------
    np.ndarray
        Array of shape ``(points_per_axis, points_per_axis, points_per_axis, 3)``
        where ``result[i, j, k]`` gives ``(theta1, theta2, theta3)`` in
        degrees for grid cell ``(i, j, k)``.
    """
    axis_values = np.linspace(theta_min, theta_max, points_per_axis)
    theta1_grid, theta2_grid, theta3_grid = np.meshgrid(
        axis_values, axis_values, axis_values, indexing="ij"
    )
    volume = np.stack([theta1_grid, theta2_grid, theta3_grid], axis=-1)
    return volume
