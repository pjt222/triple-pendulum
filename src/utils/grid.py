"""Initial condition grid construction for triple pendulum simulations.

Builds uniform grids of (theta1, theta2, theta3) initial angles spanning
the configured range. The default range of +/-170 degrees avoids the
singularity at +/-180 degrees where the mass matrix becomes degenerate.

Two grid types are supported:

* **Cube** -- uniform Cartesian grid in (theta1, theta2, theta3) space.
* **Sphere** -- concentric spherical shells with Fibonacci spiral sampling,
  inscribed in the cube with radius r_max = 170 degrees.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

# Golden ratio, used by the Fibonacci sphere algorithm.
_GOLDEN_RATIO = (1.0 + np.sqrt(5.0)) / 2.0


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


# ---------------------------------------------------------------------------
# Sphere grid  (Issue #65)
# ---------------------------------------------------------------------------


def fibonacci_sphere(
    num_points: int,
) -> npt.NDArray[np.float64]:
    """Generate quasi-uniform points on the unit sphere using the Fibonacci spiral.

    The golden-angle spiral distributes points with near-uniform spacing,
    avoiding the polar clustering that plagues latitude/longitude grids.

    Parameters
    ----------
    num_points : int
        Number of points to place on the unit sphere.  Must be >= 1.

    Returns
    -------
    np.ndarray
        Array of shape ``(num_points, 3)`` with Cartesian ``(x, y, z)``
        coordinates on the unit sphere.
    """
    indices = np.arange(num_points, dtype=np.float64)

    # Polar angle (colatitude): uniformly spaced in cos(phi) to avoid
    # pole clustering.  The +0.5 offset centres each point in its band.
    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / num_points)

    # Azimuthal angle: golden-angle increments produce the spiral.
    theta_azimuthal = 2.0 * np.pi * indices / _GOLDEN_RATIO

    sin_phi = np.sin(phi)
    x = sin_phi * np.cos(theta_azimuthal)
    y = sin_phi * np.sin(theta_azimuthal)
    z = np.cos(phi)

    return np.column_stack([x, y, z])


def make_sphere_grid(
    resolution: int,
    r_max: float = 170.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]:
    """Create a solid-sphere grid of initial-angle triplets.

    The sphere is built from concentric shells of Fibonacci-spiral points.
    The number of points per shell scales as ``(r / r_max)**2`` to maintain
    uniform volume density.  The total point count is approximately
    ``(pi/6) * resolution**3``, matching the volume ratio of an inscribed
    sphere to its bounding cube.

    Parameters
    ----------
    resolution : int
        Number of radial shells (excluding the origin).  Also controls
        point density: higher resolution = more shells and more points
        per shell.
    r_max : float
        Maximum radius in degrees (default 170).  No individual angle
        component will exceed this value since ``|sin * cos| <= 1``
        and the radius caps at *r_max*.

    Returns
    -------
    thetas : np.ndarray
        Array of shape ``(M, 3)`` with ``(theta1, theta2, theta3)`` in
        degrees — the physical initial conditions for simulation.
    positions : np.ndarray
        Array of shape ``(M, 3)`` with viewer coordinates normalised to
        ``[-1, 1]``.  These are the Cartesian coordinates of each point
        scaled so the outermost shell sits at unit radius.
    metadata : dict
        Grid metadata including ``grid_type``, ``resolution``, ``r_max``,
        ``num_shells``, ``total_points``, ``shell_radii``, and
        ``points_per_shell``.
    """
    # Calibrate so total points ~ (pi/6) * resolution^3.
    #
    # Total = 1 + sum_{k=1}^{N} points_per_full_shell * (k/N)^2
    #       ~ points_per_full_shell * N / 3   (integral approximation)
    #
    # Set this equal to (pi/6) * N^3:
    #   points_per_full_shell * N / 3 = (pi/6) * N^3
    #   points_per_full_shell = (pi/2) * N^2
    points_per_full_shell = int(np.round((np.pi / 2.0) * resolution ** 2))

    r_step = r_max / resolution

    # Origin: single degenerate point at (0, 0, 0).
    all_thetas = [np.zeros((1, 3))]
    all_positions = [np.zeros((1, 3))]

    shell_radii: list[float] = []
    points_per_shell: list[int] = []

    for shell_index in range(1, resolution + 1):
        radius = shell_index * r_step
        shell_fraction = shell_index / resolution

        # Number of points on this shell (quadratic scaling for volume density).
        num_shell_points = max(1, int(np.round(
            points_per_full_shell * shell_fraction ** 2
        )))

        # Fibonacci points on unit sphere, scaled to this shell's radius.
        unit_points = fibonacci_sphere(num_shell_points)
        shell_thetas = unit_points * radius          # degrees
        shell_positions = unit_points * shell_fraction  # normalised to [-1, 1]

        all_thetas.append(shell_thetas)
        all_positions.append(shell_positions)
        shell_radii.append(float(radius))
        points_per_shell.append(num_shell_points)

    thetas = np.vstack(all_thetas)
    positions = np.vstack(all_positions)

    total_points = thetas.shape[0]

    metadata: dict[str, Any] = {
        "grid_type": "sphere",
        "resolution": resolution,
        "r_max": float(r_max),
        "num_shells": resolution,
        "total_points": total_points,
        "shell_radii": shell_radii,
        "points_per_shell": points_per_shell,
    }

    return thetas, positions, metadata
