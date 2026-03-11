"""Boundary detection and isosurface extraction from 3D flip-time data.

Provides tools to identify chaos boundaries in the triple pendulum's
initial-condition space and extract renderable geometry from the 3D
flip-time voxel grid.

**Boundary detection** (Issue #23) computes gradient magnitudes across
the flip-time volume to locate sharp transitions — the fractal boundaries
between regular and chaotic regions.  Two methods are supported:

* Gradient-based:  boundaries where ``|nabla(flip_time)|`` exceeds a
  threshold.
* Class-adjacency:  boundaries where neighbouring voxels fall in
  different discretised flip-time bins.

**Isosurface extraction** (Issue #24) uses marching cubes (via
scikit-image) to produce triangle meshes at constant flip-time levels.
These meshes can be exported as Wavefront OBJ files for use in Blender,
Three.js, or other 3D tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# scikit-image is an optional dependency — marching cubes is only needed
# for isosurface extraction.
try:
    from skimage.measure import marching_cubes  # type: ignore[import-untyped]

    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _replace_nan_with_sentinel(
    flip_times_3d: npt.NDArray[np.float64],
    sentinel_factor: float = 2.0,
) -> tuple[npt.NDArray[np.float64], float]:
    """Replace NaN values with a large sentinel before gradient computation.

    NaN entries (never-flipped voxels) would propagate through
    ``np.gradient``, corrupting neighbouring derivatives.  We replace
    them with ``t_max * sentinel_factor`` so that never-flipped regions
    create steep gradients at their borders — exactly the behaviour we
    want for boundary detection.

    Parameters
    ----------
    flip_times_3d : np.ndarray
        3-D array of shape ``(n, n, n)`` with flip times.  ``np.nan``
        marks voxels that never flipped.
    sentinel_factor : float
        Multiplier applied to the finite maximum to produce the sentinel
        value (default 2.0).

    Returns
    -------
    filled_array : np.ndarray
        Copy of *flip_times_3d* with NaN replaced by the sentinel.
    sentinel_value : float
        The sentinel value that was substituted for NaN.
    """
    finite_values = flip_times_3d[np.isfinite(flip_times_3d)]
    if finite_values.size == 0:
        # Every voxel is NaN — use a reasonable default.
        sentinel_value = 1.0
    else:
        max_finite_time = float(np.max(finite_values))
        sentinel_value = max_finite_time * sentinel_factor

    filled_array = np.where(
        np.isfinite(flip_times_3d), flip_times_3d, sentinel_value
    )
    return filled_array, sentinel_value


def _grid_index_to_theta(
    indices: npt.NDArray[np.floating[Any]],
    grid_resolution: int,
    theta_range: tuple[float, float],
) -> npt.NDArray[np.float64]:
    """Convert fractional grid indices to theta coordinates in degrees.

    Parameters
    ----------
    indices : np.ndarray
        Array of grid indices (may be fractional, e.g. from marching
        cubes vertex positions).  Shape ``(M, 3)`` or any broadcastable
        shape.
    grid_resolution : int
        Number of points along each axis.
    theta_range : tuple of float
        ``(theta_min, theta_max)`` in degrees.

    Returns
    -------
    np.ndarray
        Theta values in degrees, same shape as *indices*.
    """
    theta_min, theta_max = theta_range
    normalised = np.asarray(indices, dtype=np.float64) / max(grid_resolution - 1, 1)
    theta_values = theta_min + normalised * (theta_max - theta_min)
    return theta_values


# ---------------------------------------------------------------------------
# Boundary / edge detection  (Issue #23)
# ---------------------------------------------------------------------------


def compute_gradient_magnitude(
    flip_times_3d: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the gradient magnitude of a 3-D flip-time volume.

    NaN values (never-flipped voxels) are replaced with a sentinel
    before differentiation so that boundaries between flipped and
    never-flipped regions produce large gradient magnitudes.

    Parameters
    ----------
    flip_times_3d : np.ndarray
        3-D array of shape ``(n, n, n)`` with flip-time values.
        ``np.nan`` marks voxels that never flipped.

    Returns
    -------
    np.ndarray
        Gradient magnitude array of shape ``(n, n, n)``.  Higher values
        indicate sharper transitions in flip time between adjacent
        voxels — i.e. chaos boundaries.
    """
    filled_array, _sentinel = _replace_nan_with_sentinel(flip_times_3d)

    # np.gradient returns a list of partial derivatives, one per axis.
    partial_derivatives = np.gradient(filled_array)

    gradient_magnitude = np.sqrt(
        partial_derivatives[0] ** 2
        + partial_derivatives[1] ** 2
        + partial_derivatives[2] ** 2
    )
    return gradient_magnitude


def extract_boundary_mask(
    flip_times_3d: npt.NDArray[np.float64],
    threshold: float | None = None,
    method: str = "gradient",
    num_bins: int = 10,
) -> npt.NDArray[np.bool_]:
    """Identify boundary voxels in the 3-D flip-time volume.

    Two methods are available:

    * ``"gradient"`` (default) — marks voxels where the gradient
      magnitude exceeds a threshold.  If *threshold* is ``None``, an
      adaptive threshold of ``mean + 1 * std`` of non-zero gradient
      values is used.

    * ``"class_adjacency"`` — discretises flip times into *num_bins*
      bins and marks voxels that have at least one face-connected
      neighbour in a different bin.  This is useful when the gradient
      field is noisy but the chaos regions form well-separated clusters.

    Parameters
    ----------
    flip_times_3d : np.ndarray
        3-D array of shape ``(n, n, n)`` with flip-time values.
    threshold : float or None
        Gradient magnitude cutoff for the ``"gradient"`` method.
        Ignored when *method* is ``"class_adjacency"``.
    method : str
        Either ``"gradient"`` or ``"class_adjacency"``.
    num_bins : int
        Number of bins for the ``"class_adjacency"`` method (default 10).
        Ignored when *method* is ``"gradient"``.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(n, n, n)`` where ``True`` marks
        boundary voxels.

    Raises
    ------
    ValueError
        If *method* is not one of the recognised options.
    """
    if method == "gradient":
        return _boundary_mask_gradient(flip_times_3d, threshold)
    elif method == "class_adjacency":
        return _boundary_mask_class_adjacency(flip_times_3d, num_bins)
    else:
        raise ValueError(
            f"Unknown boundary method {method!r}. "
            f"Use 'gradient' or 'class_adjacency'."
        )


def _boundary_mask_gradient(
    flip_times_3d: npt.NDArray[np.float64],
    threshold: float | None,
) -> npt.NDArray[np.bool_]:
    """Gradient-based boundary detection (internal).

    Parameters
    ----------
    flip_times_3d : np.ndarray
        3-D flip-time volume.
    threshold : float or None
        If ``None``, uses adaptive threshold ``mean + 1*std`` of
        non-zero gradient magnitudes.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(n, n, n)``.
    """
    gradient_magnitude = compute_gradient_magnitude(flip_times_3d)

    if threshold is None:
        nonzero_gradients = gradient_magnitude[gradient_magnitude > 0.0]
        if nonzero_gradients.size == 0:
            # Perfectly uniform field — no boundaries at all.
            return np.zeros_like(gradient_magnitude, dtype=bool)
        adaptive_mean = float(np.mean(nonzero_gradients))
        adaptive_std = float(np.std(nonzero_gradients))
        threshold = adaptive_mean + adaptive_std

    boundary_mask: npt.NDArray[np.bool_] = gradient_magnitude >= threshold
    return boundary_mask


def _boundary_mask_class_adjacency(
    flip_times_3d: npt.NDArray[np.float64],
    num_bins: int,
) -> npt.NDArray[np.bool_]:
    """Class-adjacency boundary detection (internal).

    Discretises flip times into bins and marks voxels whose
    face-connected neighbours belong to a different bin.  NaN values
    (never-flipped) are assigned to a dedicated "no-flip" bin.

    Parameters
    ----------
    flip_times_3d : np.ndarray
        3-D flip-time volume.
    num_bins : int
        Number of equal-width bins for the finite flip-time range.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(n, n, n)``.
    """
    finite_mask = np.isfinite(flip_times_3d)
    finite_values = flip_times_3d[finite_mask]

    # Assign bin indices.  NaN voxels get bin index = num_bins (a
    # dedicated "never-flipped" class).
    bin_labels = np.full(flip_times_3d.shape, num_bins, dtype=np.intp)

    if finite_values.size > 0:
        min_time = float(np.min(finite_values))
        max_time = float(np.max(finite_values))
        bin_width = (max_time - min_time) / max(num_bins, 1)

        if bin_width == 0.0:
            # All finite values are identical — put them in bin 0.
            bin_labels[finite_mask] = 0
        else:
            raw_bin_indices = ((flip_times_3d[finite_mask] - min_time) / bin_width).astype(
                np.intp
            )
            # Clamp the top edge into the last valid bin.
            raw_bin_indices = np.clip(raw_bin_indices, 0, num_bins - 1)
            bin_labels[finite_mask] = raw_bin_indices

    # Check six face-connected neighbours for class differences.
    boundary_mask = np.zeros(flip_times_3d.shape, dtype=bool)
    for axis in range(3):
        # Forward neighbour comparison.
        slices_current = [slice(None)] * 3
        slices_neighbour = [slice(None)] * 3
        slices_current[axis] = slice(None, -1)
        slices_neighbour[axis] = slice(1, None)

        different_forward = (
            bin_labels[tuple(slices_current)] != bin_labels[tuple(slices_neighbour)]
        )
        # Mark both the current voxel and its neighbour as boundary.
        boundary_mask[tuple(slices_current)] |= different_forward
        boundary_mask[tuple(slices_neighbour)] |= different_forward

    return boundary_mask


def extract_boundary_points(
    flip_times_3d: npt.NDArray[np.float64],
    theta_range: tuple[float, float] = (-170.0, 170.0),
    threshold: float | None = None,
    method: str = "gradient",
    num_bins: int = 10,
) -> dict[str, Any]:
    """Extract boundary voxel positions and their flip-time values.

    Combines :func:`extract_boundary_mask` with coordinate mapping to
    return the spatial positions (in theta-degrees) and flip-time values
    of every boundary voxel.

    Parameters
    ----------
    flip_times_3d : np.ndarray
        3-D array of shape ``(n, n, n)`` with flip-time values.
    theta_range : tuple of float
        ``(theta_min, theta_max)`` in degrees (default ``(-170, 170)``).
    threshold : float or None
        Gradient threshold.  Passed through to
        :func:`extract_boundary_mask`.
    method : str
        Boundary detection method (``"gradient"`` or
        ``"class_adjacency"``).
    num_bins : int
        Number of bins for the ``"class_adjacency"`` method.

    Returns
    -------
    dict
        ``"positions"`` : ``(M, 3)`` array of theta coordinates in
        degrees for each boundary voxel.

        ``"values"`` : ``(M,)`` array of flip-time values at those
        positions (``np.nan`` for voxels that never flipped).

        ``"mask"`` : ``(n, n, n)`` boolean boundary mask.
    """
    boundary_mask = extract_boundary_mask(
        flip_times_3d,
        threshold=threshold,
        method=method,
        num_bins=num_bins,
    )

    grid_resolution = flip_times_3d.shape[0]

    # np.argwhere returns (M, 3) integer indices of True entries.
    boundary_indices = np.argwhere(boundary_mask)

    boundary_positions = _grid_index_to_theta(
        boundary_indices, grid_resolution, theta_range
    )

    boundary_values = flip_times_3d[boundary_mask]

    return {
        "positions": boundary_positions,
        "values": boundary_values,
        "mask": boundary_mask,
    }


# ---------------------------------------------------------------------------
# Isosurface extraction  (Issue #24)
# ---------------------------------------------------------------------------


def _require_skimage() -> None:
    """Raise a clear error when scikit-image is not installed."""
    if not _HAS_SKIMAGE:
        raise ImportError(
            "scikit-image is required for isosurface extraction but is "
            "not installed.  Install it with:  pip install scikit-image"
        )


def extract_isosurface(
    flip_times_3d: npt.NDArray[np.float64],
    level: float,
    theta_range: tuple[float, float] = (-170.0, 170.0),
) -> dict[str, npt.NDArray[np.float64]]:
    """Extract a triangle-mesh isosurface at a given flip-time level.

    Uses the marching-cubes algorithm (via ``skimage.measure``) to find
    the surface where ``flip_time == level`` and maps vertex coordinates
    from grid indices to theta-degree space.

    NaN values are replaced with a sentinel (``2 * max_finite_time``)
    before extraction so that the isosurface correctly separates flipped
    from never-flipped regions when *level* falls near the finite
    maximum.

    Parameters
    ----------
    flip_times_3d : np.ndarray
        3-D array of shape ``(n, n, n)`` with flip-time values.
    level : float
        The flip-time iso-value at which to extract the surface.
    theta_range : tuple of float
        ``(theta_min, theta_max)`` in degrees (default ``(-170, 170)``).

    Returns
    -------
    dict
        ``"vertices"`` : ``(V, 3)`` array of vertex positions in
        theta-degree coordinates.

        ``"faces"`` : ``(F, 3)`` integer array of triangle face indices
        into the vertex array.

        ``"normals"`` : ``(V, 3)`` array of per-vertex surface normals
        (unit vectors, in grid-index space).

    Raises
    ------
    ImportError
        If scikit-image is not installed.
    ValueError
        If marching cubes finds no surface at the requested level (e.g.
        the level is outside the data range).
    """
    _require_skimage()

    filled_array, _sentinel = _replace_nan_with_sentinel(flip_times_3d)
    grid_resolution = flip_times_3d.shape[0]

    try:
        vertices_grid, faces, normals, _values = marching_cubes(
            filled_array, level=level
        )
    except (ValueError, RuntimeError) as exc:
        raise ValueError(
            f"Marching cubes failed at level={level!r}.  This usually "
            f"means the iso-level is outside the data range.  "
            f"Finite data range: "
            f"[{np.nanmin(flip_times_3d)}, {np.nanmax(flip_times_3d)}]."
        ) from exc

    vertices_theta = _grid_index_to_theta(
        vertices_grid, grid_resolution, theta_range
    )

    return {
        "vertices": vertices_theta,
        "faces": faces.astype(np.intp),
        "normals": normals.astype(np.float64),
    }


def extract_multi_isosurface(
    flip_times_3d: npt.NDArray[np.float64],
    levels: list[float] | npt.NDArray[np.float64],
    theta_range: tuple[float, float] = (-170.0, 170.0),
) -> list[dict[str, npt.NDArray[np.float64]]]:
    """Extract multiple nested isosurfaces at different flip-time levels.

    This is a convenience wrapper around :func:`extract_isosurface` for
    producing layered visualisations (e.g. nested shells of increasing
    flip time).

    Parameters
    ----------
    flip_times_3d : np.ndarray
        3-D array of shape ``(n, n, n)`` with flip-time values.
    levels : list of float or np.ndarray
        Sequence of flip-time iso-values.  Each value produces one
        surface.
    theta_range : tuple of float
        ``(theta_min, theta_max)`` in degrees (default ``(-170, 170)``).

    Returns
    -------
    list of dict
        One dict per level, each with ``"vertices"``, ``"faces"``, and
        ``"normals"`` keys (see :func:`extract_isosurface`).  Levels
        that produce no surface are silently skipped, so the returned
        list may be shorter than *levels*.

    Raises
    ------
    ImportError
        If scikit-image is not installed.
    """
    _require_skimage()

    isosurface_meshes: list[dict[str, npt.NDArray[np.float64]]] = []
    for iso_level in levels:
        try:
            mesh = extract_isosurface(
                flip_times_3d, level=float(iso_level), theta_range=theta_range
            )
            isosurface_meshes.append(mesh)
        except ValueError:
            # This level produced no surface — skip it.
            continue

    return isosurface_meshes


# ---------------------------------------------------------------------------
# Mesh export
# ---------------------------------------------------------------------------


def save_mesh_obj(
    path: str | Path,
    vertices: npt.NDArray[np.float64],
    faces: npt.NDArray[np.intp],
) -> None:
    """Export a triangle mesh as a Wavefront OBJ file.

    The OBJ format is widely supported by 3D applications (Blender,
    Three.js, MeshLab, etc.).  Only vertex positions and triangular
    faces are written; normals, UVs, and materials are omitted for
    simplicity.

    Parameters
    ----------
    path : str or Path
        Destination file path.  Parent directories are created if
        needed.
    vertices : np.ndarray
        Vertex positions of shape ``(V, 3)``.
    faces : np.ndarray
        Triangle face indices of shape ``(F, 3)``.  Each row contains
        three zero-based indices into *vertices*.  They are written as
        1-indexed values per the OBJ specification.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.intp)

    with open(output_path, "w", encoding="utf-8") as obj_file:
        obj_file.write("# Triple Pendulum Chaos Isosurface\n")
        obj_file.write(f"# Vertices: {vertex_array.shape[0]}\n")
        obj_file.write(f"# Faces: {face_array.shape[0]}\n\n")

        # Write vertex positions: "v x y z"
        for vertex_x, vertex_y, vertex_z in vertex_array:
            obj_file.write(f"v {vertex_x:.6f} {vertex_y:.6f} {vertex_z:.6f}\n")

        obj_file.write("\n")

        # Write face indices: "f i j k" (OBJ uses 1-based indexing)
        for index_a, index_b, index_c in face_array:
            obj_file.write(f"f {index_a + 1} {index_b + 1} {index_c + 1}\n")
