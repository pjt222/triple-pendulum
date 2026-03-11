"""Adaptive resolution exploration for triple pendulum chaos maps.

Starts with a coarse grid and progressively refines near interesting
regions (fractal boundaries between chaotic and non-chaotic zones).
Uses an octree-like multi-resolution structure so that compute is
concentrated where the chaos metric changes most rapidly, while
featureless interior volumes stay at low resolution.

Typical workflow::

    from src.visualization.adaptive import AdaptiveGrid

    grid = AdaptiveGrid(base_resolution=10, max_resolution=80)
    grid.compute_base(dt=0.01, t_max=15.0)
    grid.refine_boundaries()          # 10 -> 20
    grid.refine_boundaries()          # 20 -> 40
    grid.refine_boundaries()          # 40 -> 80
    grid.export_to_json("data/adaptive_result.json")

Or use the convenience function::

    from src.visualization.adaptive import create_progressive_visualization
    create_progressive_visualization()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.simulation.batch_sim import simulate_batch
from src.utils.grid import make_grid


# ---------------------------------------------------------------------------
# Adaptive grid data structure
# ---------------------------------------------------------------------------


class _ResolutionLevel:
    """Internal storage for one level of detail in the adaptive grid.

    Attributes
    ----------
    resolution : int
        Number of points per axis at this level.
    positions : np.ndarray
        Computed initial-condition triplets, shape ``(M, 3)`` in degrees.
    values : np.ndarray
        Flip-time results for each position, shape ``(M,)``.
    """

    __slots__ = ("resolution", "positions", "values")

    def __init__(
        self,
        resolution: int,
        positions: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
    ) -> None:
        self.resolution = resolution
        self.positions = np.asarray(positions, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)


class AdaptiveGrid:
    """Multi-resolution grid for adaptive exploration of chaos maps.

    Manages an octree-like collection of simulation results at
    multiple levels of detail.  Each refinement pass simulates only
    the grid points that have not already been computed, avoiding
    redundant work.

    Parameters
    ----------
    base_resolution : int
        Number of points per axis for the initial coarse grid
        (default 10, giving 1 000 total points).
    max_resolution : int
        Maximum target resolution per axis (default 80, giving
        512 000 total points at full refinement).
    theta_range : tuple of float
        ``(theta_min, theta_max)`` in degrees (default ``(-170, 170)``
        to avoid the singularity at +/- 180).
    """

    def __init__(
        self,
        base_resolution: int = 10,
        max_resolution: int = 80,
        theta_range: tuple[float, float] = (-170, 170),
    ) -> None:
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.theta_range = theta_range

        # Resolution levels, keyed by their resolution integer.
        self._levels: dict[int, _ResolutionLevel] = {}

        # Tracks the current "effective" resolution (highest level computed
        # so far, even if only partially filled).
        self._current_resolution: int = base_resolution

    # ----- public properties ------------------------------------------------

    @property
    def total_computed(self) -> int:
        """Total number of unique grid points that have been simulated."""
        return sum(level.positions.shape[0] for level in self._levels.values())

    @property
    def total_cells_at_max_res(self) -> int:
        """Total number of cells in a full grid at *max_resolution*."""
        return self.max_resolution ** 3

    # ----- core methods -----------------------------------------------------

    def compute_base(
        self,
        dt: float = 0.01,
        t_max: float = 15.0,
    ) -> None:
        """Simulate the base coarse grid.

        Populates the lowest resolution level with flip-time values
        for every point on the uniform ``base_resolution ** 3`` grid.

        Parameters
        ----------
        dt : float
            RK4 integration timestep in seconds.
        t_max : float
            Maximum simulation time in seconds.
        """
        theta_min, theta_max = self.theta_range
        base_positions = make_grid(
            self.base_resolution,
            theta_min=theta_min,
            theta_max=theta_max,
        )

        print(
            f"AdaptiveGrid: computing base grid "
            f"({self.base_resolution}^3 = {base_positions.shape[0]} points)"
        )

        simulation_results = simulate_batch(
            base_positions,
            dt=dt,
            t_max=t_max,
        )

        flip_times = simulation_results["flip_times"]

        self._levels[self.base_resolution] = _ResolutionLevel(
            resolution=self.base_resolution,
            positions=base_positions,
            values=flip_times,
        )
        self._current_resolution = self.base_resolution

    def refine_region(
        self,
        center_theta: tuple[float, float, float],
        radius: float = 30.0,
        target_resolution: int | None = None,
        dt: float = 0.01,
        t_max: float = 15.0,
    ) -> int:
        """Refine a spherical region around *center_theta*.

        Builds a finer grid within a sphere of the given radius
        (in degrees) centred on the supplied angle triplet, then
        simulates only those new grid points that have not yet been
        computed at any existing resolution level.

        Parameters
        ----------
        center_theta : tuple of float
            ``(theta1, theta2, theta3)`` centre of the refinement
            region in degrees.
        radius : float
            Radius of the spherical region in degrees (default 30).
        target_resolution : int, optional
            Points per axis for the refined sub-grid.  Defaults to
            ``2 * current_resolution``, clamped to *max_resolution*.
        dt : float
            RK4 integration timestep.
        t_max : float
            Maximum simulation time.

        Returns
        -------
        int
            Number of new points that were actually simulated.
        """
        if target_resolution is None:
            target_resolution = min(
                self._current_resolution * 2, self.max_resolution
            )
        target_resolution = min(target_resolution, self.max_resolution)

        theta_min, theta_max = self.theta_range
        candidate_positions = make_grid(
            target_resolution,
            theta_min=theta_min,
            theta_max=theta_max,
        )

        # Filter to the spherical region.
        center_array = np.asarray(center_theta, dtype=np.float64)
        distances = np.linalg.norm(candidate_positions - center_array, axis=1)
        inside_sphere_mask = distances <= radius
        region_positions = candidate_positions[inside_sphere_mask]

        # Remove points that are already computed at this resolution.
        new_positions = self._filter_already_computed(
            region_positions, target_resolution
        )

        if new_positions.shape[0] == 0:
            print(
                f"AdaptiveGrid.refine_region: 0 new points in sphere "
                f"around {center_theta} (radius={radius})"
            )
            return 0

        print(
            f"AdaptiveGrid.refine_region: simulating {new_positions.shape[0]} "
            f"new points (res={target_resolution}, "
            f"centre={center_theta}, radius={radius})"
        )

        simulation_results = simulate_batch(
            new_positions, dt=dt, t_max=t_max
        )
        new_flip_times = simulation_results["flip_times"]

        self._append_to_level(target_resolution, new_positions, new_flip_times)
        self._current_resolution = max(
            self._current_resolution, target_resolution
        )

        return int(new_positions.shape[0])

    def refine_boundaries(
        self,
        threshold: float | None = None,
        target_resolution: int | None = None,
        dt: float = 0.01,
        t_max: float = 15.0,
    ) -> int:
        """Auto-detect boundary regions and refine them.

        Boundaries are identified by computing the gradient magnitude
        of the flip-time field on the current coarsest available grid.
        Cells whose gradient magnitude exceeds *threshold* are
        expanded to a finer grid and simulated.

        Parameters
        ----------
        threshold : float, optional
            Gradient magnitude cutoff.  Points above this threshold
            are considered boundaries.  Defaults to the 60th
            percentile of the finite gradient magnitudes, which
            typically captures the most interesting fractal edges.
        target_resolution : int, optional
            Resolution for the refined points.  Defaults to
            ``2 * current_resolution``, clamped to *max_resolution*.
        dt : float
            RK4 integration timestep.
        t_max : float
            Maximum simulation time.

        Returns
        -------
        int
            Number of new points that were simulated.
        """
        if target_resolution is None:
            target_resolution = min(
                self._current_resolution * 2, self.max_resolution
            )
        target_resolution = min(target_resolution, self.max_resolution)

        # Use the most recent complete level to detect boundaries.
        reference_level = self._get_best_complete_level()
        if reference_level is None:
            raise RuntimeError(
                "No computed level available.  Call compute_base() first."
            )

        reference_resolution = reference_level.resolution
        boundary_positions = self._detect_boundary_positions(
            reference_level, threshold=threshold
        )

        if boundary_positions.shape[0] == 0:
            print("AdaptiveGrid.refine_boundaries: no boundary cells detected")
            return 0

        # For each boundary cell at the reference resolution, generate the
        # sub-grid of finer points that fall within that cell.
        refined_positions = self._expand_boundary_cells(
            boundary_positions,
            source_resolution=reference_resolution,
            target_resolution=target_resolution,
        )

        # Remove already-computed points.
        new_positions = self._filter_already_computed(
            refined_positions, target_resolution
        )

        if new_positions.shape[0] == 0:
            print(
                "AdaptiveGrid.refine_boundaries: all boundary points "
                "already computed"
            )
            return 0

        print(
            f"AdaptiveGrid.refine_boundaries: {boundary_positions.shape[0]} "
            f"boundary cells -> {new_positions.shape[0]} new points "
            f"(res {reference_resolution} -> {target_resolution})"
        )

        simulation_results = simulate_batch(
            new_positions, dt=dt, t_max=t_max
        )
        new_flip_times = simulation_results["flip_times"]

        self._append_to_level(target_resolution, new_positions, new_flip_times)
        self._current_resolution = max(
            self._current_resolution, target_resolution
        )

        return int(new_positions.shape[0])

    def get_combined_data(self) -> dict[str, npt.NDArray[np.float64]]:
        """Return all computed points across every resolution level.

        Returns
        -------
        dict
            ``"positions"`` : ``(M, 3)`` array of angle triplets in degrees.
            ``"values"``    : ``(M,)`` array of flip-time values.
            ``"resolutions"``: ``(M,)`` array with the resolution level
            at which each point was computed.
        """
        if not self._levels:
            empty_positions = np.empty((0, 3), dtype=np.float64)
            empty_values = np.empty((0,), dtype=np.float64)
            empty_resolutions = np.empty((0,), dtype=np.float64)
            return {
                "positions": empty_positions,
                "values": empty_values,
                "resolutions": empty_resolutions,
            }

        all_positions_list: list[npt.NDArray[np.float64]] = []
        all_values_list: list[npt.NDArray[np.float64]] = []
        all_resolutions_list: list[npt.NDArray[np.float64]] = []

        for resolution, level in sorted(self._levels.items()):
            point_count = level.positions.shape[0]
            all_positions_list.append(level.positions)
            all_values_list.append(level.values)
            all_resolutions_list.append(
                np.full(point_count, resolution, dtype=np.float64)
            )

        combined_positions = np.concatenate(all_positions_list, axis=0)
        combined_values = np.concatenate(all_values_list, axis=0)
        combined_resolutions = np.concatenate(all_resolutions_list, axis=0)

        return {
            "positions": combined_positions,
            "values": combined_values,
            "resolutions": combined_resolutions,
        }

    def export_to_json(self, path: str | Path) -> Path:
        """Export combined data for the Three.js viewer.

        Writes a JSON file with the same top-level schema used by
        :func:`~src.utils.io.save_results_json` (``grid_size``,
        ``theta_range``, ``flip_times``, ``metadata``), plus an
        additional ``"adaptive"`` block containing per-point positions
        and resolution levels for renderers that support multi-res
        point clouds.

        Parameters
        ----------
        path : str or Path
            Destination JSON file path.

        Returns
        -------
        Path
            Resolved output path.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        combined_data = self.get_combined_data()
        positions = combined_data["positions"]
        values = combined_data["values"]
        resolutions = combined_data["resolutions"]

        # Build the viewer-compatible flip_times list.
        # NaN -> null in JSON.
        serializable_flip_times: list[float | None] = [
            None if np.isnan(value) else float(value)
            for value in values
        ]

        # Per-point position data for multi-res renderers.
        serializable_positions: list[list[float]] = [
            [float(position[0]), float(position[1]), float(position[2])]
            for position in positions
        ]

        resolution_levels_used = sorted(self._levels.keys())
        points_per_level = {
            str(resolution): int(level.positions.shape[0])
            for resolution, level in sorted(self._levels.items())
        }

        document: dict[str, Any] = {
            "grid_size": self._current_resolution,
            "theta_range": list(self.theta_range),
            "flip_times": serializable_flip_times,
            "metadata": {
                "adaptive": True,
                "base_resolution": self.base_resolution,
                "max_resolution": self.max_resolution,
                "current_resolution": self._current_resolution,
                "total_computed": self.total_computed,
                "total_cells_at_max_res": self.total_cells_at_max_res,
                "resolution_levels": resolution_levels_used,
                "points_per_level": points_per_level,
            },
            "adaptive": {
                "positions": serializable_positions,
                "resolutions": [int(r) for r in resolutions],
            },
        }

        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(document, json_file)

        total_points = positions.shape[0]
        finite_count = int(np.sum(np.isfinite(values)))
        print(
            f"AdaptiveGrid: exported {total_points} points "
            f"({finite_count} flipped) to {output_path}"
        )

        return output_path

    # ----- internal helpers -------------------------------------------------

    def _get_best_complete_level(self) -> _ResolutionLevel | None:
        """Return the highest-resolution level that has data, or None."""
        if not self._levels:
            return None
        highest_resolution = max(self._levels.keys())
        return self._levels[highest_resolution]

    def _filter_already_computed(
        self,
        candidate_positions: npt.NDArray[np.float64],
        target_resolution: int,
    ) -> npt.NDArray[np.float64]:
        """Remove candidate positions that already exist in the grid.

        Two positions are considered identical if they are within half
        the grid spacing at *target_resolution* of each other.
        """
        if not self._levels:
            return candidate_positions

        theta_min, theta_max = self.theta_range
        grid_spacing = (theta_max - theta_min) / max(target_resolution - 1, 1)
        tolerance = grid_spacing * 0.5

        # Collect all existing positions into a single array.
        existing_positions_list = [
            level.positions for level in self._levels.values()
        ]
        all_existing_positions = np.concatenate(existing_positions_list, axis=0)

        # For efficiency with moderately sized grids, use a rounding-based
        # deduplication rather than pairwise distance.  Snap candidates and
        # existing points to a fine grid, then compare tuples.
        snap_resolution = 1.0 / max(tolerance, 1e-12)
        existing_keys = set()
        for existing_position in all_existing_positions:
            quantized_key = tuple(
                np.round(existing_position * snap_resolution).astype(np.int64)
            )
            existing_keys.add(quantized_key)

        is_new_mask = np.ones(candidate_positions.shape[0], dtype=bool)
        for candidate_index, candidate_position in enumerate(candidate_positions):
            quantized_key = tuple(
                np.round(candidate_position * snap_resolution).astype(np.int64)
            )
            if quantized_key in existing_keys:
                is_new_mask[candidate_index] = False

        return candidate_positions[is_new_mask]

    def _append_to_level(
        self,
        resolution: int,
        positions: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
    ) -> None:
        """Append new simulation results to the level at *resolution*."""
        if resolution in self._levels:
            existing_level = self._levels[resolution]
            merged_positions = np.concatenate(
                [existing_level.positions, positions], axis=0
            )
            merged_values = np.concatenate(
                [existing_level.values, values], axis=0
            )
            self._levels[resolution] = _ResolutionLevel(
                resolution=resolution,
                positions=merged_positions,
                values=merged_values,
            )
        else:
            self._levels[resolution] = _ResolutionLevel(
                resolution=resolution,
                positions=positions,
                values=values,
            )

    def _detect_boundary_positions(
        self,
        level: _ResolutionLevel,
        threshold: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """Find grid positions near boundaries using gradient magnitude.

        Reconstructs a 3-D volume from the level's flat data, computes
        central-difference gradients along each axis, and identifies
        cells whose gradient magnitude exceeds *threshold*.

        Returns
        -------
        np.ndarray
            Positions of shape ``(K, 3)`` in degrees that lie on or
            near chaos boundaries.
        """
        resolution = level.resolution
        theta_min, theta_max = self.theta_range

        # Reconstruct a 3-D volume.  If the level does not contain a
        # complete uniform grid, build a sparse volume and fill known
        # points.
        axis_values = np.linspace(theta_min, theta_max, resolution)
        grid_spacing = axis_values[1] - axis_values[0] if resolution > 1 else 1.0

        volume = np.full(
            (resolution, resolution, resolution), np.nan, dtype=np.float64
        )

        # Map positions back to integer indices.
        indices = np.round(
            (level.positions - theta_min) / max(grid_spacing, 1e-12)
        ).astype(int)
        indices = np.clip(indices, 0, resolution - 1)
        volume[indices[:, 0], indices[:, 1], indices[:, 2]] = level.values

        # Replace NaN with t_max-like sentinel so gradients near
        # un-simulated regions are not artificially inflated.
        finite_mask = np.isfinite(volume)
        if finite_mask.any():
            fill_value = np.nanmax(volume[finite_mask])
        else:
            fill_value = 0.0
        volume_filled = np.where(finite_mask, volume, fill_value)

        # Central-difference gradient along each axis.
        gradient_theta1 = np.gradient(volume_filled, grid_spacing, axis=0)
        gradient_theta2 = np.gradient(volume_filled, grid_spacing, axis=1)
        gradient_theta3 = np.gradient(volume_filled, grid_spacing, axis=2)

        gradient_magnitude = np.sqrt(
            gradient_theta1 ** 2 + gradient_theta2 ** 2 + gradient_theta3 ** 2
        )

        # Determine the threshold.
        finite_gradient_values = gradient_magnitude[np.isfinite(gradient_magnitude)]
        if finite_gradient_values.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        if threshold is None:
            threshold = float(np.percentile(finite_gradient_values, 60))

        boundary_mask = gradient_magnitude >= threshold

        # Convert boundary voxel indices back to angle positions.
        boundary_indices = np.argwhere(boundary_mask)  # (K, 3)
        boundary_positions = (
            boundary_indices.astype(np.float64) * grid_spacing + theta_min
        )

        return boundary_positions

    def _expand_boundary_cells(
        self,
        boundary_positions: npt.NDArray[np.float64],
        source_resolution: int,
        target_resolution: int,
    ) -> npt.NDArray[np.float64]:
        """Generate fine-grid points within each boundary cell.

        For each boundary cell identified at *source_resolution*, this
        method computes the sub-grid of *target_resolution* points that
        fall within the spatial extent of that cell.

        Parameters
        ----------
        boundary_positions : np.ndarray
            Centre positions of boundary cells, shape ``(K, 3)``.
        source_resolution : int
            Resolution of the grid the boundaries were detected on.
        target_resolution : int
            Resolution of the finer grid to generate within each cell.

        Returns
        -------
        np.ndarray
            Unique fine-grid positions of shape ``(M, 3)`` in degrees.
        """
        theta_min, theta_max = self.theta_range

        # Cell half-width at the source resolution.
        source_spacing = (theta_max - theta_min) / max(source_resolution - 1, 1)
        half_cell = source_spacing * 0.5

        # Fine grid spacing at the target resolution.
        fine_axis = np.linspace(theta_min, theta_max, target_resolution)
        fine_spacing = fine_axis[1] - fine_axis[0] if target_resolution > 1 else 1.0

        # For each boundary cell, collect fine-grid points that fall inside.
        refined_points_set: set[tuple[int, int, int]] = set()

        for boundary_centre in boundary_positions:
            cell_min = boundary_centre - half_cell
            cell_max = boundary_centre + half_cell

            # Find fine-axis indices within the cell bounds along each axis.
            for axis_index in range(3):
                axis_low = cell_min[axis_index]
                axis_high = cell_max[axis_index]

                # Indices into fine_axis that fall within [axis_low, axis_high].
                idx_low = int(
                    np.searchsorted(fine_axis, axis_low, side="left")
                )
                idx_high = int(
                    np.searchsorted(fine_axis, axis_high, side="right")
                )

                if axis_index == 0:
                    i_range = range(max(idx_low, 0), min(idx_high, target_resolution))
                elif axis_index == 1:
                    j_range = range(max(idx_low, 0), min(idx_high, target_resolution))
                else:
                    k_range = range(max(idx_low, 0), min(idx_high, target_resolution))

            for i_index in i_range:
                for j_index in j_range:
                    for k_index in k_range:
                        refined_points_set.add((i_index, j_index, k_index))

        if not refined_points_set:
            return np.empty((0, 3), dtype=np.float64)

        # Convert index tuples back to angle positions.
        index_array = np.array(sorted(refined_points_set), dtype=np.int64)
        position_array = index_array.astype(np.float64) * fine_spacing + theta_min

        return position_array


# ---------------------------------------------------------------------------
# Convenience function: progressive visualization pipeline
# ---------------------------------------------------------------------------


def create_progressive_visualization(
    output_dir: str | Path = "renders/progressive/",
    dt: float = 0.01,
    t_max: float = 15.0,
) -> Path:
    """Run the full adaptive refinement pipeline with snapshots.

    Starts at a 10^3 base resolution, refines boundaries three times
    (10 -> 20 -> 40 -> 80 equivalent resolution), saves a JSON
    snapshot after each refinement step, and exports the final
    combined data.

    Parameters
    ----------
    output_dir : str or Path
        Directory for snapshot and final output files.
    dt : float
        RK4 integration timestep in seconds.
    t_max : float
        Maximum simulation time in seconds.

    Returns
    -------
    Path
        Path to the final combined JSON export.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    adaptive_grid = AdaptiveGrid(
        base_resolution=10,
        max_resolution=80,
        theta_range=(-170, 170),
    )

    pipeline_start_time = time.monotonic()

    # Step 0: compute the base grid (10^3 = 1 000 points).
    adaptive_grid.compute_base(dt=dt, t_max=t_max)
    snapshot_path = output_path / "snapshot_step0_base10.json"
    adaptive_grid.export_to_json(snapshot_path)
    print(
        f"Step 0 complete: {adaptive_grid.total_computed} points computed, "
        f"snapshot saved to {snapshot_path}"
    )

    # Steps 1-3: refine boundaries, doubling resolution each time.
    refinement_targets = [20, 40, 80]
    for step_index, target_resolution in enumerate(refinement_targets, start=1):
        print(f"\n--- Refinement step {step_index}: target resolution {target_resolution} ---")

        newly_simulated = adaptive_grid.refine_boundaries(
            target_resolution=target_resolution,
            dt=dt,
            t_max=t_max,
        )

        snapshot_filename = (
            f"snapshot_step{step_index}_res{target_resolution}.json"
        )
        snapshot_path = output_path / snapshot_filename
        adaptive_grid.export_to_json(snapshot_path)

        print(
            f"Step {step_index} complete: {newly_simulated} new points, "
            f"{adaptive_grid.total_computed} total, "
            f"snapshot saved to {snapshot_path}"
        )

    # Final export.
    final_output_path = output_path / "adaptive_final.json"
    adaptive_grid.export_to_json(final_output_path)

    elapsed_seconds = time.monotonic() - pipeline_start_time
    print(
        f"\nProgressive visualization complete in {elapsed_seconds:.1f}s. "
        f"Total points: {adaptive_grid.total_computed} "
        f"(vs {adaptive_grid.total_cells_at_max_res} for full 80^3 grid). "
        f"Final output: {final_output_path}"
    )

    return final_output_path
