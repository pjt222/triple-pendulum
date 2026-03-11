"""Compare the 3D triple pendulum chaos map to the 2D double pendulum.

Simulates a double pendulum over a 2D grid of initial conditions
(theta1, theta2), extracts 2D slices from the 3D triple pendulum
flip-time volume, and provides quantitative structural comparisons
between the two systems. The double pendulum uses the same Lagrangian
framework as the triple pendulum but with two bobs:

    - Coupling matrix A = [[2, 1], [1, 1]]
    - Mass matrix: M_ij = A_ij * cos(theta_i - theta_j)
    - Force vector: f_i = sum_j A_ij * sin(theta_i - theta_j) * omega_j^2
                          + gravity_weight_i * g * sin(theta_i)
    - Gravity weights: [2, 1], g = 9.81
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.simulation.physics import GRAVITY


# ---------------------------------------------------------------------------
# Double pendulum physics constants
# ---------------------------------------------------------------------------

DOUBLE_COUPLING_MATRIX: NDArray[np.float64] = np.array([
    [2.0, 1.0],
    [1.0, 1.0],
])

DOUBLE_GRAVITY_WEIGHTS: NDArray[np.float64] = np.array([2.0, 1.0])


# ---------------------------------------------------------------------------
# Double pendulum equations of motion
# ---------------------------------------------------------------------------

def _double_pendulum_derivatives(
    state: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute time derivatives for N double pendulums.

    Uses the same Lagrangian approach as the triple pendulum but with
    a 2x2 coupling matrix and two bobs.

    Args:
        state: State array of shape (N, 4) where columns 0-1 are angles
            (theta1, theta2) in radians and columns 2-3 are angular
            velocities (omega1, omega2).

    Returns:
        NDArray of shape (N, 4) with time derivatives [omega1, omega2,
        alpha1, alpha2].
    """
    theta = state[:, :2]  # (N, 2)
    omega = state[:, 2:]  # (N, 2)

    coupling = DOUBLE_COUPLING_MATRIX  # (2, 2)

    # Pairwise angle differences: delta_ij = theta_i - theta_j
    angle_differences = theta[:, :, np.newaxis] - theta[:, np.newaxis, :]  # (N, 2, 2)

    # Mass matrix: M_ij = A_ij * cos(theta_i - theta_j)
    mass_matrices = coupling[np.newaxis, :, :] * np.cos(angle_differences)  # (N, 2, 2)

    # Coriolis/centripetal term: sum_j A_ij * sin(delta_ij) * omega_j^2
    omega_squared = omega ** 2  # (N, 2)
    coriolis_matrix = coupling[np.newaxis, :, :] * np.sin(angle_differences)  # (N, 2, 2)
    coriolis_term = np.einsum("nij,nj->ni", coriolis_matrix, omega_squared)  # (N, 2)

    # Gravitational term: gravity_weight_i * g * sin(theta_i)
    gravity_term = DOUBLE_GRAVITY_WEIGHTS[np.newaxis, :] * GRAVITY * np.sin(theta)  # (N, 2)

    # Force vector
    force_vectors = coriolis_term + gravity_term  # (N, 2)

    # Solve M * alpha = -f for angular accelerations
    negative_force = -force_vectors[:, :, np.newaxis]  # (N, 2, 1)
    angular_accelerations = np.linalg.solve(mass_matrices, negative_force).squeeze(-1)  # (N, 2)

    # Assemble derivative vector: [omega, alpha]
    state_derivatives = np.empty_like(state)  # (N, 4)
    state_derivatives[:, :2] = omega
    state_derivatives[:, 2:] = angular_accelerations

    return state_derivatives


def _rk4_step_double(
    state: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """Advance the double pendulum state by one RK4 timestep.

    Args:
        state: Current state array of shape (N, 4).
        dt: Timestep size in seconds.

    Returns:
        NDArray of shape (N, 4) with the updated state.
    """
    k1 = _double_pendulum_derivatives(state)
    k2 = _double_pendulum_derivatives(state + 0.5 * dt * k1)
    k3 = _double_pendulum_derivatives(state + 0.5 * dt * k2)
    k4 = _double_pendulum_derivatives(state + dt * k3)

    new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return new_state


def _detect_flips_double(
    theta_prev: NDArray[np.float64],
    theta_curr: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Detect flips (angle crossing pi) for double pendulum bobs.

    A flip occurs when the angle, wrapped to [-pi, pi], shows a
    discontinuity larger than pi between consecutive timesteps.

    Args:
        theta_prev: Angles at the previous timestep, shape (N, 2).
        theta_curr: Angles at the current timestep, shape (N, 2).

    Returns:
        Boolean array of shape (N, 2) where True indicates a flip
        occurred for that bob between the two timesteps.
    """
    wrapped_prev = (theta_prev + np.pi) % (2 * np.pi) - np.pi
    wrapped_curr = (theta_curr + np.pi) % (2 * np.pi) - np.pi
    wrapped_delta = np.abs(wrapped_curr - wrapped_prev)
    flip_detected = wrapped_delta > np.pi

    return flip_detected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_double_pendulum_grid(
    n: int = 40,
    theta_range: tuple[float, float] = (-170.0, 170.0),
    dt: float = 0.01,
    t_max: float = 15.0,
) -> dict:
    """Simulate a 2D grid of double pendulum initial conditions.

    Builds an n x n grid of (theta1, theta2) pairs spanning
    theta_range, integrates each double pendulum from rest (omega=0)
    using RK4, and records the time of first flip (either bob's angle
    crossing +/-180 degrees).

    Args:
        n: Number of grid points per axis. Total simulations = n^2.
        theta_range: (theta_min, theta_max) in degrees. Default
            (-170, 170) avoids the singularity at +/-180.
        dt: RK4 integration timestep in seconds.
        t_max: Maximum simulation time in seconds.

    Returns:
        Dictionary with:
            - "flip_times": NDArray of shape (n, n) with first-flip
              time for each (theta1, theta2) pair. NaN if never flipped.
            - "grid_size": int, number of points per axis.
            - "theta_range": tuple of (theta_min, theta_max).
    """
    theta_min, theta_max = theta_range
    axis_values = np.linspace(theta_min, theta_max, n)
    theta1_grid, theta2_grid = np.meshgrid(axis_values, axis_values, indexing="ij")

    # Flatten to (n^2, 2) array of initial conditions in degrees
    initial_conditions = np.column_stack([
        theta1_grid.ravel(),
        theta2_grid.ravel(),
    ])

    num_pendulums = initial_conditions.shape[0]
    num_steps = int(np.ceil(t_max / dt))
    progress_interval = max(1, num_steps // 10)

    # Initialize state: [theta1, theta2, omega1, omega2] with omega=0
    state = np.zeros((num_pendulums, 4), dtype=np.float64)
    state[:, :2] = np.radians(initial_conditions)

    # Flip tracking
    first_flip_times = np.full(num_pendulums, np.nan, dtype=np.float64)
    has_flipped = np.zeros(num_pendulums, dtype=bool)

    print(
        f"Double pendulum: simulating {num_pendulums} conditions "
        f"({n}x{n} grid) for {t_max}s (dt={dt})"
    )

    for step_index in range(num_steps):
        theta_prev = state[:, :2].copy()

        state = _rk4_step_double(state, dt)
        current_time = (step_index + 1) * dt

        theta_curr = state[:, :2]

        # Detect flips for either bob
        per_bob_flips = _detect_flips_double(theta_prev, theta_curr)  # (N, 2)
        any_bob_flipped = np.any(per_bob_flips, axis=1)  # (N,)
        newly_flipped = any_bob_flipped & ~has_flipped

        first_flip_times[newly_flipped] = current_time
        has_flipped[newly_flipped] = True

        # Progress reporting
        if (step_index + 1) % progress_interval == 0:
            percent_complete = 100.0 * (step_index + 1) / num_steps
            flipped_fraction = np.mean(has_flipped)
            print(
                f"  Progress: {percent_complete:5.1f}% "
                f"(t={current_time:.2f}s, "
                f"flipped={flipped_fraction:.1%})"
            )

        # Early stopping: all pendulums have flipped
        if np.all(has_flipped):
            print(
                f"  Early stop at t={current_time:.2f}s: "
                f"all {num_pendulums} pendulums have flipped."
            )
            break

    num_flipped = int(np.sum(has_flipped))
    print(
        f"Double pendulum complete: {num_flipped}/{num_pendulums} flipped"
    )

    flip_times_2d = first_flip_times.reshape(n, n)

    double_pendulum_results = {
        "flip_times": flip_times_2d,
        "grid_size": n,
        "theta_range": theta_range,
    }

    return double_pendulum_results


def extract_theta3_slice(
    triple_flip_times_3d: NDArray[np.float64],
    slice_index: int,
) -> NDArray[np.float64]:
    """Extract a 2D slice at a fixed theta3 index from the 3D volume.

    The 3D flip-time array has shape (n, n, n) indexed as
    (theta1, theta2, theta3). This function fixes the third axis and
    returns the (theta1, theta2) plane at the given index.

    Args:
        triple_flip_times_3d: 3D array of shape (n, n, n) with
            flip-time values from the triple pendulum simulation.
        slice_index: Index along the theta3 axis to extract.
            Must be in [0, n-1].

    Returns:
        NDArray of shape (n, n) with the flip-time values at the
        specified theta3 slice.

    Raises:
        IndexError: If slice_index is out of bounds.
    """
    grid_size = triple_flip_times_3d.shape[2]
    if not 0 <= slice_index < grid_size:
        raise IndexError(
            f"slice_index {slice_index} is out of bounds for theta3 "
            f"axis with size {grid_size}"
        )

    theta3_slice = triple_flip_times_3d[:, :, slice_index].copy()

    return theta3_slice


def compare_structures(
    double_flip_times_2d: NDArray[np.float64],
    triple_slice_2d: NDArray[np.float64],
    t_max: float = 15.0,
) -> dict:
    """Quantitatively compare a 2D double pendulum map to a triple slice.

    Computes multiple similarity metrics between the two 2D chaos maps:
    structural similarity (SSIM if scipy is available, otherwise
    Pearson correlation), boundary overlap via Jaccard index, and
    flip percentage comparison.

    Both input arrays must have the same shape. NaN values (no flip)
    are replaced with t_max for comparison purposes.

    Args:
        double_flip_times_2d: 2D array of shape (n, n) with double
            pendulum flip times.
        triple_slice_2d: 2D array of shape (n, n) with triple pendulum
            flip times from a theta3 slice.
        t_max: Maximum simulation time, used as a fill value for NaN
            entries and for normalisation.

    Returns:
        Dictionary with comparison metrics:
            - "correlation": Pearson correlation between the two maps.
            - "ssim": Structural similarity index (None if scipy
              unavailable).
            - "boundary_jaccard": Jaccard index of boundary masks.
            - "double_flip_fraction": Fraction of grid points that
              flipped in the double pendulum map.
            - "triple_flip_fraction": Fraction of grid points that
              flipped in the triple pendulum slice.
            - "flip_fraction_difference": Absolute difference between
              the two flip fractions.
            - "mean_flip_time_double": Mean flip time (excluding NaN)
              for the double pendulum.
            - "mean_flip_time_triple": Mean flip time (excluding NaN)
              for the triple pendulum slice.
    """
    if double_flip_times_2d.shape != triple_slice_2d.shape:
        raise ValueError(
            f"Shape mismatch: double pendulum map has shape "
            f"{double_flip_times_2d.shape} but triple slice has shape "
            f"{triple_slice_2d.shape}"
        )

    # Replace NaN with t_max for numerical comparisons
    double_filled = np.where(
        np.isfinite(double_flip_times_2d), double_flip_times_2d, t_max,
    )
    triple_filled = np.where(
        np.isfinite(triple_slice_2d), triple_slice_2d, t_max,
    )

    # --- Pearson correlation ---
    double_flat = double_filled.ravel()
    triple_flat = triple_filled.ravel()

    double_mean = np.mean(double_flat)
    triple_mean = np.mean(triple_flat)
    double_centered = double_flat - double_mean
    triple_centered = triple_flat - triple_mean

    numerator = np.sum(double_centered * triple_centered)
    denominator = np.sqrt(
        np.sum(double_centered ** 2) * np.sum(triple_centered ** 2)
    )
    if denominator > 0:
        correlation = float(numerator / denominator)
    else:
        correlation = 0.0

    # --- SSIM (if scipy available) ---
    ssim_value = None
    try:
        from scipy.ndimage import uniform_filter

        # Normalize both maps to [0, 1] for SSIM computation
        double_normalized = double_filled / t_max
        triple_normalized = triple_filled / t_max

        # SSIM with default constants
        data_range = 1.0
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        window_size = 7

        mean_double = uniform_filter(double_normalized, size=window_size)
        mean_triple = uniform_filter(triple_normalized, size=window_size)

        mean_double_sq = uniform_filter(
            double_normalized ** 2, size=window_size,
        )
        mean_triple_sq = uniform_filter(
            triple_normalized ** 2, size=window_size,
        )
        mean_double_triple = uniform_filter(
            double_normalized * triple_normalized, size=window_size,
        )

        variance_double = mean_double_sq - mean_double ** 2
        variance_triple = mean_triple_sq - mean_triple ** 2
        covariance = mean_double_triple - mean_double * mean_triple

        ssim_map = (
            (2 * mean_double * mean_triple + c1)
            * (2 * covariance + c2)
        ) / (
            (mean_double ** 2 + mean_triple ** 2 + c1)
            * (variance_double + variance_triple + c2)
        )

        ssim_value = float(np.mean(ssim_map))

    except ImportError:
        # scipy not available; SSIM remains None
        pass

    # --- Boundary detection and Jaccard index ---
    # Boundaries are regions of steep gradient in flip time (chaos edges).
    # Normalize to [0, 1] before computing gradients so thresholds are
    # consistent between the two maps.
    double_normalized_for_boundary = double_filled / t_max
    triple_normalized_for_boundary = triple_filled / t_max

    gradient_double_y, gradient_double_x = np.gradient(
        double_normalized_for_boundary,
    )
    gradient_triple_y, gradient_triple_x = np.gradient(
        triple_normalized_for_boundary,
    )

    gradient_magnitude_double = np.sqrt(
        gradient_double_x ** 2 + gradient_double_y ** 2,
    )
    gradient_magnitude_triple = np.sqrt(
        gradient_triple_x ** 2 + gradient_triple_y ** 2,
    )

    # Threshold at the 75th percentile of gradient magnitude
    threshold_double = np.percentile(gradient_magnitude_double, 75)
    threshold_triple = np.percentile(gradient_magnitude_triple, 75)

    boundary_mask_double = gradient_magnitude_double > threshold_double
    boundary_mask_triple = gradient_magnitude_triple > threshold_triple

    intersection = np.sum(boundary_mask_double & boundary_mask_triple)
    union = np.sum(boundary_mask_double | boundary_mask_triple)

    if union > 0:
        boundary_jaccard = float(intersection / union)
    else:
        boundary_jaccard = 0.0

    # --- Flip fraction comparison ---
    double_flip_fraction = float(
        np.mean(np.isfinite(double_flip_times_2d)),
    )
    triple_flip_fraction = float(
        np.mean(np.isfinite(triple_slice_2d)),
    )
    flip_fraction_difference = abs(
        double_flip_fraction - triple_flip_fraction,
    )

    # --- Mean flip times (finite values only) ---
    double_finite = double_flip_times_2d[np.isfinite(double_flip_times_2d)]
    triple_finite = triple_slice_2d[np.isfinite(triple_slice_2d)]

    mean_flip_time_double = (
        float(np.mean(double_finite)) if double_finite.size > 0 else np.nan
    )
    mean_flip_time_triple = (
        float(np.mean(triple_finite)) if triple_finite.size > 0 else np.nan
    )

    comparison_metrics = {
        "correlation": correlation,
        "ssim": ssim_value,
        "boundary_jaccard": boundary_jaccard,
        "double_flip_fraction": double_flip_fraction,
        "triple_flip_fraction": triple_flip_fraction,
        "flip_fraction_difference": flip_fraction_difference,
        "mean_flip_time_double": mean_flip_time_double,
        "mean_flip_time_triple": mean_flip_time_triple,
    }

    return comparison_metrics


def create_comparison_figure(
    double_flip_times_2d: NDArray[np.float64],
    triple_flip_times_3d: NDArray[np.float64],
    theta_range: tuple[float, float] = (-170.0, 170.0),
    t_max: float = 15.0,
    output_path: str = "renders/comparison.png",
    dpi: int = 200,
) -> str:
    """Create a comparison figure of double vs triple pendulum chaos maps.

    Produces a 2x2 figure:
        - Row 1, left:  Double pendulum flip-time map
        - Row 1, right: Triple pendulum slice at theta3=0 (middle index)
        - Row 2, left:  Difference map (triple slice minus double)
        - Row 2, right: Boundary overlay (double boundaries in cyan,
          triple boundaries in magenta, overlap in white)

    Uses the chaos_magma colormap from src.visualization.colormap and
    the matplotlib Agg backend for headless rendering.

    Args:
        double_flip_times_2d: 2D array of shape (n, n) with double
            pendulum flip times.
        triple_flip_times_3d: 3D array of shape (n, n, n) with triple
            pendulum flip times.
        theta_range: (theta_min, theta_max) in degrees for axis labels.
        t_max: Maximum simulation time for colormap normalisation.
        output_path: File path for the saved figure.
        dpi: Resolution in dots per inch.

    Returns:
        The output_path string.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from pathlib import Path

    from src.visualization.colormap import get_matplotlib_cmap

    # Get the chaos_magma colormap
    chaos_colormap = get_matplotlib_cmap("chaos_magma")

    # Extract theta3=0 slice (middle index of the third axis)
    grid_size_triple = triple_flip_times_3d.shape[2]
    middle_index = grid_size_triple // 2
    triple_slice = extract_theta3_slice(triple_flip_times_3d, middle_index)

    # Compute the theta3 value at the middle index for labelling
    theta_min, theta_max_val = theta_range
    axis_values = np.linspace(theta_min, theta_max_val, grid_size_triple)
    theta3_value = axis_values[middle_index]

    # Ensure the double map and triple slice have compatible shapes
    # for the comparison. If they differ, resize the smaller to match.
    double_map = double_flip_times_2d
    if double_map.shape != triple_slice.shape:
        # Resize using nearest-neighbor interpolation to avoid
        # introducing artificial smoothness in the fractal structures
        target_size = max(double_map.shape[0], triple_slice.shape[0])

        if double_map.shape[0] != target_size:
            double_map = _resize_nearest(double_map, target_size)
        if triple_slice.shape[0] != target_size:
            triple_slice = _resize_nearest(triple_slice, target_size)

    # Replace NaN with t_max for display and difference computation
    double_filled = np.where(np.isfinite(double_map), double_map, t_max)
    triple_filled = np.where(np.isfinite(triple_slice), triple_slice, t_max)

    # Difference map: triple slice minus double
    difference_map = triple_filled - double_filled

    # Boundary masks for overlay
    double_normalized = double_filled / t_max
    triple_normalized = triple_filled / t_max

    gradient_double_y, gradient_double_x = np.gradient(double_normalized)
    gradient_triple_y, gradient_triple_x = np.gradient(triple_normalized)

    gradient_magnitude_double = np.sqrt(
        gradient_double_x ** 2 + gradient_double_y ** 2,
    )
    gradient_magnitude_triple = np.sqrt(
        gradient_triple_x ** 2 + gradient_triple_y ** 2,
    )

    threshold_double = np.percentile(gradient_magnitude_double, 75)
    threshold_triple = np.percentile(gradient_magnitude_triple, 75)

    boundary_double = gradient_magnitude_double > threshold_double
    boundary_triple = gradient_magnitude_triple > threshold_triple

    # Build the boundary overlay as an RGB image
    # Cyan = double only, Magenta = triple only, White = overlap
    overlay_height, overlay_width = double_filled.shape
    boundary_overlay = np.zeros(
        (overlay_height, overlay_width, 3), dtype=np.float64,
    )

    # Background: normalized double map in grayscale
    background_gray = double_normalized
    boundary_overlay[:, :, 0] = background_gray * 0.3
    boundary_overlay[:, :, 1] = background_gray * 0.3
    boundary_overlay[:, :, 2] = background_gray * 0.3

    # Cyan for double-only boundaries (R=0, G=1, B=1)
    double_only = boundary_double & ~boundary_triple
    boundary_overlay[double_only, 0] = 0.0
    boundary_overlay[double_only, 1] = 1.0
    boundary_overlay[double_only, 2] = 1.0

    # Magenta for triple-only boundaries (R=1, G=0, B=1)
    triple_only = boundary_triple & ~boundary_double
    boundary_overlay[triple_only, 0] = 1.0
    boundary_overlay[triple_only, 1] = 0.0
    boundary_overlay[triple_only, 2] = 1.0

    # White for overlapping boundaries
    overlap = boundary_double & boundary_triple
    boundary_overlay[overlap, 0] = 1.0
    boundary_overlay[overlap, 1] = 1.0
    boundary_overlay[overlap, 2] = 1.0

    # --- Create the figure ---
    figure, axes = plt.subplots(
        2, 2, figsize=(12, 10), constrained_layout=True,
    )

    extent = [theta_range[0], theta_range[1], theta_range[0], theta_range[1]]
    color_normalizer = Normalize(vmin=0, vmax=t_max)

    # Row 1, left: Double pendulum map
    image_double = axes[0, 0].imshow(
        double_filled.T,
        origin="lower",
        extent=extent,
        cmap=chaos_colormap,
        norm=color_normalizer,
        aspect="equal",
    )
    axes[0, 0].set_title("Double Pendulum")
    axes[0, 0].set_xlabel(r"$\theta_1$ (deg)")
    axes[0, 0].set_ylabel(r"$\theta_2$ (deg)")
    figure.colorbar(image_double, ax=axes[0, 0], label="Flip time (s)")

    # Row 1, right: Triple pendulum slice at theta3=0
    image_triple = axes[0, 1].imshow(
        triple_filled.T,
        origin="lower",
        extent=extent,
        cmap=chaos_colormap,
        norm=color_normalizer,
        aspect="equal",
    )
    axes[0, 1].set_title(
        f"Triple Pendulum Slice "
        f"($\\theta_3$ = {theta3_value:.1f}$^\\circ$)"
    )
    axes[0, 1].set_xlabel(r"$\theta_1$ (deg)")
    axes[0, 1].set_ylabel(r"$\theta_2$ (deg)")
    figure.colorbar(image_triple, ax=axes[0, 1], label="Flip time (s)")

    # Row 2, left: Difference map
    max_abs_difference = np.max(np.abs(difference_map))
    if max_abs_difference == 0:
        max_abs_difference = 1.0  # Avoid zero-range colorbar

    image_difference = axes[1, 0].imshow(
        difference_map.T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-max_abs_difference,
        vmax=max_abs_difference,
        aspect="equal",
    )
    axes[1, 0].set_title("Difference (Triple - Double)")
    axes[1, 0].set_xlabel(r"$\theta_1$ (deg)")
    axes[1, 0].set_ylabel(r"$\theta_2$ (deg)")
    figure.colorbar(
        image_difference, ax=axes[1, 0], label="$\\Delta$ Flip time (s)",
    )

    # Row 2, right: Boundary overlay
    axes[1, 1].imshow(
        np.transpose(boundary_overlay, (1, 0, 2)),
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    axes[1, 1].set_title("Boundary Overlay")
    axes[1, 1].set_xlabel(r"$\theta_1$ (deg)")
    axes[1, 1].set_ylabel(r"$\theta_2$ (deg)")

    # Add legend patches for the boundary overlay
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="cyan", label="Double only"),
        Patch(facecolor="magenta", label="Triple only"),
        Patch(facecolor="white", edgecolor="gray", label="Overlap"),
    ]
    axes[1, 1].legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=8,
        framealpha=0.8,
    )

    # Compute and display metrics in the figure title
    comparison_metrics = compare_structures(
        double_map, triple_slice, t_max=t_max,
    )
    correlation_value = comparison_metrics["correlation"]
    jaccard_value = comparison_metrics["boundary_jaccard"]
    ssim_display = comparison_metrics["ssim"]

    title_parts = [
        f"Correlation: {correlation_value:.3f}",
        f"Boundary Jaccard: {jaccard_value:.3f}",
    ]
    if ssim_display is not None:
        title_parts.append(f"SSIM: {ssim_display:.3f}")

    figure.suptitle(
        "Double vs Triple Pendulum Chaos Map Comparison\n"
        + "  |  ".join(title_parts),
        fontsize=12,
    )

    # Save the figure
    output_directory = Path(output_path).parent
    output_directory.mkdir(parents=True, exist_ok=True)

    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"Comparison figure saved to {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resize_nearest(
    array_2d: NDArray[np.float64],
    target_size: int,
) -> NDArray[np.float64]:
    """Resize a 2D array to (target_size, target_size) using nearest neighbor.

    Uses index-based nearest-neighbor sampling to avoid introducing
    interpolation artifacts in the fractal chaos structures.

    Args:
        array_2d: Input 2D array of shape (m, m).
        target_size: Desired output size per axis.

    Returns:
        NDArray of shape (target_size, target_size).
    """
    source_size = array_2d.shape[0]
    indices = np.round(
        np.linspace(0, source_size - 1, target_size)
    ).astype(int)
    resized_array = array_2d[np.ix_(indices, indices)]

    return resized_array
