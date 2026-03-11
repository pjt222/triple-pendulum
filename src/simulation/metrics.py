"""Chaos metrics for triple pendulum simulations.

Provides flip detection and flip-time tracking. A "flip" occurs when any
pendulum bob's angle wraps past +/-180 degrees (pi radians), indicating
the bob has gone over the top. The time-to-first-flip is the primary
chaos metric used to color voxels in the 3D visualization.
"""

import numpy as np
from numpy.typing import NDArray


def detect_flips(
    theta_prev: NDArray[np.float64],
    theta_curr: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Detect when a pendulum bob goes over the top (+/-180 degrees).

    Angles accumulate freely (are not wrapped) during integration.
    A flip occurs when the angle crosses a multiple of pi — i.e., when
    the wrapped-to-[-pi,pi] versions of theta_prev and theta_curr have
    opposite signs AND the raw angle moved enough to actually cross.

    In practice, the simplest correct check: wrap both angles to [-pi, pi]
    and detect a sign change with large magnitude (the wrap discontinuity).

    Args:
        theta_prev: Angles at the previous timestep, shape (N, 3).
        theta_curr: Angles at the current timestep, shape (N, 3).

    Returns:
        Boolean array of shape (N, 3) where True indicates a flip
        occurred for that pendulum bob between the two timesteps.
    """
    wrapped_prev = (theta_prev + np.pi) % (2 * np.pi) - np.pi
    wrapped_curr = (theta_curr + np.pi) % (2 * np.pi) - np.pi
    wrapped_delta = np.abs(wrapped_curr - wrapped_prev)
    flip_detected = wrapped_delta > np.pi

    return flip_detected


class FlipTimeTracker:
    """Tracks the time of first flip for each pendulum in a batch.

    A pendulum is considered "flipped" when ANY of its three bobs
    experiences an angle wrap past +/-180 degrees. Only the first flip
    event is recorded; subsequent flips are ignored.

    Attributes:
        num_pendulums: Number of pendulums being tracked.
        first_flip_times: Array of shape (N,) holding the time of first
            flip for each pendulum. Initialized to NaN (no flip yet).
        has_flipped: Boolean array of shape (N,) indicating whether
            each pendulum has already recorded a flip.
    """

    def __init__(self, num_pendulums: int) -> None:
        """Initialize the tracker for a batch of pendulums.

        Args:
            num_pendulums: Number of pendulums to track.
        """
        self.num_pendulums = num_pendulums
        self.first_flip_times: NDArray[np.float64] = np.full(num_pendulums, np.nan)
        self.has_flipped: NDArray[np.bool_] = np.zeros(num_pendulums, dtype=bool)

    def update(
        self,
        theta_prev: NDArray[np.float64],
        theta_curr: NDArray[np.float64],
        current_time: float,
    ) -> None:
        """Check for new flips and record their times.

        For each pendulum that has not yet flipped, checks whether any
        of its three bobs flipped between theta_prev and theta_curr.
        If so, records current_time as the flip time.

        Args:
            theta_prev: Angles at the previous timestep, shape (N, 3).
            theta_curr: Angles at the current timestep, shape (N, 3).
            current_time: The simulation time of the current timestep.
        """
        # Detect per-bob flips: (N, 3) boolean
        per_bob_flips = detect_flips(theta_prev, theta_curr)

        # A pendulum flips if ANY of its 3 bobs flipped: (N,) boolean
        any_bob_flipped = np.any(per_bob_flips, axis=1)

        # Only record for pendulums that haven't flipped before
        newly_flipped = any_bob_flipped & ~self.has_flipped

        self.first_flip_times[newly_flipped] = current_time
        self.has_flipped[newly_flipped] = True

    def get_flip_times(self) -> NDArray[np.float64]:
        """Return the first-flip time for each pendulum.

        Returns:
            NDArray of shape (N,) with the time of first flip.
            Pendulums that never flipped have NaN.
        """
        return self.first_flip_times.copy()

    @property
    def all_flipped(self) -> bool:
        """Check whether every pendulum in the batch has flipped."""
        return bool(np.all(self.has_flipped))

    @property
    def fraction_flipped(self) -> float:
        """Return the fraction of pendulums that have flipped so far."""
        return float(np.mean(self.has_flipped))


# ---------------------------------------------------------------------------
# Lyapunov exponent and trajectory divergence metrics
# ---------------------------------------------------------------------------


def compute_lyapunov_exponents(
    initial_thetas: NDArray[np.float64],
    dt: float = 0.01,
    t_max: float = 15.0,
    epsilon: float = 1e-6,
    renormalization_interval: int = 100,
) -> NDArray[np.float64]:
    """Estimate the maximum Lyapunov exponent for each initial condition.

    For every initial condition in the batch, two nearby trajectories are
    integrated in parallel: a base trajectory and a perturbed copy whose
    initial theta values are shifted by ``epsilon``.  Every
    ``renormalization_interval`` RK4 steps the phase-space distance between
    the two trajectories is measured, the logarithmic growth rate is
    accumulated, and the perturbed trajectory is renormalized back to
    ``epsilon`` distance from the base.  The maximum Lyapunov exponent is
    the time-averaged sum of these logarithmic growth rates.

    A positive exponent indicates chaotic sensitivity to initial conditions;
    a near-zero or negative exponent indicates regular (non-chaotic) motion.

    Args:
        initial_thetas: Initial angles of shape (N, 3) in **degrees**.
            All pendulums start from rest (omega = 0).
        dt: RK4 integration timestep in seconds.
        t_max: Total simulation time in seconds.
        epsilon: Initial separation between base and perturbed trajectories
            in radians (applied equally to all three theta components).
        renormalization_interval: Number of RK4 steps between each
            renormalization event.

    Returns:
        NDArray of shape (N,) containing the estimated maximum Lyapunov
        exponent (in units of 1/s) for each initial condition.
    """
    from src.simulation.batch_sim import rk4_step
    from src.simulation.physics import derivatives

    num_pendulums = initial_thetas.shape[0]
    num_steps = int(np.ceil(t_max / dt))

    # --- Set up base state: angles in radians, velocities zero ----
    base_state = np.zeros((num_pendulums, 6), dtype=np.float64)
    base_state[:, :3] = np.radians(initial_thetas)

    # --- Set up perturbed state: shift each theta by +epsilon -----
    perturbed_state = base_state.copy()
    perturbed_state[:, :3] += epsilon

    # Accumulators for the running sum of log(stretch) values
    accumulated_log_stretch = np.zeros(num_pendulums, dtype=np.float64)
    renormalization_count = 0

    for step_index in range(1, num_steps + 1):
        base_state = rk4_step(base_state, dt, derivatives)
        perturbed_state = rk4_step(perturbed_state, dt, derivatives)

        # --- Renormalize at fixed intervals ----------------------
        if step_index % renormalization_interval == 0:
            separation_vector = perturbed_state - base_state  # (N, 6)
            separation_distance = np.linalg.norm(
                separation_vector, axis=1
            )  # (N,)

            # Guard against zero distance (perfectly identical trajectories)
            separation_distance = np.maximum(separation_distance, 1e-30)

            accumulated_log_stretch += np.log(separation_distance / epsilon)
            renormalization_count += 1

            # Rescale perturbed trajectory back to epsilon distance
            scale_factor = epsilon / separation_distance  # (N,)
            perturbed_state = (
                base_state + separation_vector * scale_factor[:, np.newaxis]
            )

    # If simulation was too short for even one renormalization, do a
    # final measurement so the result is not all zeros.
    if renormalization_count == 0:
        separation_vector = perturbed_state - base_state
        separation_distance = np.linalg.norm(separation_vector, axis=1)
        separation_distance = np.maximum(separation_distance, 1e-30)
        accumulated_log_stretch = np.log(separation_distance / epsilon)
        renormalization_count = 1

    elapsed_time = num_steps * dt
    lyapunov_exponents = accumulated_log_stretch / elapsed_time

    return lyapunov_exponents


def compute_trajectory_divergence(
    initial_thetas: NDArray[np.float64],
    dt: float = 0.01,
    t_max: float = 5.0,
    epsilon: float = 1e-6,
) -> NDArray[np.float64]:
    """Measure trajectory divergence as log10(final_distance / epsilon).

    A simpler alternative to the full Lyapunov exponent: for each initial
    condition, integrate a base trajectory and a perturbed copy (shifted by
    ``epsilon`` on all three theta components) for ``t_max`` seconds, then
    report the logarithmic ratio of the final phase-space separation to the
    initial perturbation.

    Large positive values indicate strong sensitivity (chaotic); values near
    zero indicate the perturbation neither grew nor shrank (regular motion).

    Args:
        initial_thetas: Initial angles of shape (N, 3) in **degrees**.
            All pendulums start from rest (omega = 0).
        dt: RK4 integration timestep in seconds.
        t_max: Total simulation time in seconds.
        epsilon: Initial perturbation magnitude in radians, applied equally
            to all three theta components.

    Returns:
        NDArray of shape (N,) with log10(final_distance / epsilon) for each
        initial condition.
    """
    from src.simulation.batch_sim import rk4_step
    from src.simulation.physics import derivatives

    num_pendulums = initial_thetas.shape[0]
    num_steps = int(np.ceil(t_max / dt))

    # --- Set up base state: angles in radians, velocities zero ----
    base_state = np.zeros((num_pendulums, 6), dtype=np.float64)
    base_state[:, :3] = np.radians(initial_thetas)

    # --- Set up perturbed state: shift each theta by +epsilon -----
    perturbed_state = base_state.copy()
    perturbed_state[:, :3] += epsilon

    # Integrate both trajectories forward without renormalization
    for step_index in range(num_steps):
        base_state = rk4_step(base_state, dt, derivatives)
        perturbed_state = rk4_step(perturbed_state, dt, derivatives)

    # Compute final phase-space separation
    final_separation_vector = perturbed_state - base_state  # (N, 6)
    final_distance = np.linalg.norm(final_separation_vector, axis=1)  # (N,)

    # Guard against zero distance to avoid log(0)
    final_distance = np.maximum(final_distance, 1e-30)

    divergence_ratio = np.log10(final_distance / epsilon)

    return divergence_ratio
