"""Energy analysis for triple pendulum simulations.

Computes kinetic energy, potential energy, and total energy for a triple
pendulum system with equal point masses (m=1) and equal rod lengths (l=1).

Bob positions are cumulative from the pivot at the origin:
    x_i = sum_{j=1}^{i} l * sin(theta_j)
    y_i = -sum_{j=1}^{i} l * cos(theta_j)

Bob velocities (time derivatives of position):
    dx_i/dt = sum_{j=1}^{i} l * cos(theta_j) * omega_j
    dy_i/dt = sum_{j=1}^{i} l * sin(theta_j) * omega_j

Energy components:
    KE_i = 0.5 * m * (dx_i/dt^2 + dy_i/dt^2)
    PE_i = -m * g * y_i
    Total = sum_i (KE_i + PE_i)
"""

import numpy as np
from numpy.typing import NDArray

from src.simulation.physics import GRAVITY, NUM_BOBS

# Physical constants for the equal-mass, equal-length system
MASS: float = 1.0
LENGTH: float = 1.0


def compute_bob_positions(
    state: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Cartesian positions of all three bobs.

    Each bob's position is the cumulative sum of rod endpoints from
    the fixed pivot at the origin. For bob i (1-indexed):
        x_i = sum_{j=1}^{i} l * sin(theta_j)
        y_i = -sum_{j=1}^{i} l * cos(theta_j)

    Args:
        state: State array of shape (N, 6) where columns 0-2 are angles
            (theta1, theta2, theta3) in radians and columns 3-5 are
            angular velocities (omega1, omega2, omega3).

    Returns:
        NDArray of shape (N, 3, 2) where positions[n, i, 0] is the x
        coordinate and positions[n, i, 1] is the y coordinate of bob i
        for pendulum n.
    """
    num_pendulums = state.shape[0]
    theta = state[:, :NUM_BOBS]  # (N, 3)

    # Compute individual rod contributions
    rod_dx = LENGTH * np.sin(theta)   # (N, 3)
    rod_dy = -LENGTH * np.cos(theta)  # (N, 3)

    # Cumulative sum along the chain gives bob positions
    bob_x = np.cumsum(rod_dx, axis=1)  # (N, 3)
    bob_y = np.cumsum(rod_dy, axis=1)  # (N, 3)

    # Stack into (N, 3, 2) with [x, y] as the last axis
    positions = np.stack([bob_x, bob_y], axis=-1)  # (N, 3, 2)

    return positions


def compute_kinetic_energy(
    state: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute kinetic energy of each bob for N pendulums.

    The velocity of bob i depends on the angular velocities of all
    joints from 1 to i (since each rod contributes to the motion):
        dx_i/dt = sum_{j=1}^{i} l * cos(theta_j) * omega_j
        dy_i/dt = sum_{j=1}^{i} l * sin(theta_j) * omega_j
        KE_i = 0.5 * m * (dx_i/dt^2 + dy_i/dt^2)

    Args:
        state: State array of shape (N, 6) where columns 0-2 are angles
            in radians and columns 3-5 are angular velocities.

    Returns:
        NDArray of shape (N, 3) with the kinetic energy of each bob.
    """
    num_pendulums = state.shape[0]
    theta = state[:, :NUM_BOBS]  # (N, 3)
    omega = state[:, NUM_BOBS:]  # (N, 3)

    # Individual rod velocity contributions:
    # vx_contribution_j = l * cos(theta_j) * omega_j
    # vy_contribution_j = l * sin(theta_j) * omega_j
    velocity_x_contributions = LENGTH * np.cos(theta) * omega  # (N, 3)
    velocity_y_contributions = LENGTH * np.sin(theta) * omega  # (N, 3)

    # Bob i's velocity is the cumulative sum of contributions from joints 1..i
    velocity_x = np.cumsum(velocity_x_contributions, axis=1)  # (N, 3)
    velocity_y = np.cumsum(velocity_y_contributions, axis=1)  # (N, 3)

    # Speed squared for each bob
    speed_squared = velocity_x ** 2 + velocity_y ** 2  # (N, 3)

    # KE_i = 0.5 * m * v_i^2
    kinetic_energy = 0.5 * MASS * speed_squared  # (N, 3)

    return kinetic_energy


def compute_potential_energy(
    state: NDArray[np.float64],
    gravity: float = GRAVITY,
) -> NDArray[np.float64]:
    """Compute gravitational potential energy of each bob for N pendulums.

    PE_i = -m * g * y_i, where y_i is the vertical position of bob i
    (negative means below the pivot).

    Args:
        state: State array of shape (N, 6) where columns 0-2 are angles
            in radians and columns 3-5 are angular velocities.
        gravity: Gravitational acceleration in m/s^2. Defaults to 9.81.

    Returns:
        NDArray of shape (N, 3) with the potential energy of each bob.
    """
    positions = compute_bob_positions(state)  # (N, 3, 2)
    bob_y = positions[:, :, 1]  # (N, 3) -- y coordinates

    # PE_i = -m * g * y_i
    # Since y_i is negative when hanging down, PE is negative (lower energy)
    # and positive when above the pivot
    potential_energy = -MASS * gravity * bob_y  # (N, 3)

    return potential_energy


def compute_total_energy(
    state: NDArray[np.float64],
    gravity: float = GRAVITY,
) -> NDArray[np.float64]:
    """Compute total mechanical energy for N pendulums.

    Total energy is the sum of kinetic and potential energy across all
    three bobs. For a conservative system, this should be constant
    (modulo numerical integration error).

    Args:
        state: State array of shape (N, 6) where columns 0-2 are angles
            in radians and columns 3-5 are angular velocities.
        gravity: Gravitational acceleration in m/s^2. Defaults to 9.81.

    Returns:
        NDArray of shape (N,) with the total energy of each pendulum.
    """
    kinetic_energy = compute_kinetic_energy(state)      # (N, 3)
    potential_energy = compute_potential_energy(state, gravity)  # (N, 3)

    total_energy = np.sum(kinetic_energy + potential_energy, axis=1)  # (N,)

    return total_energy


def track_energy_evolution(
    initial_thetas: NDArray[np.float64],
    dt: float = 0.01,
    t_max: float = 5.0,
    sample_interval: int = 10,
) -> dict:
    """Simulate pendulums and record energy components at regular intervals.

    Integrates N triple pendulums from rest using RK4 and samples the
    kinetic energy, potential energy, and total energy at every
    ``sample_interval`` timesteps.

    Args:
        initial_thetas: Initial angles of shape (N, 3) in **degrees**.
            All pendulums start from rest (omega = 0).
        dt: RK4 integration timestep in seconds.
        t_max: Maximum simulation time in seconds.
        sample_interval: Number of RK4 steps between energy samples.
            For example, with dt=0.01 and sample_interval=10, energy
            is recorded every 0.1 seconds.

    Returns:
        Dictionary with:
            - "times": NDArray of shape (T,) with sample times in seconds.
            - "kinetic": NDArray of shape (T, N, 3) with KE per bob.
            - "potential": NDArray of shape (T, N, 3) with PE per bob.
            - "total": NDArray of shape (T, N) with total energy.
    """
    from src.simulation.batch_sim import rk4_step
    from src.simulation.physics import derivatives

    num_pendulums = initial_thetas.shape[0]
    num_steps = int(np.ceil(t_max / dt))

    # Determine how many samples we will collect (including t=0)
    num_samples = 1 + num_steps // sample_interval
    if num_steps % sample_interval != 0:
        num_samples += 1  # Include the final step if it doesn't align

    # Pre-allocate output arrays
    times = np.empty(num_samples, dtype=np.float64)
    kinetic_energy_history = np.empty(
        (num_samples, num_pendulums, NUM_BOBS), dtype=np.float64,
    )
    potential_energy_history = np.empty(
        (num_samples, num_pendulums, NUM_BOBS), dtype=np.float64,
    )
    total_energy_history = np.empty(
        (num_samples, num_pendulums), dtype=np.float64,
    )

    # Initialize state: angles in radians, velocities zero
    state = np.zeros((num_pendulums, 6), dtype=np.float64)
    state[:, :NUM_BOBS] = np.radians(initial_thetas)

    # Record initial energy (t=0)
    sample_index = 0
    times[sample_index] = 0.0
    kinetic_energy_history[sample_index] = compute_kinetic_energy(state)
    potential_energy_history[sample_index] = compute_potential_energy(state)
    total_energy_history[sample_index] = compute_total_energy(state)
    sample_index += 1

    # Integrate and sample
    for step_index in range(1, num_steps + 1):
        state = rk4_step(state, dt, derivatives)

        if step_index % sample_interval == 0 or step_index == num_steps:
            current_time = step_index * dt
            times[sample_index] = current_time
            kinetic_energy_history[sample_index] = compute_kinetic_energy(state)
            potential_energy_history[sample_index] = compute_potential_energy(
                state,
            )
            total_energy_history[sample_index] = compute_total_energy(state)
            sample_index += 1

    # Trim arrays to actual number of samples recorded
    actual_samples = sample_index
    times = times[:actual_samples]
    kinetic_energy_history = kinetic_energy_history[:actual_samples]
    potential_energy_history = potential_energy_history[:actual_samples]
    total_energy_history = total_energy_history[:actual_samples]

    energy_evolution = {
        "times": times,
        "kinetic": kinetic_energy_history,
        "potential": potential_energy_history,
        "total": total_energy_history,
    }

    return energy_evolution


def classify_energy_transfer(
    initial_thetas: NDArray[np.float64],
    dt: float = 0.01,
    t_max: float = 5.0,
) -> NDArray[np.int64]:
    """Classify which bob receives the most kinetic energy before first flip.

    For each initial condition, simulates the triple pendulum and tracks
    the cumulative kinetic energy delivered to each bob up to the moment
    of first flip. The bob with the highest cumulative KE is the
    "dominant recipient." Pendulums that never flip within t_max are
    assigned a value of -1.

    The cumulative KE is approximated by trapezoidal integration of the
    instantaneous KE at each timestep.

    Args:
        initial_thetas: Initial angles of shape (N, 3) in **degrees**.
            All pendulums start from rest (omega = 0).
        dt: RK4 integration timestep in seconds.
        t_max: Maximum simulation time in seconds.

    Returns:
        NDArray of shape (N,) with integer values:
            0, 1, or 2 indicating the bob index that received the most
            cumulative kinetic energy before the first flip.
            -1 for pendulums that never flipped.
    """
    from src.simulation.batch_sim import rk4_step
    from src.simulation.metrics import detect_flips
    from src.simulation.physics import derivatives

    num_pendulums = initial_thetas.shape[0]
    num_steps = int(np.ceil(t_max / dt))

    # Initialize state: angles in radians, velocities zero
    state = np.zeros((num_pendulums, 6), dtype=np.float64)
    state[:, :NUM_BOBS] = np.radians(initial_thetas)

    # Track cumulative KE per bob (trapezoidal integration)
    cumulative_kinetic_energy = np.zeros(
        (num_pendulums, NUM_BOBS), dtype=np.float64,
    )
    previous_kinetic_energy = compute_kinetic_energy(state)  # (N, 3)

    # Track which pendulums have flipped and when
    has_flipped = np.zeros(num_pendulums, dtype=bool)
    # Store the cumulative KE at the moment of flip for each pendulum
    kinetic_energy_at_flip = np.zeros(
        (num_pendulums, NUM_BOBS), dtype=np.float64,
    )

    for step_index in range(num_steps):
        theta_prev = state[:, :NUM_BOBS].copy()

        state = rk4_step(state, dt, derivatives)

        theta_curr = state[:, :NUM_BOBS]
        current_kinetic_energy = compute_kinetic_energy(state)  # (N, 3)

        # Trapezoidal integration of KE for pendulums that haven't flipped
        active_mask = ~has_flipped  # (N,)
        cumulative_kinetic_energy[active_mask] += (
            0.5
            * dt
            * (
                previous_kinetic_energy[active_mask]
                + current_kinetic_energy[active_mask]
            )
        )

        # Check for new flips
        per_bob_flips = detect_flips(theta_prev, theta_curr)  # (N, 3)
        any_bob_flipped = np.any(per_bob_flips, axis=1)  # (N,)
        newly_flipped = any_bob_flipped & ~has_flipped

        if np.any(newly_flipped):
            kinetic_energy_at_flip[newly_flipped] = cumulative_kinetic_energy[
                newly_flipped
            ]
            has_flipped[newly_flipped] = True

        previous_kinetic_energy = current_kinetic_energy

        # Early stopping: all pendulums have flipped
        if np.all(has_flipped):
            break

    # Determine dominant recipient for each pendulum
    dominant_bob = np.full(num_pendulums, -1, dtype=np.int64)

    # For pendulums that flipped, find the bob with highest cumulative KE
    flipped_indices = np.where(has_flipped)[0]
    if flipped_indices.size > 0:
        dominant_bob[flipped_indices] = np.argmax(
            kinetic_energy_at_flip[flipped_indices], axis=1,
        ).astype(np.int64)

    return dominant_bob
