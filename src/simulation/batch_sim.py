"""Batch RK4 simulation for triple pendulum systems.

Integrates N triple pendulums simultaneously using the classical
fourth-order Runge-Kutta method. Supports flip detection via a callback
and early stopping when all pendulums have flipped.
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from src.simulation.metrics import FlipTimeTracker
from src.simulation.physics import derivatives


def rk4_step(
    state: NDArray[np.float64],
    dt: float,
    deriv_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Advance the state by one RK4 timestep.

    Implements the classical fourth-order Runge-Kutta method:
        k1 = f(y)
        k2 = f(y + dt/2 * k1)
        k3 = f(y + dt/2 * k2)
        k4 = f(y + dt * k3)
        y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Args:
        state: Current state array of shape (N, 6).
        dt: Timestep size.
        deriv_fn: Function that computes derivatives, mapping (N, 6) -> (N, 6).

    Returns:
        NDArray of shape (N, 6) with the updated state.
    """
    k1 = deriv_fn(state)
    k2 = deriv_fn(state + 0.5 * dt * k1)
    k3 = deriv_fn(state + 0.5 * dt * k2)
    k4 = deriv_fn(state + dt * k3)

    new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return new_state


def simulate_batch(
    initial_thetas: NDArray[np.float64],
    dt: float = 0.01,
    t_max: float = 15.0,
    flip_callback: Callable[[NDArray[np.float64], NDArray[np.float64], float], None]
    | None = None,
) -> dict:
    """Simulate N triple pendulums from given initial angles.

    All pendulums start from rest (omega = 0). Integration proceeds with
    fixed-step RK4 until t_max is reached or all pendulums have flipped
    (early stopping).

    Args:
        initial_thetas: Initial angles of shape (N, 3) in radians.
        dt: Integration timestep in seconds.
        t_max: Maximum simulation time in seconds.
        flip_callback: Optional callback invoked each step with signature
            (theta_prev, theta_curr, current_time). Typically a
            FlipTimeTracker.update method.

    Returns:
        Dictionary with:
            - "flip_times": NDArray of shape (N,) with first-flip time
              for each pendulum (NaN if never flipped).
            - "final_states": NDArray of shape (N, 6) with final state.
            - "metadata": dict with simulation parameters and statistics.
    """
    num_pendulums = initial_thetas.shape[0]
    num_steps = int(np.ceil(t_max / dt))
    progress_interval = max(1, num_steps // 10)

    # Build the flip tracker (used internally and optionally via callback)
    flip_tracker = FlipTimeTracker(num_pendulums)

    # Initialize state: [theta1, theta2, theta3, 0, 0, 0]
    state = np.zeros((num_pendulums, 6), dtype=np.float64)
    state[:, :3] = initial_thetas

    print(f"Simulating {num_pendulums} pendulums for {t_max}s (dt={dt}, steps={num_steps})")

    current_time = 0.0
    actual_steps_taken = 0

    for step_index in range(num_steps):
        theta_prev = state[:, :3].copy()

        state = rk4_step(state, dt, derivatives)
        current_time = (step_index + 1) * dt
        actual_steps_taken = step_index + 1

        theta_curr = state[:, :3]

        # Update internal flip tracker
        flip_tracker.update(theta_prev, theta_curr, current_time)

        # Invoke external callback if provided
        if flip_callback is not None:
            flip_callback(theta_prev, theta_curr, current_time)

        # Progress reporting every 10%
        if (step_index + 1) % progress_interval == 0:
            percent_complete = 100.0 * (step_index + 1) / num_steps
            flipped_fraction = flip_tracker.fraction_flipped
            print(
                f"  Progress: {percent_complete:5.1f}% "
                f"(t={current_time:.2f}s, "
                f"flipped={flipped_fraction:.1%})"
            )

        # Early stopping: all pendulums have flipped
        if flip_tracker.all_flipped:
            print(
                f"  Early stop at t={current_time:.2f}s: "
                f"all {num_pendulums} pendulums have flipped."
            )
            break

    flip_times = flip_tracker.get_flip_times()
    num_flipped = int(np.sum(~np.isnan(flip_times)))

    print(
        f"Simulation complete: {num_flipped}/{num_pendulums} pendulums flipped "
        f"({actual_steps_taken} steps, t_final={current_time:.2f}s)"
    )

    simulation_results = {
        "flip_times": flip_times,
        "final_states": state,
        "metadata": {
            "num_pendulums": num_pendulums,
            "dt": dt,
            "t_max": t_max,
            "actual_steps": actual_steps_taken,
            "t_final": current_time,
            "num_flipped": num_flipped,
            "fraction_flipped": num_flipped / num_pendulums,
        },
    }

    return simulation_results
