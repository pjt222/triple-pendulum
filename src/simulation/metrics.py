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
