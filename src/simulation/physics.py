"""Triple pendulum equations of motion.

Implements the Lagrangian mechanics for a triple pendulum system with
equal point masses (m=1) and equal rod lengths (l=1). The equations are
fully vectorized with NumPy for batch computation over N pendulums.

Physics summary:
    - Coupling matrix A = [[3,2,1],[2,2,1],[1,1,1]]
    - Mass matrix: M_ij = A_ij * cos(theta_i - theta_j)
    - Force vector: f_i = sum_j A_ij * sin(theta_i - theta_j) * omega_j^2
                          + gravity_weight_i * g * sin(theta_i)
    - Gravity weights: [3, 2, 1] (derived from n - i for n=3 pendulums)
    - Solve M * alpha = -f for angular accelerations
"""

import numpy as np
from numpy.typing import NDArray

GRAVITY: float = 9.81
NUM_BOBS: int = 3
GRAVITY_WEIGHTS: NDArray[np.float64] = np.array([3.0, 2.0, 1.0])


def coupling_matrix() -> NDArray[np.float64]:
    """Return the 3x3 coupling matrix for a triple pendulum.

    The coupling matrix encodes the inertial coupling between pendulum
    segments. For equal masses, A_ij = min(n - max(i, j), n - min(i, j))
    simplifies to A = [[3,2,1],[2,2,1],[1,1,1]].

    Returns:
        NDArray of shape (3, 3) with the coupling coefficients.
    """
    return np.array([
        [3.0, 2.0, 1.0],
        [2.0, 2.0, 1.0],
        [1.0, 1.0, 1.0],
    ])


def mass_matrix(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the mass (inertia) matrix for N pendulums.

    M_ij = A_ij * cos(theta_i - theta_j) for each pendulum in the batch.

    Args:
        theta: Angle array of shape (N, 3), one row per pendulum.

    Returns:
        NDArray of shape (N, 3, 3) containing the mass matrix for each
        pendulum configuration.
    """
    coupling = coupling_matrix()  # (3, 3)

    # Compute pairwise angle differences: delta_ij = theta_i - theta_j
    # theta[:, :, None] has shape (N, 3, 1), theta[:, None, :] has shape (N, 1, 3)
    angle_differences = theta[:, :, np.newaxis] - theta[:, np.newaxis, :]  # (N, 3, 3)

    # Element-wise: M_ij = A_ij * cos(theta_i - theta_j)
    mass_matrices = coupling[np.newaxis, :, :] * np.cos(angle_differences)  # (N, 3, 3)

    return mass_matrices


def force_vector(
    theta: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the generalized force vector for N pendulums.

    f_i = sum_j A_ij * sin(theta_i - theta_j) * omega_j^2
          + gravity_weight_i * g * sin(theta_i)

    The first term captures Coriolis/centripetal coupling between segments.
    The second term captures gravitational torque.

    Args:
        theta: Angle array of shape (N, 3).
        omega: Angular velocity array of shape (N, 3).

    Returns:
        NDArray of shape (N, 3) with the force vector for each pendulum.
    """
    coupling = coupling_matrix()  # (3, 3)
    num_pendulums = theta.shape[0]

    # Pairwise angle differences: (N, 3, 3)
    angle_differences = theta[:, :, np.newaxis] - theta[:, np.newaxis, :]

    # Coriolis/centripetal term: sum_j A_ij * sin(delta_ij) * omega_j^2
    # coupling * sin(delta) has shape (N, 3, 3), omega^2 has shape (N, 3)
    omega_squared = omega ** 2  # (N, 3)
    coriolis_matrix = coupling[np.newaxis, :, :] * np.sin(angle_differences)  # (N, 3, 3)
    coriolis_term = np.einsum("nij,nj->ni", coriolis_matrix, omega_squared)  # (N, 3)

    # Gravitational term: gravity_weight_i * g * sin(theta_i)
    gravity_term = GRAVITY_WEIGHTS[np.newaxis, :] * GRAVITY * np.sin(theta)  # (N, 3)

    force_vectors = coriolis_term + gravity_term  # (N, 3)

    return force_vectors


def derivatives(state: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute time derivatives of the pendulum state vector.

    Given state = [theta_1, theta_2, theta_3, omega_1, omega_2, omega_3],
    returns d(state)/dt = [omega_1, omega_2, omega_3, alpha_1, alpha_2, alpha_3]
    where alpha = M^{-1}(-f) are the angular accelerations.

    Args:
        state: State array of shape (N, 6) where columns 0-2 are angles
            and columns 3-5 are angular velocities.

    Returns:
        NDArray of shape (N, 6) with time derivatives of the state.
    """
    theta = state[:, :3]  # (N, 3)
    omega = state[:, 3:]  # (N, 3)

    # Build mass matrices and force vectors for all pendulums
    mass_matrices = mass_matrix(theta)  # (N, 3, 3)
    force_vectors = force_vector(theta, omega)  # (N, 3)

    # Solve M * alpha = -f for angular accelerations
    # np.linalg.solve broadcasts over the batch dimension
    negative_force = -force_vectors[..., np.newaxis]  # (N, 3, 1)
    angular_accelerations = np.linalg.solve(mass_matrices, negative_force).squeeze(-1)  # (N, 3)

    # Assemble derivative vector: [omega, alpha]
    state_derivatives = np.empty_like(state)  # (N, 6)
    state_derivatives[:, :3] = omega
    state_derivatives[:, 3:] = angular_accelerations

    return state_derivatives
