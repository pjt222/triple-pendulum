"""Triple pendulum equations of motion.

Implements the Lagrangian mechanics for a triple pendulum system with
equal point masses (m=1) and equal rod lengths (l=1). The equations are
fully vectorized with NumPy for batch computation over N pendulums.

PyTorch implementations are also provided for GPU-accelerated batch
simulation via torchdiffeq. These functions mirror the NumPy versions
exactly but operate on torch tensors and support both CPU and CUDA devices.

Physics summary:
    - Coupling matrix A = [[3,2,1],[2,2,1],[1,1,1]]
    - Mass matrix: M_ij = A_ij * cos(theta_i - theta_j)
    - Force vector: f_i = sum_j A_ij * sin(theta_i - theta_j) * omega_j^2
                          + gravity_weight_i * g * sin(theta_i)
    - Gravity weights: [3, 2, 1] (derived from n - i for n=3 pendulums)
    - Solve M * alpha = -f for angular accelerations
"""

import math

import numpy as np
from numpy.typing import NDArray

try:
    import torch
except ImportError:
    torch = None

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


# ---------------------------------------------------------------------------
# PyTorch implementations (GPU-accelerated, torchdiffeq-compatible)
# ---------------------------------------------------------------------------

def _require_torch() -> None:
    """Raise an ImportError if PyTorch is not available."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for the torch-based physics functions. "
            "Install it with: pip install torch"
        )


def _coupling_matrix_torch(device: "torch.device") -> "torch.Tensor":
    """Return the 3x3 coupling matrix as a torch tensor on the given device.

    This is a private helper that mirrors ``coupling_matrix()`` but produces
    a torch tensor placed on the requested device (CPU or CUDA).

    Args:
        device: The torch device to place the tensor on.

    Returns:
        Tensor of shape (3, 3) with the coupling coefficients.
    """
    return torch.tensor(
        [[3.0, 2.0, 1.0],
         [2.0, 2.0, 1.0],
         [1.0, 1.0, 1.0]],
        dtype=torch.float64,
        device=device,
    )


def mass_matrix_torch(theta: "torch.Tensor") -> "torch.Tensor":
    """Compute the mass (inertia) matrix for N pendulums using PyTorch.

    Mirrors ``mass_matrix()`` exactly:
        M_ij = A_ij * cos(theta_i - theta_j)

    Works on both CPU and CUDA tensors. The output device and dtype match
    the input.

    Args:
        theta: Angle tensor of shape (N, 3), one row per pendulum.

    Returns:
        Tensor of shape (N, 3, 3) containing the mass matrix for each
        pendulum configuration.
    """
    _require_torch()

    coupling = _coupling_matrix_torch(theta.device)  # (3, 3)

    # Pairwise angle differences: delta_ij = theta_i - theta_j
    # theta[:, :, None] is (N, 3, 1), theta[:, None, :] is (N, 1, 3)
    angle_differences = theta.unsqueeze(2) - theta.unsqueeze(1)  # (N, 3, 3)

    # Element-wise: M_ij = A_ij * cos(theta_i - theta_j)
    mass_matrices = coupling.unsqueeze(0) * torch.cos(angle_differences)  # (N, 3, 3)

    return mass_matrices


def force_vector_torch(
    theta: "torch.Tensor",
    omega: "torch.Tensor",
) -> "torch.Tensor":
    """Compute the generalized force vector for N pendulums using PyTorch.

    Mirrors ``force_vector()`` exactly:
        f_i = sum_j A_ij * sin(theta_i - theta_j) * omega_j^2
              + gravity_weight_i * g * sin(theta_i)

    Works on both CPU and CUDA tensors. The output device and dtype match
    the inputs.

    Args:
        theta: Angle tensor of shape (N, 3).
        omega: Angular velocity tensor of shape (N, 3).

    Returns:
        Tensor of shape (N, 3) with the force vector for each pendulum.
    """
    _require_torch()

    coupling = _coupling_matrix_torch(theta.device)  # (3, 3)
    gravity_weights = torch.tensor(
        [3.0, 2.0, 1.0], dtype=torch.float64, device=theta.device,
    )  # (3,)

    # Pairwise angle differences: (N, 3, 3)
    angle_differences = theta.unsqueeze(2) - theta.unsqueeze(1)

    # Coriolis/centripetal term: sum_j A_ij * sin(delta_ij) * omega_j^2
    omega_squared = omega ** 2  # (N, 3)
    coriolis_matrix = coupling.unsqueeze(0) * torch.sin(angle_differences)  # (N, 3, 3)
    coriolis_term = torch.einsum("nij,nj->ni", coriolis_matrix, omega_squared)  # (N, 3)

    # Gravitational term: gravity_weight_i * g * sin(theta_i)
    gravity_term = gravity_weights.unsqueeze(0) * GRAVITY * torch.sin(theta)  # (N, 3)

    force_vectors = coriolis_term + gravity_term  # (N, 3)

    return force_vectors


def derivatives_torch(t: "torch.Tensor", state: "torch.Tensor") -> "torch.Tensor":
    """Compute time derivatives of the pendulum state, torchdiffeq-compatible.

    This function has the signature ``f(t, y) -> dy/dt`` expected by
    ``torchdiffeq.odeint``. The system is autonomous (time-independent), so
    *t* is accepted but unused.

    Given state = [theta_1, theta_2, theta_3, omega_1, omega_2, omega_3],
    returns d(state)/dt = [omega_1, omega_2, omega_3, alpha_1, alpha_2, alpha_3]
    where alpha = M^{-1}(-f) are the angular accelerations.

    Args:
        t: Current time as a scalar tensor. Required by torchdiffeq but
            unused because the triple pendulum ODE is autonomous.
        state: State tensor of shape (N, 6) where columns 0-2 are angles
            and columns 3-5 are angular velocities.

    Returns:
        Tensor of shape (N, 6) with time derivatives of the state.
    """
    _require_torch()

    theta = state[:, :3]  # (N, 3)
    omega = state[:, 3:]  # (N, 3)

    # Build mass matrices and force vectors for all pendulums
    mass_matrices = mass_matrix_torch(theta)  # (N, 3, 3)
    force_vectors = force_vector_torch(theta, omega)  # (N, 3)

    # Solve M * alpha = -f for angular accelerations
    # torch.linalg.solve broadcasts over the batch dimension
    negative_force = -force_vectors.unsqueeze(-1)  # (N, 3, 1)
    angular_accelerations = torch.linalg.solve(mass_matrices, negative_force).squeeze(-1)  # (N, 3)

    # Assemble derivative vector: [omega, alpha]
    state_derivatives = torch.cat([omega, angular_accelerations], dim=1)  # (N, 6)

    return state_derivatives


# ---------------------------------------------------------------------------
# Numba JIT-compiled implementations (high-performance path)
# ---------------------------------------------------------------------------

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:
    @njit(cache=True)
    def _deriv_single(t0, t1, t2, w0, w1, w2):
        """Compute derivatives for a single triple pendulum.

        All physics inlined: mass matrix, force vector, 3x3 Cramer's rule.
        Returns (dtheta0, dtheta1, dtheta2, domega0, domega1, domega2).
        """
        g = 9.81

        # Angle differences (3 unique pairs)
        d01 = t0 - t1
        d02 = t0 - t2
        d12 = t1 - t2

        cd01 = math.cos(d01)
        cd02 = math.cos(d02)
        cd12 = math.cos(d12)

        sd01 = math.sin(d01)
        sd02 = math.sin(d02)
        sd12 = math.sin(d12)

        # Mass matrix: M[i][j] = A[i][j] * cos(theta_i - theta_j)
        # A = [[3,2,1],[2,2,1],[1,1,1]], diagonals: cos(0)=1
        m00 = 3.0
        m01 = 2.0 * cd01
        m02 = cd02
        m10 = m01  # symmetric
        m11 = 2.0
        m12 = cd12
        m20 = m02
        m21 = m12
        m22 = 1.0

        # Force vector: f_i = sum_j A[i][j]*sin(ti-tj)*wj^2 + gw[i]*g*sin(ti)
        # sin(ti-ti)=0 so diagonal coriolis terms vanish
        w0sq = w0 * w0
        w1sq = w1 * w1
        w2sq = w2 * w2

        f0 = 2.0 * sd01 * w1sq + sd02 * w2sq + 3.0 * g * math.sin(t0)
        f1 = -2.0 * sd01 * w0sq + sd12 * w2sq + 2.0 * g * math.sin(t1)
        f2 = -sd02 * w0sq - sd12 * w1sq + g * math.sin(t2)

        # Solve M * alpha = -f via Cramer's rule
        b0 = -f0
        b1 = -f1
        b2 = -f2

        det = (m00 * (m11 * m22 - m12 * m21)
             - m01 * (m10 * m22 - m12 * m20)
             + m02 * (m10 * m21 - m11 * m20))
        inv_det = 1.0 / det

        a0 = (b0 * (m11 * m22 - m12 * m21)
            - m01 * (b1 * m22 - m12 * b2)
            + m02 * (b1 * m21 - m11 * b2)) * inv_det
        a1 = (m00 * (b1 * m22 - m12 * b2)
            - b0 * (m10 * m22 - m12 * m20)
            + m02 * (m10 * b2 - b1 * m20)) * inv_det
        a2 = (m00 * (m11 * b2 - b1 * m21)
            - m01 * (m10 * b2 - b1 * m20)
            + b0 * (m10 * m21 - m11 * m20)) * inv_det

        return (w0, w1, w2, a0, a1, a2)

    @njit(parallel=True, cache=True)
    def rk4_step_numba(state, dt):
        """Full RK4 step for N pendulums, fully fused with Cramer's rule.

        Each pendulum is independent — computed in parallel via prange.
        Zero intermediate arrays: all values are scalars in CPU registers.
        """
        num_pendulums = state.shape[0]
        result = np.empty((num_pendulums, 6), dtype=np.float64)
        dt_half = 0.5 * dt
        dt_sixth = dt / 6.0

        for n in prange(num_pendulums):
            s0 = state[n, 0]
            s1 = state[n, 1]
            s2 = state[n, 2]
            s3 = state[n, 3]
            s4 = state[n, 4]
            s5 = state[n, 5]

            # k1 = f(state)
            k1_0, k1_1, k1_2, k1_3, k1_4, k1_5 = _deriv_single(
                s0, s1, s2, s3, s4, s5)

            # k2 = f(state + 0.5*dt*k1)
            k2_0, k2_1, k2_2, k2_3, k2_4, k2_5 = _deriv_single(
                s0 + dt_half * k1_0, s1 + dt_half * k1_1, s2 + dt_half * k1_2,
                s3 + dt_half * k1_3, s4 + dt_half * k1_4, s5 + dt_half * k1_5)

            # k3 = f(state + 0.5*dt*k2)
            k3_0, k3_1, k3_2, k3_3, k3_4, k3_5 = _deriv_single(
                s0 + dt_half * k2_0, s1 + dt_half * k2_1, s2 + dt_half * k2_2,
                s3 + dt_half * k2_3, s4 + dt_half * k2_4, s5 + dt_half * k2_5)

            # k4 = f(state + dt*k3)
            k4_0, k4_1, k4_2, k4_3, k4_4, k4_5 = _deriv_single(
                s0 + dt * k3_0, s1 + dt * k3_1, s2 + dt * k3_2,
                s3 + dt * k3_3, s4 + dt * k3_4, s5 + dt * k3_5)

            # result = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            result[n, 0] = s0 + dt_sixth * (k1_0 + 2.0 * k2_0 + 2.0 * k3_0 + k4_0)
            result[n, 1] = s1 + dt_sixth * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1)
            result[n, 2] = s2 + dt_sixth * (k1_2 + 2.0 * k2_2 + 2.0 * k3_2 + k4_2)
            result[n, 3] = s3 + dt_sixth * (k1_3 + 2.0 * k2_3 + 2.0 * k3_3 + k4_3)
            result[n, 4] = s4 + dt_sixth * (k1_4 + 2.0 * k2_4 + 2.0 * k3_4 + k4_4)
            result[n, 5] = s5 + dt_sixth * (k1_5 + 2.0 * k2_5 + 2.0 * k3_5 + k4_5)

        return result
