"""Symbolic derivation of n-link pendulum equations of motion.

Uses SymPy's Kane's method (sympy.physics.mechanics) to derive the
equations of motion for an n-link planar pendulum from first principles.
This provides a ground-truth reference for verifying the hardcoded
equations in physics.py, and can generate NumPy-callable derivative
functions via lambdify.

The approach follows Jake VanderPlas's symbolic pendulum derivation:
    1. Define generalized coordinates (angles) and speeds (angular velocities).
    2. Build a chain of reference frames, each rotated by its angle.
    3. Place point masses at the end of each rod.
    4. Apply gravity as the sole external force.
    5. Use KanesMethod to extract the mass matrix and forcing vector.

References:
    - VanderPlas, J. "Triple Pendulum CHAOS!" (blog post & notebook)
    - Kane, T.R. & Levinson, D.A. "Dynamics: Theory and Applications" (1985)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

try:
    import sympy as sp
    from sympy import Matrix, Symbol, cos, sin, symbols
    from sympy.physics.mechanics import (
        KanesMethod,
        Particle,
        Point,
        ReferenceFrame,
        dynamicsymbols,
    )

    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False


def _require_sympy() -> None:
    """Raise an ImportError if SymPy is not available."""
    if not _SYMPY_AVAILABLE:
        raise ImportError(
            "SymPy is required for symbolic equation derivation. "
            "Install it with: pip install sympy"
        )


def derive_n_pendulum_eom(
    n: int = 3,
    masses: list[float] | None = None,
    lengths: list[float] | None = None,
    g_val: float = 9.81,
) -> dict[str, Any]:
    """Derive equations of motion for an n-link planar pendulum using Kane's method.

    Constructs the full symbolic model: reference frames, points, particles,
    and gravity forces. Then applies Kane's method to obtain the mass matrix
    and forcing vector such that M * u_dot = forcing.

    Args:
        n: Number of pendulum links. Defaults to 3 (triple pendulum).
        masses: List of n mass values. If None, all masses are 1.0.
        lengths: List of n rod lengths. If None, all lengths are 1.0.
        g_val: Gravitational acceleration. Defaults to 9.81.

    Returns:
        Dictionary with keys:
            - ``"kane"``: The KanesMethod object.
            - ``"coordinates"``: List of n generalized coordinate symbols (angles).
            - ``"speeds"``: List of n generalized speed symbols (angular velocities).
            - ``"mass_matrix"``: Symbolic mass matrix (n x n Matrix).
            - ``"forcing"``: Symbolic forcing vector (n x 1 Matrix).
            - ``"parameters"``: Dict of symbol-to-value mappings for masses,
              lengths, and gravity.

    Raises:
        ImportError: If SymPy is not installed.
        ValueError: If n < 1, or if masses/lengths have wrong length.
    """
    _require_sympy()

    if n < 1:
        raise ValueError(f"Number of links must be >= 1, got {n}")
    if masses is not None and len(masses) != n:
        raise ValueError(f"Expected {n} masses, got {len(masses)}")
    if lengths is not None and len(lengths) != n:
        raise ValueError(f"Expected {n} lengths, got {len(lengths)}")

    # Default parameters: equal unit masses and lengths
    if masses is None:
        masses = [1.0] * n
    if lengths is None:
        lengths = [1.0] * n

    # --- Symbolic parameters ---
    gravity_symbol = symbols("g", positive=True)
    mass_symbols = symbols(f"m1:{n + 1}", positive=True)
    length_symbols = symbols(f"l1:{n + 1}", positive=True)

    # Map symbolic parameters to numeric values
    parameter_values = {gravity_symbol: g_val}
    for i in range(n):
        parameter_values[mass_symbols[i]] = masses[i]
        parameter_values[length_symbols[i]] = lengths[i]

    # --- Generalized coordinates and speeds ---
    # q_i = angle of link i from vertical (measured in the inertial frame)
    # u_i = dq_i/dt = angular velocity of link i
    generalized_coordinates = dynamicsymbols(f"q1:{n + 1}")
    generalized_speeds = dynamicsymbols(f"u1:{n + 1}")

    # Kinematic differential equations: dq_i/dt = u_i
    kinematic_des = [
        generalized_coordinates[i].diff(sp.Symbol("t")) - generalized_speeds[i]
        for i in range(n)
    ]

    # --- Build the mechanical system ---
    # Inertial reference frame
    inertial_frame = ReferenceFrame("N")

    # Pivot point (fixed origin at the top)
    pivot = Point("O")
    pivot.set_vel(inertial_frame, 0)

    # Downward unit vector in the inertial frame: -N.y
    # Gravity acts in the -y direction (downward)
    gravity_direction = -inertial_frame.y

    # Storage for the Kane's method inputs
    frames = [inertial_frame]
    points = [pivot]
    particles = []
    force_list = []

    for i in range(n):
        # Reference frame for link i, rotated by q_i about the z-axis
        # (z-axis points out of the plane for a 2D pendulum)
        link_frame = ReferenceFrame(f"B{i + 1}")
        link_frame.orient_axis(inertial_frame, inertial_frame.z, generalized_coordinates[i])
        link_frame.set_ang_vel(
            inertial_frame,
            generalized_speeds[i] * inertial_frame.z,
        )
        frames.append(link_frame)

        # Position of the bob at end of link i
        # The rod hangs from the previous point. In the link's rotated frame,
        # "down along the rod" corresponds to -N.y rotated by q_i, which is:
        #   sin(q_i) * N.x - cos(q_i) * N.y
        # We express this using the inertial frame components directly.
        bob_point = Point(f"P{i + 1}")
        rod_vector = length_symbols[i] * (
            sin(generalized_coordinates[i]) * inertial_frame.x
            - cos(generalized_coordinates[i]) * inertial_frame.y
        )
        bob_point.set_pos(points[i], rod_vector)

        # Velocity: must be computed in the inertial frame
        bob_point.set_vel(inertial_frame, bob_point.pos_from(pivot).diff(sp.Symbol("t"), inertial_frame))

        points.append(bob_point)

        # Particle (point mass) at the bob position
        particle = Particle(f"Pa{i + 1}", bob_point, mass_symbols[i])
        particles.append(particle)

        # Gravity force on this particle
        gravity_force = (bob_point, mass_symbols[i] * gravity_symbol * gravity_direction)
        force_list.append(gravity_force)

    # --- Apply Kane's method ---
    kane = KanesMethod(
        inertial_frame,
        q_ind=list(generalized_coordinates),
        u_ind=list(generalized_speeds),
        kd_eqs=kinematic_des,
    )
    kane.kanes_equations([p for p in particles], force_list)

    # Extract mass matrix and forcing vector
    symbolic_mass_matrix = kane.mass_matrix_full
    symbolic_forcing = kane.forcing_full

    # The "full" versions include both kinematic and dynamic equations.
    # The dynamic portion (lower n rows) gives M * u_dot = forcing.
    # Extract just the dynamic portion for the n generalized speeds.
    dynamic_mass_matrix = symbolic_mass_matrix[n:, n:]
    dynamic_forcing = symbolic_forcing[n:, :]

    return {
        "kane": kane,
        "coordinates": list(generalized_coordinates),
        "speeds": list(generalized_speeds),
        "mass_matrix": dynamic_mass_matrix,
        "forcing": dynamic_forcing,
        "parameters": parameter_values,
    }


def generate_numpy_function(
    eom_result: dict[str, Any],
    function_name: str = "pendulum_derivatives",
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Generate a NumPy-callable derivative function from symbolic EOM.

    Uses ``sympy.utilities.lambdify`` to convert the symbolic mass matrix
    and forcing vector into fast NumPy functions. The returned function
    takes a flat state array [q1, ..., qn, u1, ..., un] and returns the
    time derivatives [u1, ..., un, udot1, ..., udotn].

    Args:
        eom_result: Dictionary returned by ``derive_n_pendulum_eom``.
        function_name: Name for the generated function (used in its
            ``__name__`` attribute). Defaults to ``"pendulum_derivatives"``.

    Returns:
        A callable ``f(state) -> derivatives`` where:
            - ``state`` is a 1D array of length 2n (angles then angular velocities).
            - ``derivatives`` is a 1D array of length 2n (angular velocities then
              angular accelerations).

    Raises:
        ImportError: If SymPy is not installed.
    """
    _require_sympy()
    from sympy.utilities.lambdify import lambdify

    coordinates = eom_result["coordinates"]
    speeds = eom_result["speeds"]
    parameter_values = eom_result["parameters"]
    symbolic_mass_matrix = eom_result["mass_matrix"]
    symbolic_forcing = eom_result["forcing"]

    n = len(coordinates)

    # Substitute numeric parameter values into the symbolic expressions
    substituted_mass_matrix = symbolic_mass_matrix.subs(parameter_values)
    substituted_forcing = symbolic_forcing.subs(parameter_values)

    # The remaining free symbols are the generalized coordinates and speeds
    state_symbols = list(coordinates) + list(speeds)

    # Create lambdified functions for mass matrix and forcing
    mass_matrix_func = lambdify(state_symbols, substituted_mass_matrix, modules="numpy")
    forcing_func = lambdify(state_symbols, substituted_forcing, modules="numpy")

    def numpy_derivatives(state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute state derivatives for a single pendulum state.

        Args:
            state: Flat array of length 2n: [q1, ..., qn, u1, ..., un].

        Returns:
            Flat array of length 2n: [u1, ..., un, udot1, ..., udotn].
        """
        state_values = list(state)

        # Evaluate mass matrix and forcing at the current state
        mass_matrix_numerical = np.array(mass_matrix_func(*state_values), dtype=np.float64)
        forcing_numerical = np.array(forcing_func(*state_values), dtype=np.float64).flatten()

        # Solve M * u_dot = forcing for angular accelerations
        angular_accelerations = np.linalg.solve(mass_matrix_numerical, forcing_numerical)

        # Assemble derivative vector: [velocities, accelerations]
        derivative_vector = np.empty(2 * n, dtype=np.float64)
        derivative_vector[:n] = state[n:]  # dq/dt = u
        derivative_vector[n:] = angular_accelerations  # du/dt = M^{-1} * forcing

        return derivative_vector

    numpy_derivatives.__name__ = function_name
    numpy_derivatives.__doc__ = (
        f"NumPy derivative function for {n}-link pendulum, "
        f"generated from symbolic EOM via lambdify."
    )

    return numpy_derivatives


def verify_against_hardcoded(
    n_test_states: int = 100,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Compare symbolic EOM derivatives against the hardcoded physics.py.

    Generates random states and evaluates derivatives using both the
    symbolically-derived function and the hardcoded ``physics.derivatives``
    function. Reports the maximum and mean absolute errors.

    Args:
        n_test_states: Number of random states to test. Defaults to 100.
        atol: Absolute tolerance for the ``all_close`` check.
            Defaults to 1e-10.

    Returns:
        Dictionary with keys:
            - ``"max_error"``: Maximum absolute error across all states
              and components.
            - ``"mean_error"``: Mean absolute error across all states
              and components.
            - ``"all_close"``: True if all errors are below ``atol``.
            - ``"n_states_tested"``: Number of states tested.

    Raises:
        ImportError: If SymPy is not installed.
    """
    _require_sympy()

    from src.simulation.physics import derivatives as hardcoded_derivatives

    # Derive symbolic EOM for the standard triple pendulum (n=3, m=1, l=1)
    eom_result = derive_n_pendulum_eom(n=3, masses=None, lengths=None, g_val=9.81)
    symbolic_func = generate_numpy_function(eom_result)

    # Generate random test states
    # Angles in [-pi, pi], angular velocities in [-5, 5]
    rng = np.random.default_rng(seed=42)
    random_angles = rng.uniform(-np.pi, np.pi, size=(n_test_states, 3))
    random_velocities = rng.uniform(-5.0, 5.0, size=(n_test_states, 3))
    test_states = np.hstack([random_angles, random_velocities])

    # Compute derivatives using both methods
    hardcoded_results = hardcoded_derivatives(test_states)  # (N, 6)

    symbolic_results = np.empty_like(test_states)
    for i in range(n_test_states):
        symbolic_results[i] = symbolic_func(test_states[i])

    # Compute errors
    absolute_errors = np.abs(hardcoded_results - symbolic_results)
    max_error = float(np.max(absolute_errors))
    mean_error = float(np.mean(absolute_errors))
    all_close = bool(np.all(absolute_errors < atol))

    return {
        "max_error": max_error,
        "mean_error": mean_error,
        "all_close": all_close,
        "n_states_tested": n_test_states,
    }


def print_eom_latex(eom_result: dict[str, Any]) -> None:
    """Pretty-print the equations of motion in LaTeX form.

    Displays the symbolic mass matrix and forcing vector using SymPy's
    LaTeX printer. Also prints a summary of the generalized coordinates
    and speeds.

    Args:
        eom_result: Dictionary returned by ``derive_n_pendulum_eom``.

    Raises:
        ImportError: If SymPy is not installed.
    """
    _require_sympy()

    coordinates = eom_result["coordinates"]
    speeds = eom_result["speeds"]
    symbolic_mass_matrix = eom_result["mass_matrix"]
    symbolic_forcing = eom_result["forcing"]

    n = len(coordinates)

    print(f"=== {n}-Link Pendulum Equations of Motion ===")
    print()

    # Generalized coordinates and speeds
    print("Generalized coordinates (angles):")
    for i, coordinate in enumerate(coordinates):
        print(f"  q{i + 1} = {coordinate}")
    print()

    print("Generalized speeds (angular velocities):")
    for i, speed in enumerate(speeds):
        print(f"  u{i + 1} = {speed}")
    print()

    # Mass matrix
    print("Mass matrix M (such that M * u_dot = forcing):")
    print()
    print("  LaTeX:")
    print(f"  {sp.latex(symbolic_mass_matrix)}")
    print()
    print("  Pretty print:")
    sp.pprint(symbolic_mass_matrix)
    print()

    # Forcing vector
    print("Forcing vector F:")
    print()
    print("  LaTeX:")
    print(f"  {sp.latex(symbolic_forcing)}")
    print()
    print("  Pretty print:")
    sp.pprint(symbolic_forcing)
    print()

    # Full equation
    print("Equation of motion: M * u_dot = F")
    print()

    # Substitute default parameters for a cleaner view
    parameter_values = eom_result["parameters"]
    substituted_mass_matrix = symbolic_mass_matrix.subs(parameter_values)
    substituted_forcing = symbolic_forcing.subs(parameter_values)

    print("With default parameters substituted:")
    print()
    print("  Mass matrix:")
    sp.pprint(substituted_mass_matrix)
    print()
    print("  Forcing vector:")
    sp.pprint(substituted_forcing)
    print()


if __name__ == "__main__":
    _require_sympy()

    print("Deriving triple pendulum EOM symbolically...")
    result = derive_n_pendulum_eom(n=3)
    print("Done.\n")

    print_eom_latex(result)

    print("\nGenerating NumPy function...")
    numpy_func = generate_numpy_function(result)
    print(f"Generated: {numpy_func.__name__}")
    print(f"  {numpy_func.__doc__}")

    # Quick smoke test
    test_state = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    derivative_output = numpy_func(test_state)
    print(f"\nSmoke test state:       {test_state}")
    print(f"Computed derivatives:   {derivative_output}")

    print("\nVerifying against hardcoded physics.py...")
    verification = verify_against_hardcoded(n_test_states=100)
    print(f"  States tested: {verification['n_states_tested']}")
    print(f"  Max error:     {verification['max_error']:.2e}")
    print(f"  Mean error:    {verification['mean_error']:.2e}")
    print(f"  All close:     {verification['all_close']}")
