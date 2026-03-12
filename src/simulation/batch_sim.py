"""Batch simulation for triple pendulum systems (CPU and GPU).

CPU path: Integrates N triple pendulums simultaneously using the classical
fourth-order Runge-Kutta method. Supports flip detection via a callback
and early stopping when all pendulums have flipped.

GPU path: Uses torchdiffeq with the dopri5 adaptive solver on CUDA.
Processes pendulums in chunks to limit VRAM usage. Flip detection is
performed post-hoc on the full trajectory returned by the ODE solver.

Memmap path: Wraps CPU or GPU simulation with memory-mapped output for
large grids that do not fit in RAM. Supports resuming interrupted runs.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from src.simulation.metrics import FlipTimeTracker
from src.simulation.physics import derivatives
from src.utils.grid import make_grid
from src.utils.io import create_memmap_output

try:
    import torch
    import torchdiffeq
except ImportError:
    torch = None
    torchdiffeq = None

try:
    from src.simulation.cuda_sim import (
        HAS_CUPY,
        HAS_PYCUDA,
        HAS_TORCH as HAS_TORCH_CUDA,
        simulate_batch_cuda,
        simulate_batch_cupy,
        simulate_batch_gpu_rk4,
    )
except ImportError:
    HAS_CUPY = False
    HAS_PYCUDA = False
    HAS_TORCH_CUDA = False


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
    logfile: str | None = None,
) -> dict:
    """Simulate N triple pendulums from given initial angles.

    All pendulums start from rest (omega = 0). Integration proceeds with
    fixed-step RK4 until t_max is reached or all pendulums have flipped
    (early stopping).

    Args:
        initial_thetas: Initial angles of shape (N, 3) in degrees.
        dt: Integration timestep in seconds.
        t_max: Maximum simulation time in seconds.
        flip_callback: Optional callback invoked each step with signature
            (theta_prev, theta_curr, current_time). Typically a
            FlipTimeTracker.update method.
        logfile: Optional path to a progress logfile. When provided,
            timestamped progress lines are written and flushed after
            each update so the file can be monitored in real time.

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

    # ── Logging helper ──────────────────────────────────────────────
    log_fh = open(logfile, "w") if logfile else None

    def log(msg: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        if log_fh:
            log_fh.write(line + "\n")
            log_fh.flush()

    try:
        # Build the flip tracker (used internally and optionally via callback)
        flip_tracker = FlipTimeTracker(num_pendulums)

        # Convert degrees to radians and initialize state: [theta1, theta2, theta3, 0, 0, 0]
        state = np.zeros((num_pendulums, 6), dtype=np.float64)
        state[:, :3] = np.radians(initial_thetas)

        log(f"Simulating {num_pendulums} pendulums for {t_max}s (dt={dt}, steps={num_steps})")

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
                log(
                    f"Progress: {percent_complete:5.1f}% "
                    f"(t={current_time:.2f}s, "
                    f"flipped={flipped_fraction:.1%}, "
                    f"step {step_index + 1}/{num_steps})"
                )

            # Early stopping: all pendulums have flipped
            if flip_tracker.all_flipped:
                log(
                    f"Early stop at t={current_time:.2f}s: "
                    f"all {num_pendulums} pendulums have flipped."
                )
                break

        flip_times = flip_tracker.get_flip_times()
        num_flipped = int(np.sum(~np.isnan(flip_times)))

        log(
            f"Simulation complete: {num_flipped}/{num_pendulums} pendulums flipped "
            f"({actual_steps_taken} steps, t_final={current_time:.2f}s)"
        )

    finally:
        if log_fh:
            log_fh.close()

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


def simulate_batch_fast(
    initial_thetas: NDArray[np.float64],
    dt: float = 0.01,
    t_max: float = 15.0,
    logfile: str | None = None,
) -> dict:
    """Simulate N triple pendulums using Numba JIT (fast path).

    Uses a fully fused RK4 integrator with analytical 3x3 Cramer's rule
    solver, compiled via Numba. Typically 10x faster than the NumPy path.
    Falls back to the standard NumPy path if Numba is not installed.

    Args:
        initial_thetas: Initial angles of shape (N, 3) in degrees.
        dt: Integration timestep in seconds.
        t_max: Maximum simulation time in seconds.
        logfile: Optional path to a progress logfile.

    Returns:
        Dictionary with "flip_times", "final_states", and "metadata".
    """
    from src.simulation.physics import HAS_NUMBA

    if not HAS_NUMBA:
        print("Numba not available, falling back to NumPy path")
        return simulate_batch(initial_thetas, dt=dt, t_max=t_max, logfile=logfile)

    from src.simulation.physics import rk4_step_numba

    num_pendulums = initial_thetas.shape[0]
    num_steps = int(np.ceil(t_max / dt))
    progress_interval = max(1, num_steps // 10)

    # ── Logging helper ──────────────────────────────────────────────
    log_fh = open(logfile, "w") if logfile else None

    def log(msg: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        if log_fh:
            log_fh.write(line + "\n")
            log_fh.flush()

    try:
        flip_tracker = FlipTimeTracker(num_pendulums)

        # Convert degrees to radians, initialize state [theta, omega=0]
        state = np.zeros((num_pendulums, 6), dtype=np.float64)
        state[:, :3] = np.radians(initial_thetas)

        log(f"[numba] Simulating {num_pendulums} pendulums for {t_max}s "
            f"(dt={dt}, steps={num_steps})")

        # JIT warmup: compile on a tiny batch
        log("[numba] Compiling JIT kernels (first call only)...")
        warmup_start = time.monotonic()
        _warmup_state = np.zeros((2, 6), dtype=np.float64)
        rk4_step_numba(_warmup_state, dt)
        warmup_elapsed = time.monotonic() - warmup_start
        log(f"[numba] JIT compilation done in {warmup_elapsed:.1f}s")

        current_time = 0.0
        actual_steps_taken = 0
        sim_start = time.monotonic()

        for step_index in range(num_steps):
            prev_thetas = state[:, :3]

            state = rk4_step_numba(state, dt)
            current_time = (step_index + 1) * dt
            actual_steps_taken = step_index + 1

            curr_thetas = state[:, :3]

            # Flip detection (fast NumPy operations, not the bottleneck)
            flip_tracker.update(prev_thetas, curr_thetas, current_time)

            # Progress reporting every 10%
            if (step_index + 1) % progress_interval == 0:
                percent_complete = 100.0 * (step_index + 1) / num_steps
                flipped_fraction = flip_tracker.fraction_flipped
                elapsed = time.monotonic() - sim_start
                steps_per_sec = (step_index + 1) / elapsed
                log(
                    f"Progress: {percent_complete:5.1f}% "
                    f"(t={current_time:.2f}s, "
                    f"flipped={flipped_fraction:.1%}, "
                    f"step {step_index + 1}/{num_steps}, "
                    f"{steps_per_sec:.0f} steps/s)"
                )

            # Early stopping: all pendulums have flipped
            if flip_tracker.all_flipped:
                log(
                    f"Early stop at t={current_time:.2f}s: "
                    f"all {num_pendulums} pendulums have flipped."
                )
                break

        flip_times = flip_tracker.get_flip_times()
        num_flipped = int(np.sum(~np.isnan(flip_times)))
        total_elapsed = time.monotonic() - sim_start

        log(
            f"Simulation complete: {num_flipped}/{num_pendulums} pendulums flipped "
            f"({actual_steps_taken} steps, t_final={current_time:.2f}s, "
            f"wall={total_elapsed:.1f}s)"
        )

    finally:
        if log_fh:
            log_fh.close()

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
            "backend": "numba",
            "wall_time_seconds": round(total_elapsed, 2),
        },
    }

    return simulation_results


# ---------------------------------------------------------------------------
# GPU simulation via torchdiffeq
# ---------------------------------------------------------------------------


def _require_torch_and_torchdiffeq() -> None:
    """Raise an ImportError if PyTorch or torchdiffeq is not available."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for GPU simulation. "
            "Install it with: pip install torch"
        )
    if torchdiffeq is None:
        raise ImportError(
            "torchdiffeq is required for GPU simulation. "
            "Install it with: pip install torchdiffeq"
        )


def _detect_flips_from_trajectory_torch(
    trajectory: "torch.Tensor",
    time_points: "torch.Tensor",
) -> NDArray[np.float64]:
    """Detect the first flip time for each pendulum from a full trajectory.

    A flip is detected when the angle, wrapped to [-pi, pi], shows a
    discontinuity larger than pi between consecutive timesteps. This
    mirrors the CPU-side ``detect_flips`` logic from ``metrics.py``.

    Args:
        trajectory: Full ODE solution of shape (T, N, 6) where T is the
            number of time points, N is the number of pendulums, and the
            last dimension holds [theta1, theta2, theta3, omega1, omega2,
            omega3].
        time_points: 1-D tensor of shape (T,) with the simulation times.

    Returns:
        NumPy array of shape (N,) with the time of first flip for each
        pendulum. Pendulums that never flipped have NaN.
    """
    # Extract angle trajectories: (T, N, 3)
    theta_trajectory = trajectory[:, :, :3]

    # Wrap angles to [-pi, pi]
    wrapped_theta = torch.remainder(theta_trajectory + torch.pi, 2 * torch.pi) - torch.pi

    # Compute wrapped difference between consecutive timesteps: (T-1, N, 3)
    wrapped_prev = wrapped_theta[:-1]
    wrapped_curr = wrapped_theta[1:]
    wrapped_delta = torch.abs(wrapped_curr - wrapped_prev)

    # A flip occurs when the wrapped delta exceeds pi: (T-1, N, 3)
    per_bob_flips = wrapped_delta > torch.pi

    # A pendulum flips if ANY of its 3 bobs flipped: (T-1, N)
    any_bob_flipped = torch.any(per_bob_flips, dim=2)

    # Move to CPU for the scan (small relative cost)
    any_bob_flipped_cpu = any_bob_flipped.cpu().numpy()  # (T-1, N)
    time_values = time_points.cpu().numpy()  # (T,)

    num_pendulums = any_bob_flipped_cpu.shape[1]
    first_flip_times = np.full(num_pendulums, np.nan, dtype=np.float64)

    # For each timestep transition, record the time for newly-flipped pendulums
    for step_index in range(any_bob_flipped_cpu.shape[0]):
        flipped_this_step = any_bob_flipped_cpu[step_index]
        not_yet_recorded = np.isnan(first_flip_times)
        newly_flipped = flipped_this_step & not_yet_recorded
        # The flip happened at the end of this transition, so use step_index+1
        first_flip_times[newly_flipped] = time_values[step_index + 1]

        # Early exit if all pendulums have flipped
        if not np.any(not_yet_recorded & ~flipped_this_step):
            break

    return first_flip_times


def simulate_batch_gpu(
    initial_thetas: NDArray[np.float64],
    dt: float = 0.01,
    t_max: float = 15.0,
    chunk_size: int = 10000,
    device: str = "cuda",
    logfile: str | None = None,
) -> dict:
    """Simulate N triple pendulums on the GPU using torchdiffeq.

    All pendulums start from rest (omega = 0). Uses the dopri5 adaptive
    solver. Pendulums are processed in chunks of ``chunk_size`` to limit
    GPU memory usage.

    Args:
        initial_thetas: Initial angles of shape (N, 3) in degrees.
        dt: Time spacing for the output time points in seconds. The
            adaptive solver may use finer internal steps.
        t_max: Maximum simulation time in seconds.
        chunk_size: Number of pendulums to process per GPU chunk.
        device: Torch device string (e.g. "cuda", "cuda:0", "cpu").
        logfile: Optional path to a progress logfile. When provided,
            timestamped progress lines are written and flushed after
            each update so the file can be monitored in real time.

    Returns:
        Dictionary with:
            - "flip_times": NDArray of shape (N,) with first-flip time
              for each pendulum (NaN if never flipped).
            - "final_states": NDArray of shape (N, 6) with final state.
            - "metadata": dict with simulation parameters and statistics.
    """
    _require_torch_and_torchdiffeq()

    from src.simulation.physics import derivatives_torch

    num_pendulums = initial_thetas.shape[0]
    num_chunks = int(np.ceil(num_pendulums / chunk_size))

    # ── Logging helper ──────────────────────────────────────────────
    log_fh = open(logfile, "w") if logfile else None

    def log(msg: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        if log_fh:
            log_fh.write(line + "\n")
            log_fh.flush()

    try:
        # Convert all initial angles from degrees to radians
        initial_thetas_rad = np.radians(initial_thetas.astype(np.float64))

        # Build time points on the target device
        torch_device = torch.device(device)
        time_points = torch.arange(
            0, t_max + dt, dt, dtype=torch.float64, device=torch_device,
        )

        log(
            f"GPU simulation: {num_pendulums} pendulums, "
            f"{num_chunks} chunk(s) of up to {chunk_size}, "
            f"t_max={t_max}s, device={device}"
        )

        # Pre-allocate result arrays
        all_flip_times = np.full(num_pendulums, np.nan, dtype=np.float64)
        all_final_states = np.zeros((num_pendulums, 6), dtype=np.float64)

        for chunk_index in range(num_chunks):
            chunk_start = chunk_index * chunk_size
            chunk_end = min(chunk_start + chunk_size, num_pendulums)
            chunk_count = chunk_end - chunk_start

            log(
                f"Chunk {chunk_index + 1}/{num_chunks}: "
                f"pendulums {chunk_start}..{chunk_end - 1} ({chunk_count} total)"
            )

            # Build initial state tensor: [theta1, theta2, theta3, 0, 0, 0]
            initial_state = torch.zeros(
                (chunk_count, 6), dtype=torch.float64, device=torch_device,
            )
            initial_state[:, :3] = torch.from_numpy(
                initial_thetas_rad[chunk_start:chunk_end]
            ).to(torch_device)

            # Integrate using dopri5 (returns shape (T, N_chunk, 6))
            with torch.no_grad():
                trajectory = torchdiffeq.odeint(
                    derivatives_torch,
                    initial_state,
                    time_points,
                    method="dopri5",
                )

            # Detect flips from the full trajectory
            chunk_flip_times = _detect_flips_from_trajectory_torch(
                trajectory, time_points,
            )
            all_flip_times[chunk_start:chunk_end] = chunk_flip_times

            # Store final state (last timestep)
            final_state_chunk = trajectory[-1].cpu().numpy()
            all_final_states[chunk_start:chunk_end] = final_state_chunk

            num_flipped_chunk = int(np.sum(~np.isnan(chunk_flip_times)))
            cumulative_done = chunk_end
            percent_complete = 100.0 * cumulative_done / num_pendulums
            log(
                f"  Chunk {chunk_index + 1}/{num_chunks} done: "
                f"flipped {num_flipped_chunk}/{chunk_count} "
                f"({num_flipped_chunk / chunk_count:.1%}) — "
                f"overall {percent_complete:.1f}%"
            )

        total_flipped = int(np.sum(~np.isnan(all_flip_times)))
        log(
            f"GPU simulation complete: {total_flipped}/{num_pendulums} pendulums "
            f"flipped ({total_flipped / num_pendulums:.1%})"
        )

    finally:
        if log_fh:
            log_fh.close()

    simulation_results = {
        "flip_times": all_flip_times,
        "final_states": all_final_states,
        "metadata": {
            "num_pendulums": num_pendulums,
            "dt": dt,
            "t_max": t_max,
            "chunk_size": chunk_size,
            "device": device,
            "num_chunks": num_chunks,
            "num_flipped": total_flipped,
            "fraction_flipped": total_flipped / num_pendulums,
            "method": "dopri5",
        },
    }

    return simulation_results


# ---------------------------------------------------------------------------
# Automatic backend selection
# ---------------------------------------------------------------------------


def simulate_batch_auto(
    initial_thetas: NDArray[np.float64],
    dt: float = 0.01,
    t_max: float = 15.0,
    logfile: str | None = None,
) -> dict:
    """Simulate N triple pendulums using the best available backend.

    Selection order: PyCUDA CUDA C > PyTorch GPU RK4 > Numba CPU > NumPy CPU.

    Args:
        initial_thetas: Initial angles of shape (N, 3) in degrees.
        dt: Integration timestep in seconds.
        t_max: Maximum simulation time in seconds.
        logfile: Optional path to a progress logfile.

    Returns:
        Dictionary with "flip_times", "final_states", and "metadata".
    """
    if HAS_PYCUDA:
        print("Backend: PyCUDA CUDA C kernel")
        return simulate_batch_cuda(initial_thetas, dt=dt, t_max=t_max, logfile=logfile)

    if HAS_CUPY:
        print("Backend: CuPy CUDA C kernel")
        return simulate_batch_cupy(initial_thetas, dt=dt, t_max=t_max, logfile=logfile)

    if HAS_TORCH_CUDA:
        print("Backend: PyTorch GPU RK4")
        return simulate_batch_gpu_rk4(initial_thetas, dt=dt, t_max=t_max, logfile=logfile)

    from src.simulation.physics import HAS_NUMBA

    if HAS_NUMBA:
        print("Backend: Numba CPU")
        return simulate_batch_fast(initial_thetas, dt=dt, t_max=t_max, logfile=logfile)

    print("Backend: NumPy CPU")
    return simulate_batch(initial_thetas, dt=dt, t_max=t_max, logfile=logfile)


# ---------------------------------------------------------------------------
# Memory-mapped batch processing  (Issue #13)
# ---------------------------------------------------------------------------


def simulate_batch_memmap(
    grid_size: int,
    theta_range: tuple[float, float] = (-170.0, 170.0),
    dt: float = 0.01,
    t_max: float = 15.0,
    chunk_size: int = 5000,
    output_path: str = "data/simulation",
    use_gpu: bool = False,
    gpu_device: str = "cuda",
) -> str:
    """Run a full-grid simulation writing results to a memory-mapped file.

    This function orchestrates the complete pipeline: grid construction,
    chunked simulation (CPU or GPU), and incremental persistence to a
    NumPy memmap file.  A JSON metadata sidecar is written alongside the
    array so the results are fully self-describing.

    The function supports **resuming** interrupted runs.  On startup it
    checks the existing memmap for chunks that already contain non-NaN
    values and skips them automatically.

    Args:
        grid_size: Number of sample points per theta axis.  Total grid
            points will be ``grid_size ** 3``.
        theta_range: ``(theta_min, theta_max)`` in degrees.
        dt: Integration timestep (CPU) or output spacing (GPU) in seconds.
        t_max: Maximum simulation time in seconds.
        chunk_size: Number of pendulums to simulate per batch.
        output_path: Base path for output files (without extension).  Two
            files are created: ``{output_path}.npy`` (memmap array) and
            ``{output_path}_meta.json`` (metadata sidecar).
        use_gpu: If ``True``, use :func:`simulate_batch_gpu`; otherwise
            use :func:`simulate_batch` (CPU).
        gpu_device: Torch device string, only used when *use_gpu* is True.

    Returns:
        The *output_path* string so callers know where results were saved.
    """
    total_pendulums = grid_size ** 3
    grid_shape = (grid_size, grid_size, grid_size)
    memmap_file_path = Path(f"{output_path}.npy")
    metadata_file_path = Path(f"{output_path}_meta.json")

    # ------------------------------------------------------------------
    # 1. Build the full initial-condition grid
    # ------------------------------------------------------------------
    initial_conditions = make_grid(grid_size, theta_range[0], theta_range[1])

    # ------------------------------------------------------------------
    # 2. Create (or reopen) the memmap output file
    # ------------------------------------------------------------------
    if memmap_file_path.exists():
        # Resume mode: reopen the existing file for read/write
        flip_times_memmap: np.memmap = np.memmap(
            str(memmap_file_path),
            dtype=np.float64,
            mode="r+",
            shape=grid_shape,
        )
        print(f"Resuming: reopened existing memmap at {memmap_file_path}")
    else:
        flip_times_memmap = create_memmap_output(
            str(memmap_file_path), grid_shape, dtype=np.float64
        )
        # Fill with NaN so un-simulated voxels are distinguishable
        flip_times_memmap[:] = np.nan
        flip_times_memmap.flush()
        print(f"Created new memmap at {memmap_file_path}")

    # ------------------------------------------------------------------
    # 3. Process the grid in chunks
    # ------------------------------------------------------------------
    num_chunks = int(np.ceil(total_pendulums / chunk_size))
    start_time = time.monotonic()

    print(
        f"Memmap batch: {total_pendulums} pendulums ({grid_size}^3), "
        f"{num_chunks} chunks of up to {chunk_size}, "
        f"backend={'GPU' if use_gpu else 'CPU'}"
    )

    for chunk_index in range(num_chunks):
        chunk_start = chunk_index * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_pendulums)

        # --- Resume check: skip chunks whose memmap values are all non-NaN ---
        flat_view = flip_times_memmap.ravel()
        chunk_slice = flat_view[chunk_start:chunk_end]
        if not np.any(np.isnan(chunk_slice)):
            already_flipped = int(np.sum(np.isfinite(chunk_slice)))
            print(
                f"  Chunk {chunk_index + 1}/{num_chunks}: "
                f"already complete ({already_flipped} values present), skipping"
            )
            continue

        # --- Simulate this chunk ---
        chunk_thetas = initial_conditions[chunk_start:chunk_end]

        if use_gpu:
            chunk_results = simulate_batch_gpu(
                chunk_thetas,
                dt=dt,
                t_max=t_max,
                chunk_size=chunk_size,
                device=gpu_device,
            )
        else:
            chunk_results = simulate_batch(
                chunk_thetas,
                dt=dt,
                t_max=t_max,
            )

        chunk_flip_times = chunk_results["flip_times"]

        # --- Write results into the memmap at the correct flat indices ---
        flat_view = flip_times_memmap.ravel()
        flat_view[chunk_start:chunk_end] = chunk_flip_times
        flip_times_memmap.flush()

        # --- Progress reporting ---
        cumulative_finite = int(np.sum(np.isfinite(flip_times_memmap)))
        cumulative_fraction = cumulative_finite / total_pendulums
        print(
            f"  Chunk {chunk_index + 1}/{num_chunks} done — "
            f"cumulative flipped: {cumulative_fraction:.1%}"
        )

    # ------------------------------------------------------------------
    # 4. Save metadata sidecar
    # ------------------------------------------------------------------
    elapsed_seconds = time.monotonic() - start_time
    total_flipped = int(np.sum(np.isfinite(flip_times_memmap)))

    metadata = {
        "grid_size": grid_size,
        "theta_range": list(theta_range),
        "dt": dt,
        "t_max": t_max,
        "total_flipped": total_flipped,
        "total_pendulums": total_pendulums,
        "fraction_flipped": total_flipped / total_pendulums,
        "computation_time_seconds": round(elapsed_seconds, 2),
        "use_gpu": use_gpu,
        "gpu_device": gpu_device if use_gpu else None,
        "chunk_size": chunk_size,
    }

    metadata_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(
        f"Memmap batch complete: {total_flipped}/{total_pendulums} flipped "
        f"({elapsed_seconds:.1f}s). Output: {output_path}"
    )

    # ------------------------------------------------------------------
    # 5. Return the output path
    # ------------------------------------------------------------------
    return output_path
