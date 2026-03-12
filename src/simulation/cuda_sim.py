"""CUDA C kernel for triple pendulum simulation.

Each CUDA thread simulates one pendulum with fixed-step RK4 integration,
analytical 3x3 Cramer's rule solver, and inline flip detection with early
exit. State lives entirely in GPU registers (6 doubles = 48 bytes per
thread), with only initial conditions and flip times in global memory.

Backend priority: PyCUDA > CuPy > PyTorch vectorized RK4.
"""

import math
import os
import time
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

# --- PyCUDA availability ---
try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule

    HAS_PYCUDA = True
except (ImportError, Exception):
    HAS_PYCUDA = False

# --- CuPy availability ---
try:
    import cupy as cp

    HAS_CUPY = cp.cuda.is_available()
except (ImportError, Exception):
    cp = None
    HAS_CUPY = False

# --- PyTorch availability (for fallback) ---
try:
    import torch

    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_TORCH = False


# ─── CUDA C Kernel Source ─────────────────────────────────────────────────────

CUDA_KERNEL_SOURCE = r"""
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef NAN
#define NAN __longlong_as_double(0x7FF8000000000000LL)
#endif

/* Wrap angle to [-pi, pi] — matches Python's (a + pi) % (2*pi) - pi */
__device__ double wrap_angle(double a) {
    double r = fmod(a + M_PI, 2.0 * M_PI);
    if (r < 0.0) r += 2.0 * M_PI;
    return r - M_PI;
}

/* Compute derivatives for a single triple pendulum.
   Exact port of _deriv_single from physics.py (Numba version).
   Mass matrix via coupling matrix A = [[3,2,1],[2,2,1],[1,1,1]],
   solved with Cramer's rule. */
__device__ void deriv_single(
    double t0, double t1, double t2,
    double w0, double w1, double w2,
    double *dt0, double *dt1, double *dt2,
    double *dw0, double *dw1, double *dw2
) {
    const double g = 9.81;

    /* Angle differences (3 unique pairs) */
    double d01 = t0 - t1;
    double d02 = t0 - t2;
    double d12 = t1 - t2;

    double cd01 = cos(d01);
    double cd02 = cos(d02);
    double cd12 = cos(d12);

    double sd01 = sin(d01);
    double sd02 = sin(d02);
    double sd12 = sin(d12);

    /* Mass matrix: M[i][j] = A[i][j] * cos(theta_i - theta_j)
       A = [[3,2,1],[2,2,1],[1,1,1]], diagonals: cos(0)=1 */
    double m00 = 3.0;
    double m01 = 2.0 * cd01;
    double m02 = cd02;
    double m10 = m01;  /* symmetric */
    double m11 = 2.0;
    double m12 = cd12;
    double m20 = m02;
    double m21 = m12;
    double m22 = 1.0;

    /* Force vector: f_i = sum_j A[i][j]*sin(ti-tj)*wj^2 + gw[i]*g*sin(ti)
       sin(ti-ti)=0 so diagonal coriolis terms vanish */
    double w0sq = w0 * w0;
    double w1sq = w1 * w1;
    double w2sq = w2 * w2;

    double f0 = 2.0 * sd01 * w1sq + sd02 * w2sq + 3.0 * g * sin(t0);
    double f1 = -2.0 * sd01 * w0sq + sd12 * w2sq + 2.0 * g * sin(t1);
    double f2 = -sd02 * w0sq - sd12 * w1sq + g * sin(t2);

    /* Solve M * alpha = -f via Cramer's rule */
    double b0 = -f0;
    double b1 = -f1;
    double b2 = -f2;

    double det = (m00 * (m11 * m22 - m12 * m21)
               - m01 * (m10 * m22 - m12 * m20)
               + m02 * (m10 * m21 - m11 * m20));
    double inv_det = 1.0 / det;

    double a0 = (b0 * (m11 * m22 - m12 * m21)
              - m01 * (b1 * m22 - m12 * b2)
              + m02 * (b1 * m21 - m11 * b2)) * inv_det;
    double a1 = (m00 * (b1 * m22 - m12 * b2)
              - b0 * (m10 * m22 - m12 * m20)
              + m02 * (m10 * b2 - b1 * m20)) * inv_det;
    double a2 = (m00 * (m11 * b2 - b1 * m21)
              - m01 * (m10 * b2 - b1 * m20)
              + b0 * (m10 * m21 - m11 * m20)) * inv_det;

    *dt0 = w0;
    *dt1 = w1;
    *dt2 = w2;
    *dw0 = a0;
    *dw1 = a1;
    *dw2 = a2;
}

/* Main kernel: one thread per pendulum.
   Fixed-step RK4, inline flip detection, early exit on flip. */
extern "C"
__global__ void simulate_pendulums(
    const double *initial_thetas_rad,  /* (N, 3) flattened, row-major */
    double *flip_times,                /* (N,) output */
    double dt,
    int num_steps,
    int num_pendulums
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pendulums) return;

    /* Load initial conditions (all start from rest: omega = 0) */
    double t0 = initial_thetas_rad[idx * 3 + 0];
    double t1 = initial_thetas_rad[idx * 3 + 1];
    double t2 = initial_thetas_rad[idx * 3 + 2];
    double w0 = 0.0, w1 = 0.0, w2 = 0.0;

    double flip_time = NAN;
    double dt_half = 0.5 * dt;
    double dt_sixth = dt / 6.0;

    for (int step = 0; step < num_steps; step++) {
        /* Save wrapped angles for flip detection */
        double prev_w0 = wrap_angle(t0);
        double prev_w1 = wrap_angle(t1);
        double prev_w2 = wrap_angle(t2);

        /* RK4 step — 4 calls to deriv_single, all in registers */
        double k1_0, k1_1, k1_2, k1_3, k1_4, k1_5;
        double k2_0, k2_1, k2_2, k2_3, k2_4, k2_5;
        double k3_0, k3_1, k3_2, k3_3, k3_4, k3_5;
        double k4_0, k4_1, k4_2, k4_3, k4_4, k4_5;

        /* k1 = f(state) */
        deriv_single(t0, t1, t2, w0, w1, w2,
                     &k1_0, &k1_1, &k1_2, &k1_3, &k1_4, &k1_5);

        /* k2 = f(state + 0.5*dt*k1) */
        deriv_single(
            t0 + dt_half * k1_0, t1 + dt_half * k1_1, t2 + dt_half * k1_2,
            w0 + dt_half * k1_3, w1 + dt_half * k1_4, w2 + dt_half * k1_5,
            &k2_0, &k2_1, &k2_2, &k2_3, &k2_4, &k2_5);

        /* k3 = f(state + 0.5*dt*k2) */
        deriv_single(
            t0 + dt_half * k2_0, t1 + dt_half * k2_1, t2 + dt_half * k2_2,
            w0 + dt_half * k2_3, w1 + dt_half * k2_4, w2 + dt_half * k2_5,
            &k3_0, &k3_1, &k3_2, &k3_3, &k3_4, &k3_5);

        /* k4 = f(state + dt*k3) */
        deriv_single(
            t0 + dt * k3_0, t1 + dt * k3_1, t2 + dt * k3_2,
            w0 + dt * k3_3, w1 + dt * k3_4, w2 + dt * k3_5,
            &k4_0, &k4_1, &k4_2, &k4_3, &k4_4, &k4_5);

        /* Update state: state += dt/6 * (k1 + 2*k2 + 2*k3 + k4) */
        t0 += dt_sixth * (k1_0 + 2.0 * k2_0 + 2.0 * k3_0 + k4_0);
        t1 += dt_sixth * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1);
        t2 += dt_sixth * (k1_2 + 2.0 * k2_2 + 2.0 * k3_2 + k4_2);
        w0 += dt_sixth * (k1_3 + 2.0 * k2_3 + 2.0 * k3_3 + k4_3);
        w1 += dt_sixth * (k1_4 + 2.0 * k2_4 + 2.0 * k3_4 + k4_4);
        w2 += dt_sixth * (k1_5 + 2.0 * k2_5 + 2.0 * k3_5 + k4_5);

        /* Flip detection: check if any bob's wrapped angle jumped > pi */
        double curr_w0 = wrap_angle(t0);
        double curr_w1 = wrap_angle(t1);
        double curr_w2 = wrap_angle(t2);

        if (fabs(curr_w0 - prev_w0) > M_PI ||
            fabs(curr_w1 - prev_w1) > M_PI ||
            fabs(curr_w2 - prev_w2) > M_PI) {
            flip_time = (step + 1) * dt;
            break;  /* Early exit for this thread */
        }
    }

    flip_times[idx] = flip_time;
}
"""

# Cache the compiled module to avoid recompilation
_compiled_module = None


def _get_kernel():
    """Compile the CUDA kernel (cached after first call)."""
    global _compiled_module
    if _compiled_module is None:
        _compiled_module = SourceModule(CUDA_KERNEL_SOURCE)
    return _compiled_module.get_function("simulate_pendulums")


def simulate_batch_cuda(
    initial_thetas: NDArray[np.float64],
    dt: float = 0.01,
    t_max: float = 15.0,
    logfile: str | None = None,
) -> dict:
    """Simulate N triple pendulums on GPU using a raw CUDA C kernel.

    Each CUDA thread simulates one pendulum with RK4 + Cramer's rule +
    inline flip detection. State lives in registers — only initial
    conditions and flip times touch global memory.

    Args:
        initial_thetas: Initial angles of shape (N, 3) in degrees.
        dt: Integration timestep in seconds.
        t_max: Maximum simulation time in seconds.
        logfile: Optional path to a progress logfile.

    Returns:
        Dictionary with "flip_times", "final_states" (zeros), and "metadata".
    """
    if not HAS_PYCUDA:
        raise ImportError(
            "PyCUDA is required for CUDA simulation. "
            "Install with: pip install pycuda nvidia-cuda-nvcc-cu12"
        )

    num_pendulums = initial_thetas.shape[0]
    num_steps = int(np.ceil(t_max / dt))

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
        # Convert degrees to radians
        initial_thetas_rad = np.radians(initial_thetas.astype(np.float64))
        thetas_flat = np.ascontiguousarray(initial_thetas_rad.ravel())
        flip_times = np.empty(num_pendulums, dtype=np.float64)

        # Kernel configuration
        threads_per_block = 256
        num_blocks = (num_pendulums + threads_per_block - 1) // threads_per_block

        input_mb = thetas_flat.nbytes / (1024 * 1024)
        output_mb = flip_times.nbytes / (1024 * 1024)

        log(f"[cuda] Simulating {num_pendulums} pendulums for {t_max}s "
            f"(dt={dt}, steps={num_steps})")
        log(f"[cuda] Kernel config: {num_blocks} blocks x {threads_per_block} threads")
        log(f"[cuda] VRAM: input {input_mb:.1f} MB + output {output_mb:.1f} MB "
            f"= {input_mb + output_mb:.1f} MB")

        # Compile kernel (cached after first call)
        log("[cuda] Compiling CUDA kernel (first call only)...")
        compile_start = time.monotonic()
        kernel = _get_kernel()
        compile_elapsed = time.monotonic() - compile_start
        log(f"[cuda] Kernel compilation done in {compile_elapsed:.2f}s")

        # Allocate GPU memory and transfer input
        thetas_gpu = drv.mem_alloc(thetas_flat.nbytes)
        flip_times_gpu = drv.mem_alloc(flip_times.nbytes)
        drv.memcpy_htod(thetas_gpu, thetas_flat)

        # Launch kernel
        log("[cuda] Launching kernel...")
        sim_start = time.monotonic()

        kernel(
            thetas_gpu,
            flip_times_gpu,
            np.float64(dt),
            np.int32(num_steps),
            np.int32(num_pendulums),
            block=(threads_per_block, 1, 1),
            grid=(num_blocks, 1),
        )

        # Synchronize and copy results back
        drv.Context.synchronize()
        sim_elapsed = time.monotonic() - sim_start

        drv.memcpy_dtoh(flip_times, flip_times_gpu)

        # Free GPU memory
        thetas_gpu.free()
        flip_times_gpu.free()

        num_flipped = int(np.sum(~np.isnan(flip_times)))
        throughput = num_pendulums / sim_elapsed if sim_elapsed > 0 else 0

        log(f"[cuda] Kernel complete in {sim_elapsed:.3f}s")
        log(f"[cuda] {num_flipped}/{num_pendulums} flipped "
            f"({num_flipped / num_pendulums:.1%})")
        log(f"[cuda] Throughput: {throughput:.0f} pendulums/s")

    finally:
        if log_fh:
            log_fh.close()

    simulation_results = {
        "flip_times": flip_times,
        "final_states": np.zeros((num_pendulums, 6), dtype=np.float64),
        "metadata": {
            "num_pendulums": num_pendulums,
            "dt": dt,
            "t_max": t_max,
            "actual_steps": num_steps,
            "t_final": num_steps * dt,
            "num_flipped": num_flipped,
            "fraction_flipped": num_flipped / num_pendulums,
            "backend": "cuda",
            "wall_time_seconds": round(sim_elapsed, 3),
            "throughput_per_sec": round(throughput),
        },
    }

    return simulation_results


# ─── CuPy CUDA C Kernel ──────────────────────────────────────────────────────

_cupy_kernel = None


def _get_cupy_kernel():
    """Compile the CUDA kernel via CuPy RawKernel (cached after first call)."""
    global _cupy_kernel
    if _cupy_kernel is None:
        _cupy_kernel = cp.RawKernel(CUDA_KERNEL_SOURCE, "simulate_pendulums")
    return _cupy_kernel


def simulate_batch_cupy(
    initial_thetas: NDArray[np.float64],
    dt: float = 0.01,
    t_max: float = 15.0,
    logfile: str | None = None,
) -> dict:
    """Simulate N triple pendulums on GPU using a CUDA C kernel via CuPy.

    Same kernel as the PyCUDA path but compiled through CuPy's RawKernel.
    CuPy is easier to install (pre-built wheels, no python3-dev needed).

    Args:
        initial_thetas: Initial angles of shape (N, 3) in degrees.
        dt: Integration timestep in seconds.
        t_max: Maximum simulation time in seconds.
        logfile: Optional path to a progress logfile.

    Returns:
        Dictionary with "flip_times", "final_states" (zeros), and "metadata".
    """
    if not HAS_CUPY:
        raise ImportError(
            "CuPy is required for this backend. "
            "Install with: pip install cupy-cuda12x"
        )

    num_pendulums = initial_thetas.shape[0]
    num_steps = int(np.ceil(t_max / dt))

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
        # Convert degrees to radians
        initial_thetas_rad = np.radians(initial_thetas.astype(np.float64))
        thetas_flat = np.ascontiguousarray(initial_thetas_rad.ravel())

        # Transfer to GPU via CuPy
        thetas_gpu = cp.asarray(thetas_flat)
        flip_times_gpu = cp.empty(num_pendulums, dtype=cp.float64)

        # Kernel configuration
        threads_per_block = 256
        num_blocks = (num_pendulums + threads_per_block - 1) // threads_per_block

        input_mb = thetas_flat.nbytes / (1024 * 1024)
        output_mb = num_pendulums * 8 / (1024 * 1024)

        log(f"[cupy] Simulating {num_pendulums} pendulums for {t_max}s "
            f"(dt={dt}, steps={num_steps})")
        log(f"[cupy] Kernel config: {num_blocks} blocks x "
            f"{threads_per_block} threads")
        log(f"[cupy] VRAM: input {input_mb:.1f} MB + output {output_mb:.1f} MB "
            f"= {input_mb + output_mb:.1f} MB")

        # Compile kernel (cached after first call)
        log("[cupy] Compiling CUDA kernel (first call only)...")
        compile_start = time.monotonic()
        kernel = _get_cupy_kernel()
        compile_elapsed = time.monotonic() - compile_start
        log(f"[cupy] Kernel compilation done in {compile_elapsed:.2f}s")

        # Launch kernel
        log("[cupy] Launching kernel...")
        sim_start = time.monotonic()

        kernel(
            (num_blocks,), (threads_per_block,),
            (thetas_gpu, flip_times_gpu,
             np.float64(dt), np.int32(num_steps), np.int32(num_pendulums)),
        )

        # Synchronize and copy results back
        cp.cuda.Device().synchronize()
        sim_elapsed = time.monotonic() - sim_start

        flip_times = cp.asnumpy(flip_times_gpu)

        num_flipped = int(np.sum(~np.isnan(flip_times)))
        throughput = num_pendulums / sim_elapsed if sim_elapsed > 0 else 0

        log(f"[cupy] Kernel complete in {sim_elapsed:.3f}s")
        log(f"[cupy] {num_flipped}/{num_pendulums} flipped "
            f"({num_flipped / num_pendulums:.1%})")
        log(f"[cupy] Throughput: {throughput:.0f} pendulums/s")

    finally:
        if log_fh:
            log_fh.close()

    simulation_results = {
        "flip_times": flip_times,
        "final_states": np.zeros((num_pendulums, 6), dtype=np.float64),
        "metadata": {
            "num_pendulums": num_pendulums,
            "dt": dt,
            "t_max": t_max,
            "actual_steps": num_steps,
            "t_final": num_steps * dt,
            "num_flipped": num_flipped,
            "fraction_flipped": num_flipped / num_pendulums,
            "backend": "cupy",
            "wall_time_seconds": round(sim_elapsed, 3),
            "throughput_per_sec": round(throughput),
        },
    }

    return simulation_results


# ─── Chunked CuPy Simulation ─────────────────────────────────────────────────


def simulate_chunked_cupy(
    grid_size: int,
    chunk_size: int = 5_000_000,
    dt: float = 0.01,
    t_max: float = 15.0,
    logfile: str | None = None,
) -> tuple[np.ndarray, str]:
    """Simulate grid_size^3 pendulums with streaming grid + chunked GPU launches.

    Uses :func:`make_grid_chunks` to avoid materializing the full grid in RAM,
    and launches the CUDA kernel in chunks that fit in VRAM.  Results are written
    to a NumPy memmap file with resume support — if interrupted, restarting will
    skip already-computed chunks.

    Args:
        grid_size: Number of points per axis (total = grid_size^3).
        chunk_size: Pendulums per GPU launch (default 5M ~ 160 MB VRAM).
        dt: Integration timestep in seconds.
        t_max: Maximum simulation time in seconds.
        logfile: Optional path to a progress logfile.

    Returns:
        Tuple of (flip_times_array, memmap_path) where flip_times_array is the
        full flat array of shape (grid_size^3,).
    """
    if not HAS_CUPY:
        raise ImportError(
            "CuPy is required for chunked simulation. "
            "Install with: pip install cupy-cuda12x"
        )

    from src.utils.grid import make_grid_chunks

    total = grid_size ** 3
    memmap_path = f"data/simulation_{grid_size}_gpu.npy"
    num_steps = int(np.ceil(t_max / dt))

    # ── Logging helper ──────────────────────────────────────────────
    log_fh = open(logfile, "a") if logfile else None

    def log(msg: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        if log_fh:
            log_fh.write(line + "\n")
            log_fh.flush()

    try:
        # Create or reopen memmap (NaN = not yet computed)
        if os.path.exists(memmap_path):
            flip_times = np.memmap(
                memmap_path, dtype=np.float64, mode="r+", shape=(total,)
            )
            completed_chunks = int(np.sum(~np.isnan(flip_times)))
            log(f"[chunked] Resuming {grid_size}^3: {completed_chunks}/{total} "
                f"already computed")
        else:
            flip_times = np.memmap(
                memmap_path, dtype=np.float64, mode="w+", shape=(total,)
            )
            flip_times[:] = np.nan
            flip_times.flush()
            log(f"[chunked] Created memmap: {memmap_path}")

        total_chunks = (total + chunk_size - 1) // chunk_size
        vram_per_chunk_mb = chunk_size * 32 / (1024 * 1024)

        log(f"[chunked] Grid {grid_size}^3 = {total:,} pendulums, "
            f"{total_chunks} chunks of {chunk_size:,}")
        log(f"[chunked] VRAM per chunk: ~{vram_per_chunk_mb:.0f} MB, "
            f"dt={dt}, t_max={t_max}, steps={num_steps}")

        kernel = _get_cupy_kernel()
        threads_per_block = 256
        overall_start = time.monotonic()
        pendulums_done = 0

        for chunk_idx, (chunk_thetas, start, end) in enumerate(
            make_grid_chunks(grid_size, chunk_size)
        ):
            # Skip already-computed chunks (resume support)
            chunk_slice = flip_times[start:end]
            if not np.any(np.isnan(chunk_slice)):
                pendulums_done += (end - start)
                log(f"[chunked] Chunk {chunk_idx + 1}/{total_chunks}: "
                    f"skipping (already computed)")
                continue

            chunk_count = end - start

            # Convert to radians and transfer to GPU
            thetas_rad = np.radians(chunk_thetas.astype(np.float64))
            thetas_flat = np.ascontiguousarray(thetas_rad.ravel())
            thetas_gpu = cp.asarray(thetas_flat)
            flip_times_gpu = cp.empty(chunk_count, dtype=cp.float64)

            num_blocks = (chunk_count + threads_per_block - 1) // threads_per_block

            chunk_start_time = time.monotonic()

            kernel(
                (num_blocks,), (threads_per_block,),
                (thetas_gpu, flip_times_gpu,
                 np.float64(dt), np.int32(num_steps), np.int32(chunk_count)),
            )

            cp.cuda.Device().synchronize()
            chunk_elapsed = time.monotonic() - chunk_start_time

            # Write results to memmap
            chunk_results = cp.asnumpy(flip_times_gpu)
            flip_times[start:end] = chunk_results
            flip_times.flush()

            pendulums_done += chunk_count
            overall_elapsed = time.monotonic() - overall_start
            throughput = pendulums_done / overall_elapsed if overall_elapsed > 0 else 0
            remaining = total - pendulums_done
            eta_seconds = remaining / throughput if throughput > 0 else 0

            num_flipped_chunk = int(np.sum(~np.isnan(chunk_results)))

            log(f"[chunked] Chunk {chunk_idx + 1}/{total_chunks}: "
                f"{chunk_count:,} pendulums in {chunk_elapsed:.1f}s, "
                f"{num_flipped_chunk}/{chunk_count} flipped, "
                f"ETA {eta_seconds / 60:.1f}min "
                f"({throughput:.0f} pend/s)")

        overall_elapsed = time.monotonic() - overall_start
        total_flipped = int(np.sum(~np.isnan(flip_times)))

        log(f"[chunked] DONE {grid_size}^3: {total:,} pendulums in "
            f"{overall_elapsed:.1f}s ({overall_elapsed / 60:.1f}min), "
            f"{total_flipped}/{total} flipped "
            f"({total_flipped / total:.1%})")

    finally:
        if log_fh:
            log_fh.close()

    return np.array(flip_times), memmap_path


# ─── PyTorch Vectorized RK4 Fallback ─────────────────────────────────────────


def _cramer_deriv_torch(theta, omega):
    """Compute derivatives for N pendulums using vectorized Cramer's rule.

    Mirrors _deriv_single but operates on batches of (N, 3) tensors.
    Returns (d_theta, d_omega) each of shape (N, 3).
    """
    gravity = 9.81

    theta_0, theta_1, theta_2 = theta[:, 0], theta[:, 1], theta[:, 2]
    omega_0, omega_1, omega_2 = omega[:, 0], omega[:, 1], omega[:, 2]

    diff_01 = theta_0 - theta_1
    diff_02 = theta_0 - theta_2
    diff_12 = theta_1 - theta_2

    cos_01 = torch.cos(diff_01)
    cos_02 = torch.cos(diff_02)
    cos_12 = torch.cos(diff_12)

    sin_01 = torch.sin(diff_01)
    sin_02 = torch.sin(diff_02)
    sin_12 = torch.sin(diff_12)

    # Mass matrix off-diagonal elements (diagonals are constants: 3, 2, 1)
    m01 = 2.0 * cos_01
    m02 = cos_02
    m12 = cos_12

    # Force vector
    omega_0_sq = omega_0 * omega_0
    omega_1_sq = omega_1 * omega_1
    omega_2_sq = omega_2 * omega_2

    force_0 = (2.0 * sin_01 * omega_1_sq + sin_02 * omega_2_sq
               + 3.0 * gravity * torch.sin(theta_0))
    force_1 = (-2.0 * sin_01 * omega_0_sq + sin_12 * omega_2_sq
               + 2.0 * gravity * torch.sin(theta_1))
    force_2 = (-sin_02 * omega_0_sq - sin_12 * omega_1_sq
               + gravity * torch.sin(theta_2))

    rhs_0 = -force_0
    rhs_1 = -force_1
    rhs_2 = -force_2

    # Cramer's rule with m00=3, m11=2, m22=1 (symmetric: m10=m01, m20=m02, m21=m12)
    determinant = (3.0 * (2.0 - m12 * m12)
                   - m01 * (m01 - m12 * m02)
                   + m02 * (m01 * m12 - 2.0 * m02))
    inverse_determinant = 1.0 / determinant

    accel_0 = (rhs_0 * (2.0 - m12 * m12)
               - m01 * (rhs_1 - m12 * rhs_2)
               + m02 * (rhs_1 * m12 - 2.0 * rhs_2)) * inverse_determinant
    accel_1 = (3.0 * (rhs_1 - m12 * rhs_2)
               - rhs_0 * (m01 - m12 * m02)
               + m02 * (m01 * rhs_2 - rhs_1 * m02)) * inverse_determinant
    accel_2 = (3.0 * (2.0 * rhs_2 - rhs_1 * m12)
               - m01 * (m01 * rhs_2 - rhs_1 * m02)
               + rhs_0 * (m01 * m12 - 2.0 * m02)) * inverse_determinant

    derivative_theta = omega
    derivative_omega = torch.stack([accel_0, accel_1, accel_2], dim=1)

    return derivative_theta, derivative_omega


def simulate_batch_gpu_rk4(
    initial_thetas: NDArray[np.float64],
    dt: float = 0.01,
    t_max: float = 15.0,
    device: str = "cuda",
    logfile: str | None = None,
) -> dict:
    """Simulate N triple pendulums using PyTorch vectorized RK4 on GPU.

    Fallback for when PyCUDA is not available. Uses fixed-step RK4 with
    Cramer's rule, inline flip detection, and early stopping.

    Args:
        initial_thetas: Initial angles of shape (N, 3) in degrees.
        dt: Integration timestep in seconds.
        t_max: Maximum simulation time in seconds.
        device: Torch device string.
        logfile: Optional path to a progress logfile.

    Returns:
        Dictionary with "flip_times", "final_states", and "metadata".
    """
    if torch is None or not torch.cuda.is_available():
        raise ImportError(
            "PyTorch with CUDA is required for GPU RK4 fallback. "
            "Install with: pip install torch"
        )

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
        torch_device = torch.device(device)

        # Convert degrees to radians and move to GPU
        theta = torch.from_numpy(
            np.radians(initial_thetas.astype(np.float64))
        ).to(torch_device)
        omega = torch.zeros_like(theta)

        # Flip tracking on GPU
        flip_times_tensor = torch.full(
            (num_pendulums,), float("nan"),
            dtype=torch.float64, device=torch_device,
        )
        has_flipped = torch.zeros(
            num_pendulums, dtype=torch.bool, device=torch_device,
        )

        vram_bytes = num_pendulums * (3 + 3 + 1) * 8
        vram_mb = vram_bytes / (1024 * 1024)
        log(f"[gpu_rk4] Simulating {num_pendulums} pendulums for {t_max}s "
            f"(dt={dt}, steps={num_steps})")
        log(f"[gpu_rk4] VRAM (state): ~{vram_mb:.1f} MB, device={device}")

        current_time = 0.0
        actual_steps_taken = 0
        sim_start = time.monotonic()
        dt_half = 0.5 * dt
        dt_sixth = dt / 6.0
        two_pi = 2.0 * math.pi

        with torch.no_grad():
            for step_index in range(num_steps):
                # Wrap prev angles for flip detection
                wrapped_prev = torch.remainder(theta + math.pi, two_pi) - math.pi

                # RK4 step
                k1_t, k1_w = _cramer_deriv_torch(theta, omega)

                k2_t, k2_w = _cramer_deriv_torch(
                    theta + dt_half * k1_t, omega + dt_half * k1_w)

                k3_t, k3_w = _cramer_deriv_torch(
                    theta + dt_half * k2_t, omega + dt_half * k2_w)

                k4_t, k4_w = _cramer_deriv_torch(
                    theta + dt * k3_t, omega + dt * k3_w)

                theta = theta + dt_sixth * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t)
                omega = omega + dt_sixth * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)

                current_time = (step_index + 1) * dt
                actual_steps_taken = step_index + 1

                # Flip detection
                wrapped_curr = torch.remainder(theta + math.pi, two_pi) - math.pi
                wrapped_delta = torch.abs(wrapped_curr - wrapped_prev)
                any_bob_flipped = torch.any(wrapped_delta > math.pi, dim=1)

                newly_flipped = any_bob_flipped & ~has_flipped
                flip_times_tensor[newly_flipped] = current_time
                has_flipped |= any_bob_flipped

                # Progress reporting
                if (step_index + 1) % progress_interval == 0:
                    flipped_frac = float(has_flipped.sum()) / num_pendulums
                    elapsed = time.monotonic() - sim_start
                    log(f"Progress: {100.0 * (step_index + 1) / num_steps:5.1f}% "
                        f"(t={current_time:.2f}s, flipped={flipped_frac:.1%}, "
                        f"{(step_index + 1) / elapsed:.0f} steps/s)")

                # Early stopping
                if torch.all(has_flipped):
                    log(f"Early stop at t={current_time:.2f}s: "
                        f"all {num_pendulums} pendulums flipped.")
                    break

        sim_elapsed = time.monotonic() - sim_start

        flip_times = flip_times_tensor.cpu().numpy()
        final_states = torch.cat([theta, omega], dim=1).cpu().numpy()

        num_flipped = int(np.sum(~np.isnan(flip_times)))
        throughput = num_pendulums / sim_elapsed if sim_elapsed > 0 else 0

        log(f"[gpu_rk4] Complete in {sim_elapsed:.2f}s: "
            f"{num_flipped}/{num_pendulums} flipped "
            f"({num_flipped / num_pendulums:.1%})")
        log(f"[gpu_rk4] Throughput: {throughput:.0f} pendulums/s")

    finally:
        if log_fh:
            log_fh.close()

    simulation_results = {
        "flip_times": flip_times,
        "final_states": final_states,
        "metadata": {
            "num_pendulums": num_pendulums,
            "dt": dt,
            "t_max": t_max,
            "actual_steps": actual_steps_taken,
            "t_final": current_time,
            "num_flipped": num_flipped,
            "fraction_flipped": num_flipped / num_pendulums,
            "backend": "gpu_rk4",
            "wall_time_seconds": round(sim_elapsed, 2),
            "throughput_per_sec": round(throughput),
        },
    }

    return simulation_results
