#!/usr/bin/env python3
"""Run triple pendulum simulations across resolutions, realms, and backends.

Iterates through the canonical resolution grid, runs simulations using the
selected backend (GPU or CPU), and saves results as JSON + binary to data/.
Updates the data registry after each run.

Supports two realms:
* **cube** (default) -- uniform Cartesian grid, N^3 total points.
* **sphere** -- Fibonacci-spiral shells, ~(pi/6)*N^3 total points.

Supports two backends:
* **gpu** -- CuPy CUDA C kernel (default, ~170K pendulums/sec).
* **cpu** -- Numba JIT RK4 (~1.8K pendulums/sec), capped at --cpu-max-resolution.

Usage:
    python3 run_simulations.py --backend gpu --realm cube
    python3 run_simulations.py --backend gpu --realm sphere
    python3 run_simulations.py --backend cpu --realm cube
    python3 run_simulations.py --backend cpu --realm sphere
    python3 run_simulations.py --resolutions 10 20 30 --backend gpu --realm cube
    python3 run_simulations.py --dry-run
"""

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.utils.grid import make_grid, make_sphere_grid
from src.utils.io import save_results_json, save_results_binary
from src.utils.registry import RESOLUTION_GRID, write_registry

# Canonical resolution grid
ALL_RESOLUTIONS = RESOLUTION_GRID.copy()

# Default cap for CPU simulations (200^3 = 8M points takes ~1.2hrs on Numba)
DEFAULT_CPU_MAX_RESOLUTION = 200


def get_gpu_backend():
    """Select the best available GPU backend."""
    from src.simulation.cuda_sim import HAS_CUPY, HAS_PYCUDA, HAS_TORCH

    if HAS_CUPY:
        from src.simulation.cuda_sim import simulate_batch_cupy
        return simulate_batch_cupy, "cupy"
    elif HAS_PYCUDA:
        from src.simulation.cuda_sim import simulate_batch_cuda
        return simulate_batch_cuda, "pycuda"
    elif HAS_TORCH:
        from src.simulation.cuda_sim import simulate_batch_gpu_rk4
        return simulate_batch_gpu_rk4, "gpu_rk4"
    else:
        raise RuntimeError(
            "No GPU backend available. Install CuPy, PyCUDA, or PyTorch with CUDA."
        )


def get_cpu_backend():
    """Select the best available CPU backend."""
    from src.simulation.physics import HAS_NUMBA

    if HAS_NUMBA:
        from src.simulation.batch_sim import simulate_batch_fast
        return simulate_batch_fast, "numba"
    else:
        from src.simulation.batch_sim import simulate_batch
        return simulate_batch, "numpy"


def format_count(number):
    """Format a large number with K/M/B suffix."""
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.1f}B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number / 1_000:.0f}K"
    return str(number)


def make_filename(resolution, realm, backend_name):
    """Build the canonical filename stem for a simulation dataset.

    Naming convention:
        simulation_{N}_gpu           (cube GPU)
        simulation_{N}_sphere_gpu    (sphere GPU)
        simulation_{N}_cpu           (cube CPU)
        simulation_{N}_sphere_cpu    (sphere CPU)
    """
    backend_label = "gpu" if backend_name in ("cupy", "pycuda", "gpu_rk4") else "cpu"
    if realm == "sphere":
        return f"simulation_{resolution}_sphere_{backend_label}"
    return f"simulation_{resolution}_{backend_label}"


def run_single_resolution(
    resolution, simulate_fn, data_directory, hf_data_directory, dt, t_max, log,
    realm="cube", backend_name="cupy",
):
    """Run a single resolution and save JSON + binary output."""
    # Build initial conditions for the chosen realm.
    if realm == "sphere":
        grid, sphere_positions, sphere_meta = make_sphere_grid(resolution)
        num_pendulums = sphere_meta["total_points"]
        log(f"Sphere grid: {num_pendulums:,} points across "
            f"{sphere_meta['num_shells']} shells")
    else:
        grid = make_grid(resolution)
        num_pendulums = resolution ** 3
        sphere_positions = None
        sphere_meta = None

    per_resolution_log = str(data_directory / f"sim_{resolution}.log")

    result = simulate_fn(
        grid, dt=dt, t_max=t_max, logfile=per_resolution_log,
    )

    flip_times = result["flip_times"]
    metadata = result["metadata"]
    wall_time = metadata.get("wall_time_seconds", 0)
    fraction_flipped = metadata["fraction_flipped"]

    # Determine filenames
    file_stem = make_filename(resolution, realm, backend_name)
    json_filename = f"{file_stem}.json"
    bin_filename = f"{file_stem}.bin"
    json_path = data_directory / json_filename
    bin_path = data_directory / bin_filename

    # Save binary first (small, fast, never OOMs)
    save_results_binary(
        bin_path,
        grid_size=resolution,
        theta_range=(-170.0, 170.0),
        flip_times=flip_times,
        metadata=metadata,
        grid_type=realm,
        grid_params=sphere_meta,
    )

    bin_size_mb = bin_path.stat().st_size / (1024 * 1024)

    # Save JSON (can OOM for large grids due to Python list creation)
    json_size_mb = 0.0
    try:
        save_results_json(
            json_path,
            grid_size=resolution,
            theta_range=(-170.0, 170.0),
            flip_times=flip_times,
            metadata=metadata,
            grid_type=realm,
            positions=sphere_positions,
            grid_params=sphere_meta,
        )
        json_size_mb = json_path.stat().st_size / (1024 * 1024)
    except MemoryError:
        log(f"WARNING: JSON save skipped (MemoryError for {format_count(num_pendulums)} points)")

    realm_label = f"Sphere {resolution}" if realm == "sphere" else f"{resolution}^3"
    log(f"{realm_label}: "
        f"{format_count(num_pendulums)} pendulums, "
        f"{wall_time:.2f}s, {fraction_flipped:.1%} flipped, "
        f"BIN={bin_size_mb:.1f}MB"
        + (f", JSON={json_size_mb:.1f}MB" if json_size_mb > 0 else ""))
    log(f"Saved: {bin_path}" + (f", {json_path}" if json_size_mb > 0 else ""))

    # Copy to HF Space data directory if configured
    if hf_data_directory:
        for src_file in [json_path, bin_path, Path(f"{bin_path}.meta.json")]:
            if src_file.exists():
                dest = hf_data_directory / src_file.name
                shutil.copy2(src_file, dest)
        log(f"Copied to HF: {hf_data_directory}")


def main():
    parser = argparse.ArgumentParser(
        description="Run triple pendulum simulations",
    )
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=None,
        help="Specific resolutions to run (default: all in canonical grid)",
    )
    parser.add_argument(
        "--start-from", type=int, default=None,
        help="Start from this resolution (skip lower)",
    )
    parser.add_argument(
        "--backend", type=str, choices=["cpu", "gpu", "auto"], default="auto",
        help="Simulation backend: 'cpu' (Numba), 'gpu' (CuPy), or 'auto' (default)",
    )
    parser.add_argument(
        "--realm", type=str, choices=["cube", "sphere"], default="cube",
        help="Grid type: 'cube' (default) or 'sphere' (Fibonacci shells)",
    )
    parser.add_argument(
        "--cpu-max-resolution", type=int, default=DEFAULT_CPU_MAX_RESOLUTION,
        help=f"Max resolution for CPU backend (default: {DEFAULT_CPU_MAX_RESOLUTION})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be run without executing",
    )
    parser.add_argument(
        "--dt", type=float, default=0.01,
        help="Integration timestep (default: 0.01)",
    )
    parser.add_argument(
        "--t-max", type=float, default=15.0,
        help="Max simulation time (default: 15.0)",
    )
    parser.add_argument(
        "--hf-space-dir", type=str, default=None,
        help="Path to HF Space checkout; copies results to <dir>/data/",
    )
    parser.add_argument(
        "--no-registry", action="store_true",
        help="Skip updating the data registry after simulation",
    )
    args = parser.parse_args()

    # Determine which resolutions to run
    if args.resolutions:
        resolutions = sorted(args.resolutions)
    elif args.start_from:
        resolutions = [r for r in ALL_RESOLUTIONS if r >= args.start_from]
    else:
        resolutions = ALL_RESOLUTIONS[:]

    # Filter by CPU max resolution
    is_cpu = args.backend == "cpu"
    if is_cpu:
        original_count = len(resolutions)
        resolutions = [r for r in resolutions if r <= args.cpu_max_resolution]
        if len(resolutions) < original_count:
            skipped = original_count - len(resolutions)
            print(f"CPU mode: skipping {skipped} resolution(s) above "
                  f"{args.cpu_max_resolution}")

    # Setup paths
    data_directory = Path("data")
    data_directory.mkdir(exist_ok=True)

    hf_data_directory = None
    if args.hf_space_dir:
        hf_data_directory = Path(args.hf_space_dir) / "data"
        hf_data_directory.mkdir(parents=True, exist_ok=True)

    # Master log
    master_log_path = data_directory / "sim_run.log"
    master_log = open(master_log_path, "w")

    def log(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        master_log.write(line + "\n")
        master_log.flush()

    try:
        realm = args.realm

        if args.dry_run:
            log(f"DRY RUN -- realm={realm}, backend={args.backend}, "
                f"would run {len(resolutions)} resolutions: {resolutions}")
            for resolution in resolutions:
                if realm == "sphere":
                    _, _, sphere_meta = make_sphere_grid(resolution)
                    count = sphere_meta["total_points"]
                    label = f"  Sphere {resolution}"
                else:
                    count = resolution ** 3
                    label = f"  {resolution}^3"
                ram_gb = count * 3 * 8 / (1024 ** 3)
                vram_gb = count * 32 / (1024 ** 3)
                log(f"{label} = {format_count(count)} pendulums "
                    f"RAM={ram_gb:.1f}GB VRAM={vram_gb:.1f}GB")
            total_pendulums = sum(
                make_sphere_grid(r)[2]["total_points"] if realm == "sphere" else r ** 3
                for r in resolutions
            )
            log(f"Total pendulums: {format_count(total_pendulums)}")
            return

        # Select backend
        if args.backend == "gpu":
            simulate_fn, backend_name = get_gpu_backend()
        elif args.backend == "cpu":
            simulate_fn, backend_name = get_cpu_backend()
        else:  # auto
            try:
                simulate_fn, backend_name = get_gpu_backend()
            except RuntimeError:
                simulate_fn, backend_name = get_cpu_backend()

        log(f"Backend: {backend_name}")
        log(f"Realm: {realm}")
        log(f"Running {len(resolutions)} resolutions: {resolutions}")
        log(f"Parameters: dt={args.dt}, t_max={args.t_max}")

        total_run_start = time.monotonic()
        total_pendulums_simulated = 0

        for run_index, resolution in enumerate(resolutions):
            if realm == "sphere":
                estimated_points = int(np.round((np.pi / 6) * resolution ** 3))
                num_pendulums = estimated_points
            else:
                num_pendulums = resolution ** 3
            total_pendulums_simulated += num_pendulums

            log("=" * 60)
            realm_label = f"Sphere {resolution}" if realm == "sphere" else f"{resolution}^3"
            log(f"{realm_label}: "
                f"~{format_count(num_pendulums)} pendulums "
                f"[{run_index + 1}/{len(resolutions)}]")

            run_single_resolution(
                resolution, simulate_fn, data_directory, hf_data_directory,
                args.dt, args.t_max, log,
                realm=realm, backend_name=backend_name,
            )

        total_elapsed = time.monotonic() - total_run_start
        log("=" * 60)
        log(f"ALL DONE: {len(resolutions)} resolutions, "
            f"{format_count(total_pendulums_simulated)} total pendulums, "
            f"{total_elapsed:.1f}s total ({total_elapsed / 60:.1f}min)")

        # Update registry
        if not args.no_registry:
            registry_path = write_registry(data_directory)
            log(f"Registry updated: {registry_path}")

    finally:
        master_log.close()


if __name__ == "__main__":
    main()
