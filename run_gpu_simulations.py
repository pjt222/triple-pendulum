#!/usr/bin/env python3
"""Run GPU simulations for all target resolutions.

Iterates through resolutions [20, 30, 40, ..., 200, 300, ..., 1000], runs the
CUDA kernel for each, and saves results as JSON to data/. Optionally copies
results to a Hugging Face Space checkout via --hf-space-dir.

Supports two realms:
* **cube** (default) -- uniform Cartesian grid, N^3 total points.
* **sphere** -- Fibonacci-spiral shells, ~(pi/6)*N^3 total points.

Small resolutions (<=600^3) use single-launch simulation and direct JSON output.
Large resolutions (>=700^3) use chunked simulation with memmap output, then
downsample to 200^3 JSON for the web viewer.

Usage:
    python3 run_gpu_simulations.py
    python3 run_gpu_simulations.py --realm sphere --resolutions 20 30 40
    python3 run_gpu_simulations.py --resolutions 300 400 500
    python3 run_gpu_simulations.py --resolutions 700 800 900 1000
    python3 run_gpu_simulations.py --start-from 300
    python3 run_gpu_simulations.py --dry-run
"""

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.simulation.cuda_sim import HAS_CUPY, HAS_PYCUDA, HAS_TORCH
from src.utils.grid import make_grid, make_sphere_grid
from src.utils.io import save_results_json

ALL_RESOLUTIONS = list(range(20, 210, 10))  # [20, 30, 40, ..., 200]

# VRAM threshold: resolutions above this use chunked simulation
CHUNKED_THRESHOLD = 600


def get_backend():
    """Select the best available GPU backend."""
    if HAS_PYCUDA:
        from src.simulation.cuda_sim import simulate_batch_cuda
        return simulate_batch_cuda, "cuda"
    elif HAS_CUPY:
        from src.simulation.cuda_sim import simulate_batch_cupy
        return simulate_batch_cupy, "cupy"
    elif HAS_TORCH:
        from src.simulation.cuda_sim import simulate_batch_gpu_rk4
        return simulate_batch_gpu_rk4, "gpu_rk4"
    else:
        raise RuntimeError(
            "No GPU backend available. Install PyCUDA, CuPy, or PyTorch with CUDA."
        )


def format_count(number):
    """Format a large number with K/M/B suffix."""
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.1f}B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number / 1_000:.0f}K"
    return str(number)


def run_single_resolution(
    resolution, simulate_fn, data_directory, hf_data_directory, dt, t_max, log,
    realm="cube",
):
    """Run a single resolution using the single-launch path (<=600^3)."""
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

    per_resolution_log = str(data_directory / f"sim_cuda_{resolution}.log")

    result = simulate_fn(
        grid, dt=dt, t_max=t_max, logfile=per_resolution_log,
    )

    flip_times = result["flip_times"]
    metadata = result["metadata"]
    wall_time = metadata.get("wall_time_seconds", 0)
    fraction_flipped = metadata["fraction_flipped"]

    # Save JSON to data/
    if realm == "sphere":
        json_filename = f"simulation_{resolution}_sphere_gpu.json"
    else:
        json_filename = f"simulation_{resolution}_gpu.json"
    json_path = data_directory / json_filename

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

    file_size_mb = json_path.stat().st_size / (1024 * 1024)

    realm_label = f"Sphere {resolution}" if realm == "sphere" else f"{resolution}^3"
    log(f"{realm_label}: "
        f"{format_count(num_pendulums)} pendulums, "
        f"{wall_time:.2f}s, {fraction_flipped:.1%} flipped, "
        f"{file_size_mb:.1f}MB")
    log(f"Saved: {json_path}")

    # Copy to HF Space data directory if configured
    if hf_data_directory:
        hf_destination = hf_data_directory / json_filename
        shutil.copy2(json_path, hf_destination)
        log(f"Copied to HF: {hf_destination}")


def run_chunked_resolution(
    resolution, data_directory, hf_data_directory, dt, t_max, log
):
    """Run a single resolution using the chunked path (>=700^3)."""
    from src.simulation.cuda_sim import simulate_chunked_cupy
    from src.utils.downsample import downsample_memmap_to_json

    num_pendulums = resolution ** 3
    per_resolution_log = str(data_directory / f"sim_cuda_{resolution}.log")

    log(f"Using chunked simulation for {resolution}^3 "
        f"({format_count(num_pendulums)} pendulums)")

    flip_times, memmap_path = simulate_chunked_cupy(
        resolution,
        chunk_size=5_000_000,
        dt=dt,
        t_max=t_max,
        logfile=per_resolution_log,
    )

    num_flipped = int(np.sum(~np.isnan(flip_times)))
    fraction_flipped = num_flipped / num_pendulums

    log(f"Resolution {resolution}^3: {format_count(num_pendulums)} pendulums, "
        f"{fraction_flipped:.1%} flipped")
    log(f"Memmap saved: {memmap_path}")

    # Downsample to 200^3 for the web viewer
    target_resolution = 200
    ds_json_filename = (
        f"simulation_{resolution}_ds{target_resolution}_gpu.json"
    )
    ds_json_path = data_directory / ds_json_filename

    log(f"Downsampling {resolution}^3 -> {target_resolution}^3 for viewer...")
    downsample_memmap_to_json(
        memmap_path,
        source_resolution=resolution,
        target_resolution=target_resolution,
        output_path=ds_json_path,
    )

    file_size_mb = ds_json_path.stat().st_size / (1024 * 1024)
    log(f"Downsampled: {ds_json_path} ({file_size_mb:.1f}MB)")

    # Copy downsampled JSON to HF Space if configured
    if hf_data_directory:
        hf_destination = hf_data_directory / ds_json_filename
        shutil.copy2(ds_json_path, hf_destination)
        log(f"Copied to HF: {hf_destination}")


def main():
    parser = argparse.ArgumentParser(
        description="Run GPU simulations for all resolutions",
    )
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=None,
        help="Specific resolutions to run (default: all 20-200 by 10)",
    )
    parser.add_argument(
        "--start-from", type=int, default=None,
        help="Start from this resolution (skip lower)",
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
        "--realm", type=str, choices=["cube", "sphere"], default="cube",
        help="Grid type: 'cube' (default) or 'sphere' (Fibonacci shells)",
    )
    args = parser.parse_args()

    # Determine which resolutions to run
    if args.resolutions:
        resolutions = sorted(args.resolutions)
    elif args.start_from:
        resolutions = [r for r in ALL_RESOLUTIONS if r >= args.start_from]
    else:
        resolutions = ALL_RESOLUTIONS

    # Setup paths
    data_directory = Path("data")
    data_directory.mkdir(exist_ok=True)

    # HF Space directory for viewer data (replaces old docs/ copy)
    hf_data_directory = None
    if args.hf_space_dir:
        hf_data_directory = Path(args.hf_space_dir) / "data"
        hf_data_directory.mkdir(parents=True, exist_ok=True)

    # Master log
    master_log_path = data_directory / "sim_cuda_run.log"
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
            log(f"DRY RUN -- realm={realm}, "
                f"would run {len(resolutions)} resolutions: {resolutions}")
            for resolution in resolutions:
                if realm == "sphere":
                    _, _, sphere_meta = make_sphere_grid(resolution)
                    count = sphere_meta["total_points"]
                    label = f"  Sphere {resolution}"
                else:
                    count = resolution ** 3
                    label = f"  {resolution}^3"
                mode = "chunked" if realm == "cube" and resolution > CHUNKED_THRESHOLD else "single"
                ram_gb = count * 3 * 8 / (1024 ** 3)
                vram_gb = count * 32 / (1024 ** 3)
                log(f"{label} = {format_count(count)} pendulums "
                    f"[{mode}] RAM={ram_gb:.1f}GB VRAM={vram_gb:.1f}GB")
            total_pendulums = sum(
                make_sphere_grid(r)[2]["total_points"] if realm == "sphere" else r ** 3
                for r in resolutions
            )
            log(f"Total pendulums: {format_count(total_pendulums)}")
            return

        simulate_fn, backend_name = get_backend()
        log(f"Backend: {backend_name}")
        log(f"Realm: {realm}")
        log(f"Running {len(resolutions)} resolutions: {resolutions}")
        log(f"Parameters: dt={args.dt}, t_max={args.t_max}")
        if realm == "cube":
            log(f"Chunked threshold: >{CHUNKED_THRESHOLD}^3")

        total_run_start = time.monotonic()
        total_pendulums_simulated = 0

        for run_index, resolution in enumerate(resolutions):
            if realm == "sphere":
                # Estimate point count for logging (actual count computed in run_single_resolution).
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

            if realm == "cube" and resolution > CHUNKED_THRESHOLD:
                run_chunked_resolution(
                    resolution, data_directory, hf_data_directory,
                    args.dt, args.t_max, log,
                )
            else:
                run_single_resolution(
                    resolution, simulate_fn, data_directory, hf_data_directory,
                    args.dt, args.t_max, log,
                    realm=realm,
                )

        total_elapsed = time.monotonic() - total_run_start
        log("=" * 60)
        log(f"ALL DONE: {len(resolutions)} resolutions, "
            f"{format_count(total_pendulums_simulated)} total pendulums, "
            f"{total_elapsed:.1f}s total ({total_elapsed / 60:.1f}min)")

    finally:
        master_log.close()


if __name__ == "__main__":
    main()
