"""CPU vs GPU performance benchmarks and GPU accuracy validation.

Benchmarks wall-clock time and memory usage for triple pendulum batch
simulation across a range of grid sizes.  When a CUDA-capable GPU is
available, also validates that the GPU solver produces flip times that
agree with the CPU reference within acceptable tolerances.

Issues: #14, #15
"""

import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.simulation.batch_sim import simulate_batch
from src.utils.grid import make_grid

try:
    import torch
except ImportError:
    torch = None

try:
    from src.simulation.batch_sim import simulate_batch_gpu
except ImportError:
    simulate_batch_gpu = None


# ---------------------------------------------------------------------------
# CPU benchmark
# ---------------------------------------------------------------------------


def benchmark_cpu(
    grid_sizes: list[int] | None = None,
    dt: float = 0.01,
    t_max: float = 15.0,
) -> list[dict[str, Any]]:
    """Benchmark CPU simulation across multiple grid sizes.

    For each grid size N, an N**3 grid of initial conditions is built and
    simulated using the fixed-step RK4 integrator.  Wall time and peak
    memory usage (via ``tracemalloc``) are recorded.

    Args:
        grid_sizes: List of per-axis grid resolutions to benchmark.
            Defaults to ``[10, 20, 40]``.
        dt: Integration timestep in seconds.
        t_max: Maximum simulation time in seconds.

    Returns:
        List of dicts, one per grid size, each containing:
            - ``grid_size``: int
            - ``total_pendulums``: int
            - ``wall_time_seconds``: float
            - ``peak_memory_bytes``: int
            - ``fraction_flipped``: float
            - ``dt``: float
            - ``t_max``: float
    """
    if grid_sizes is None:
        grid_sizes = [10, 20, 40]

    benchmark_results: list[dict[str, Any]] = []

    for grid_size in grid_sizes:
        total_pendulums = grid_size ** 3
        initial_conditions = make_grid(grid_size)

        print(f"\n{'='*60}")
        print(f"CPU benchmark: grid_size={grid_size} ({total_pendulums} pendulums)")
        print(f"{'='*60}")

        tracemalloc.start()

        wall_start = time.perf_counter()
        simulation_results = simulate_batch(initial_conditions, dt=dt, t_max=t_max)
        wall_elapsed = time.perf_counter() - wall_start

        _current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        flip_times = simulation_results["flip_times"]
        fraction_flipped = float(np.sum(~np.isnan(flip_times)) / total_pendulums)

        result_entry = {
            "grid_size": grid_size,
            "total_pendulums": total_pendulums,
            "wall_time_seconds": round(wall_elapsed, 3),
            "peak_memory_bytes": peak_memory,
            "fraction_flipped": round(fraction_flipped, 4),
            "dt": dt,
            "t_max": t_max,
        }
        benchmark_results.append(result_entry)

        print(
            f"  Wall time: {wall_elapsed:.2f}s | "
            f"Peak memory: {peak_memory / (1024**2):.1f} MB | "
            f"Flipped: {fraction_flipped:.1%}"
        )

    return benchmark_results


# ---------------------------------------------------------------------------
# GPU benchmark
# ---------------------------------------------------------------------------


def benchmark_gpu(
    grid_sizes: list[int] | None = None,
    dt: float = 0.01,
    t_max: float = 15.0,
    device: str = "cuda",
) -> list[dict[str, Any]] | None:
    """Benchmark GPU simulation across multiple grid sizes.

    Uses ``simulate_batch_gpu`` with the dopri5 adaptive solver via
    torchdiffeq.  GPU peak memory is tracked with
    ``torch.cuda.max_memory_allocated`` when running on a CUDA device.

    Args:
        grid_sizes: List of per-axis grid resolutions to benchmark.
            Defaults to ``[10, 20, 40]``.
        dt: Output time-point spacing in seconds.
        t_max: Maximum simulation time in seconds.
        device: Torch device string (e.g. ``"cuda"``, ``"cuda:0"``).

    Returns:
        List of dicts (same schema as :func:`benchmark_cpu` plus a
        ``gpu_peak_memory_bytes`` key), or ``None`` if CUDA is not
        available.
    """
    if grid_sizes is None:
        grid_sizes = [10, 20, 40]

    if torch is None:
        print("GPU benchmark skipped: PyTorch is not installed.")
        return None

    if simulate_batch_gpu is None:
        print("GPU benchmark skipped: simulate_batch_gpu could not be imported.")
        return None

    is_cuda_device = device.startswith("cuda")

    if is_cuda_device and not torch.cuda.is_available():
        print("GPU benchmark skipped: CUDA is not available.")
        return None

    benchmark_results: list[dict[str, Any]] = []

    for grid_size in grid_sizes:
        total_pendulums = grid_size ** 3
        initial_conditions = make_grid(grid_size)

        print(f"\n{'='*60}")
        print(
            f"GPU benchmark: grid_size={grid_size} "
            f"({total_pendulums} pendulums, device={device})"
        )
        print(f"{'='*60}")

        try:
            # Reset GPU memory stats before each run
            if is_cuda_device and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            wall_start = time.perf_counter()
            simulation_results = simulate_batch_gpu(
                initial_conditions, dt=dt, t_max=t_max, device=device,
            )

            if is_cuda_device and torch.cuda.is_available():
                torch.cuda.synchronize(device)

            wall_elapsed = time.perf_counter() - wall_start

            # Record GPU memory
            gpu_peak_memory = 0
            if is_cuda_device and torch.cuda.is_available():
                gpu_peak_memory = torch.cuda.max_memory_allocated(device)

            flip_times = simulation_results["flip_times"]
            fraction_flipped = float(
                np.sum(~np.isnan(flip_times)) / total_pendulums
            )

            result_entry = {
                "grid_size": grid_size,
                "total_pendulums": total_pendulums,
                "wall_time_seconds": round(wall_elapsed, 3),
                "gpu_peak_memory_bytes": gpu_peak_memory,
                "fraction_flipped": round(fraction_flipped, 4),
                "dt": dt,
                "t_max": t_max,
                "device": device,
            }
            benchmark_results.append(result_entry)

            print(
                f"  Wall time: {wall_elapsed:.2f}s | "
                f"GPU peak memory: {gpu_peak_memory / (1024**2):.1f} MB | "
                f"Flipped: {fraction_flipped:.1%}"
            )

        except Exception as gpu_error:
            print(f"  ERROR for grid_size={grid_size}: {gpu_error}")
            result_entry = {
                "grid_size": grid_size,
                "total_pendulums": total_pendulums,
                "wall_time_seconds": None,
                "gpu_peak_memory_bytes": None,
                "fraction_flipped": None,
                "dt": dt,
                "t_max": t_max,
                "device": device,
                "error": str(gpu_error),
            }
            benchmark_results.append(result_entry)

    return benchmark_results


# ---------------------------------------------------------------------------
# GPU accuracy validation
# ---------------------------------------------------------------------------


def validate_gpu_accuracy(
    grid_size: int = 20,
    dt: float = 0.01,
    t_max: float = 10.0,
) -> dict[str, Any] | None:
    """Compare CPU and GPU flip times to quantify GPU solver accuracy.

    Both backends simulate the same initial-condition grid. The resulting
    flip-time arrays are compared element-wise to produce error statistics
    and an agreement percentage.

    Args:
        grid_size: Per-axis grid resolution (total = grid_size**3).
        dt: Timestep (CPU) or output spacing (GPU) in seconds.
        t_max: Maximum simulation time in seconds.

    Returns:
        Dict with validation metrics, or ``None`` if GPU is not available.
        Keys:
            - ``max_error``: maximum absolute flip-time error (seconds)
            - ``mean_error``: mean absolute flip-time error (seconds)
            - ``agreement_pct``: percentage of pendulums where both
              backends agree on flip/no-flip status
            - ``cpu_flipped``: count of pendulums that flipped (CPU)
            - ``gpu_flipped``: count of pendulums that flipped (GPU)
            - ``both_flipped``: count of pendulums that flipped on both
            - ``error_histogram``: 10-bin histogram of absolute errors
              for pendulums that flipped on both backends
            - ``boundary_disagreements``: count of pendulums near flip
              boundaries where CPU and GPU disagree
    """
    if torch is None:
        print("GPU validation skipped: PyTorch is not installed.")
        return None

    if simulate_batch_gpu is None:
        print("GPU validation skipped: simulate_batch_gpu could not be imported.")
        return None

    if not torch.cuda.is_available():
        print("GPU validation skipped: CUDA is not available.")
        return None

    total_pendulums = grid_size ** 3
    initial_conditions = make_grid(grid_size)

    print(f"\n{'='*60}")
    print(
        f"GPU accuracy validation: grid_size={grid_size} "
        f"({total_pendulums} pendulums)"
    )
    print(f"{'='*60}")

    # --- CPU reference run ---
    print("\nRunning CPU reference simulation...")
    cpu_results = simulate_batch(initial_conditions, dt=dt, t_max=t_max)
    cpu_flip_times: NDArray[np.float64] = cpu_results["flip_times"]

    # --- GPU run ---
    print("\nRunning GPU simulation...")
    try:
        gpu_results = simulate_batch_gpu(initial_conditions, dt=dt, t_max=t_max)
    except Exception as gpu_error:
        print(f"GPU simulation failed: {gpu_error}")
        return None
    gpu_flip_times: NDArray[np.float64] = gpu_results["flip_times"]

    # --- Classify flip status ---
    cpu_flipped_mask = ~np.isnan(cpu_flip_times)
    gpu_flipped_mask = ~np.isnan(gpu_flip_times)

    cpu_flipped_count = int(np.sum(cpu_flipped_mask))
    gpu_flipped_count = int(np.sum(gpu_flipped_mask))
    both_flipped_mask = cpu_flipped_mask & gpu_flipped_mask
    both_flipped_count = int(np.sum(both_flipped_mask))

    # Agreement: both flipped or both did not flip
    agree_mask = cpu_flipped_mask == gpu_flipped_mask
    agreement_percentage = float(np.mean(agree_mask) * 100.0)

    # --- Error statistics for pendulums that flipped on both ---
    if both_flipped_count > 0:
        absolute_errors = np.abs(
            cpu_flip_times[both_flipped_mask] - gpu_flip_times[both_flipped_mask]
        )
        maximum_error = float(np.max(absolute_errors))
        mean_error = float(np.mean(absolute_errors))

        # 10-bin histogram of absolute errors
        histogram_counts, histogram_bin_edges = np.histogram(
            absolute_errors, bins=10
        )
        error_histogram = {
            "counts": histogram_counts.tolist(),
            "bin_edges": histogram_bin_edges.tolist(),
        }
    else:
        maximum_error = 0.0
        mean_error = 0.0
        error_histogram = {"counts": [], "bin_edges": []}

    # --- Identify boundary disagreements ---
    # Pendulums where one backend flipped and the other did not,
    # AND the flipped side's flip time is in the last 20% of t_max
    # (near the detection boundary).
    boundary_threshold = 0.8 * t_max
    disagree_mask = ~agree_mask

    boundary_disagreement_count = 0
    if np.any(disagree_mask):
        disagree_indices = np.where(disagree_mask)[0]
        for pendulum_index in disagree_indices:
            cpu_time = cpu_flip_times[pendulum_index]
            gpu_time = gpu_flip_times[pendulum_index]
            # Whichever side flipped, check if it was near the boundary
            flipped_time = cpu_time if not np.isnan(cpu_time) else gpu_time
            if not np.isnan(flipped_time) and flipped_time > boundary_threshold:
                boundary_disagreement_count += 1

    # --- Print summary ---
    print(f"\n  CPU flipped:  {cpu_flipped_count}/{total_pendulums}")
    print(f"  GPU flipped:  {gpu_flipped_count}/{total_pendulums}")
    print(f"  Both flipped: {both_flipped_count}/{total_pendulums}")
    print(f"  Agreement:    {agreement_percentage:.2f}%")
    if both_flipped_count > 0:
        print(f"  Max error:    {maximum_error:.6f}s")
        print(f"  Mean error:   {mean_error:.6f}s")
    print(f"  Boundary disagreements: {boundary_disagreement_count}")

    validation_results = {
        "max_error": maximum_error,
        "mean_error": mean_error,
        "agreement_pct": round(agreement_percentage, 4),
        "cpu_flipped": cpu_flipped_count,
        "gpu_flipped": gpu_flipped_count,
        "both_flipped": both_flipped_count,
        "error_histogram": error_histogram,
        "boundary_disagreements": boundary_disagreement_count,
        "grid_size": grid_size,
        "total_pendulums": total_pendulums,
        "dt": dt,
        "t_max": t_max,
    }

    return validation_results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def create_benchmark_report(
    cpu_results: list[dict[str, Any]],
    gpu_results: list[dict[str, Any]] | None,
    validation: dict[str, Any] | None,
    output_path: str = "renders/benchmark.png",
) -> None:
    """Create a visual benchmark report and print a text summary.

    Generates a matplotlib figure with a bar chart comparing CPU and GPU
    wall times across grid sizes, plus a table of GPU accuracy validation
    metrics.  Falls back to text-only output if matplotlib is not available.

    Args:
        cpu_results: List of CPU benchmark dicts from :func:`benchmark_cpu`.
        gpu_results: List of GPU benchmark dicts from :func:`benchmark_gpu`,
            or ``None`` if GPU was not benchmarked.
        validation: Validation dict from :func:`validate_gpu_accuracy`,
            or ``None`` if GPU validation was skipped.
        output_path: File path for the saved figure.
    """
    # ------------------------------------------------------------------
    # Text summary (always printed)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("BENCHMARK REPORT")
    print(f"{'='*70}")

    # CPU results table
    print(f"\n{'--- CPU Results ---':^70}")
    header_format = f"  {'Grid':>6s}  {'N':>8s}  {'Time (s)':>10s}  {'Memory (MB)':>12s}  {'Flipped':>8s}"
    print(header_format)
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*8}")
    for entry in cpu_results:
        memory_megabytes = entry["peak_memory_bytes"] / (1024 ** 2)
        print(
            f"  {entry['grid_size']:>6d}  "
            f"{entry['total_pendulums']:>8d}  "
            f"{entry['wall_time_seconds']:>10.2f}  "
            f"{memory_megabytes:>12.1f}  "
            f"{entry['fraction_flipped']:>7.1%}"
        )

    # GPU results table
    if gpu_results is not None:
        print(f"\n{'--- GPU Results ---':^70}")
        gpu_header = (
            f"  {'Grid':>6s}  {'N':>8s}  {'Time (s)':>10s}  "
            f"{'VRAM (MB)':>10s}  {'Flipped':>8s}  {'Speedup':>8s}"
        )
        print(gpu_header)
        print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

        # Build a lookup of CPU times by grid_size for speedup calculation
        cpu_time_by_grid = {
            entry["grid_size"]: entry["wall_time_seconds"]
            for entry in cpu_results
        }

        for entry in gpu_results:
            if entry.get("wall_time_seconds") is None:
                print(
                    f"  {entry['grid_size']:>6d}  "
                    f"{entry['total_pendulums']:>8d}  "
                    f"{'FAILED':>10s}  {'--':>10s}  {'--':>8s}  {'--':>8s}"
                )
                continue

            gpu_memory_megabytes = (
                entry.get("gpu_peak_memory_bytes", 0) or 0
            ) / (1024 ** 2)

            cpu_time = cpu_time_by_grid.get(entry["grid_size"])
            if cpu_time is not None and entry["wall_time_seconds"] > 0:
                speedup = cpu_time / entry["wall_time_seconds"]
                speedup_string = f"{speedup:.1f}x"
            else:
                speedup_string = "--"

            print(
                f"  {entry['grid_size']:>6d}  "
                f"{entry['total_pendulums']:>8d}  "
                f"{entry['wall_time_seconds']:>10.2f}  "
                f"{gpu_memory_megabytes:>10.1f}  "
                f"{entry['fraction_flipped']:>7.1%}  "
                f"{speedup_string:>8s}"
            )

    # Validation summary
    if validation is not None:
        print(f"\n{'--- GPU Accuracy Validation ---':^70}")
        print(f"  Grid size:        {validation['grid_size']}")
        print(f"  Total pendulums:  {validation['total_pendulums']}")
        print(f"  CPU flipped:      {validation['cpu_flipped']}")
        print(f"  GPU flipped:      {validation['gpu_flipped']}")
        print(f"  Both flipped:     {validation['both_flipped']}")
        print(f"  Agreement:        {validation['agreement_pct']:.2f}%")
        print(f"  Max error:        {validation['max_error']:.6f}s")
        print(f"  Mean error:       {validation['mean_error']:.6f}s")
        print(f"  Boundary issues:  {validation['boundary_disagreements']}")

    # ------------------------------------------------------------------
    # Matplotlib figure (optional)
    # ------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nmatplotlib not available — skipping figure generation.")
        return

    has_gpu_data = gpu_results is not None and any(
        entry.get("wall_time_seconds") is not None for entry in gpu_results
    )
    has_validation = validation is not None

    # Determine subplot layout
    num_subplots = 1 + (1 if has_validation else 0)
    figure, axes = plt.subplots(
        1, num_subplots, figsize=(6 * num_subplots, 5),
    )
    if num_subplots == 1:
        axes = [axes]

    # --- Bar chart: wall time vs grid size ---
    axis_timing = axes[0]
    cpu_grid_sizes = [entry["grid_size"] for entry in cpu_results]
    cpu_wall_times = [entry["wall_time_seconds"] for entry in cpu_results]

    bar_width = 0.35
    x_positions = np.arange(len(cpu_grid_sizes))

    axis_timing.bar(
        x_positions - bar_width / 2,
        cpu_wall_times,
        bar_width,
        label="CPU (RK4)",
        color="#4C72B0",
    )

    if has_gpu_data:
        # Match GPU results to CPU grid sizes
        gpu_time_by_grid = {}
        for entry in gpu_results:
            if entry.get("wall_time_seconds") is not None:
                gpu_time_by_grid[entry["grid_size"]] = entry["wall_time_seconds"]

        gpu_wall_times = [
            gpu_time_by_grid.get(grid_size, 0) for grid_size in cpu_grid_sizes
        ]
        axis_timing.bar(
            x_positions + bar_width / 2,
            gpu_wall_times,
            bar_width,
            label="GPU (dopri5)",
            color="#DD8452",
        )

    axis_timing.set_xlabel("Grid size (per axis)")
    axis_timing.set_ylabel("Wall time (seconds)")
    axis_timing.set_title("CPU vs GPU Simulation Time")
    axis_timing.set_xticks(x_positions)
    axis_timing.set_xticklabels(
        [f"{grid_size}\n({grid_size**3})" for grid_size in cpu_grid_sizes]
    )
    axis_timing.legend()
    axis_timing.set_yscale("log")
    axis_timing.grid(axis="y", alpha=0.3)

    # --- Validation metrics table ---
    if has_validation:
        axis_table = axes[1]
        axis_table.axis("off")
        axis_table.set_title("GPU Accuracy Validation", pad=20)

        table_data = [
            ["CPU flipped", str(validation["cpu_flipped"])],
            ["GPU flipped", str(validation["gpu_flipped"])],
            ["Both flipped", str(validation["both_flipped"])],
            ["Agreement", f"{validation['agreement_pct']:.2f}%"],
            ["Max error", f"{validation['max_error']:.6f}s"],
            ["Mean error", f"{validation['mean_error']:.6f}s"],
            ["Boundary issues", str(validation["boundary_disagreements"])],
        ]

        table_widget = axis_table.table(
            cellText=table_data,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="left",
        )
        table_widget.auto_set_font_size(False)
        table_widget.set_fontsize(10)
        table_widget.scale(1.0, 1.5)

        # Style header row
        for column_index in range(2):
            header_cell = table_widget[0, column_index]
            header_cell.set_facecolor("#4C72B0")
            header_cell.set_text_props(color="white", fontweight="bold")

    figure.tight_layout()

    output_directory = Path(output_path).parent
    output_directory.mkdir(parents=True, exist_ok=True)

    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"\nBenchmark figure saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Triple Pendulum Performance Benchmark Suite")
    print(f"Python {sys.version}")
    print()

    # --- Configuration ---
    benchmark_grid_sizes = [10, 20, 40]
    benchmark_dt = 0.01
    benchmark_t_max = 15.0
    validation_grid_size = 20
    validation_t_max = 10.0
    report_output_path = "renders/benchmark.png"

    # --- Run CPU benchmarks ---
    cpu_benchmark_results = benchmark_cpu(
        grid_sizes=benchmark_grid_sizes,
        dt=benchmark_dt,
        t_max=benchmark_t_max,
    )

    # --- Run GPU benchmarks ---
    gpu_benchmark_results = benchmark_gpu(
        grid_sizes=benchmark_grid_sizes,
        dt=benchmark_dt,
        t_max=benchmark_t_max,
    )

    # --- Run GPU accuracy validation ---
    gpu_validation_results = validate_gpu_accuracy(
        grid_size=validation_grid_size,
        dt=benchmark_dt,
        t_max=validation_t_max,
    )

    # --- Generate report ---
    create_benchmark_report(
        cpu_results=cpu_benchmark_results,
        gpu_results=gpu_benchmark_results,
        validation=gpu_validation_results,
        output_path=report_output_path,
    )
