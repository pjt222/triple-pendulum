#!/usr/bin/env python3
"""Export sphere realm point cloud for Blender rendering.

Reconstructs Fibonacci spiral positions and pairs them with flip-time
colors for the Blender render script.

Usage:
    python3 scripts/prep_sphere.py [--resolution 150]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.grid import make_sphere_grid


def main():
    parser = argparse.ArgumentParser(description="Export sphere point cloud")
    parser.add_argument("--resolution", type=int, default=150)
    parser.add_argument("--max-points", type=int, default=200_000,
                        help="Subsample to this many points for render speed")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "renders" / "_meshes"
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution = args.resolution
    bin_path = data_dir / f"simulation_{resolution}_sphere_gpu.bin"
    meta_path = data_dir / f"simulation_{resolution}_sphere_gpu.bin.meta.json"

    if not bin_path.exists():
        print(f"ERROR: {bin_path} not found")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    print(f"Loading {resolution} sphere data...")
    flip_times = np.fromfile(str(bin_path), dtype="<f4").astype(np.float64)

    # Reconstruct sphere positions
    print("Reconstructing Fibonacci spiral positions...")
    _thetas, positions, _grid_meta = make_sphere_grid(
        resolution, r_max=meta["grid_params"]["r_max"]
    )

    total_points = len(flip_times)
    t_max = meta["metadata"]["t_max"]
    print(f"  {total_points:,} points, t_max={t_max}")

    # Filter: only keep points that flipped (finite flip times)
    finite_mask = np.isfinite(flip_times)
    flip_times_finite = flip_times[finite_mask]
    positions_finite = positions[finite_mask]
    print(f"  {len(flip_times_finite):,} flipped points ({100*len(flip_times_finite)/total_points:.1f}%)")

    # Subsample if too many points for Blender
    if len(flip_times_finite) > args.max_points:
        indices = np.random.default_rng(42).choice(
            len(flip_times_finite), args.max_points, replace=False
        )
        indices.sort()
        flip_times_finite = flip_times_finite[indices]
        positions_finite = positions_finite[indices]
        print(f"  Subsampled to {len(flip_times_finite):,} points")

    # Normalize flip times to [0, 1] for colormap
    normalized_times = np.clip(flip_times_finite / t_max, 0, 1)

    # Pack as interleaved (x, y, z, t) float32
    num_points = len(flip_times_finite)
    packed = np.empty((num_points, 4), dtype=np.float32)
    packed[:, :3] = positions_finite.astype(np.float32)
    packed[:, 3] = normalized_times.astype(np.float32)

    points_path = output_dir / "sphere_points.bin"
    packed.tofile(str(points_path))

    meta_out = {
        "num_points": num_points,
        "t_max": t_max,
        "resolution": resolution,
    }
    meta_out_path = output_dir / "sphere_meta.json"
    with open(meta_out_path, "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nExported {num_points:,} points to {points_path}")
    print(f"Metadata: {meta_out_path}")


if __name__ == "__main__":
    main()
