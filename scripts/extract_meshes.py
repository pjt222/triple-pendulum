#!/usr/bin/env python3
"""Extract isosurface OBJ meshes from simulation data for Blender rendering.

Usage:
    python3 scripts/extract_meshes.py [--resolution 150] [--levels 5]

Outputs OBJ files to renders/_meshes/ for the Blender render script.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualization.volume_render import extract_isosurface, save_mesh_obj


def load_binary_data(resolution: int, data_dir: Path) -> tuple[np.ndarray, dict]:
    """Load binary simulation data and metadata."""
    bin_path = data_dir / f"simulation_{resolution}_gpu.bin"
    meta_path = data_dir / f"simulation_{resolution}_gpu.bin.meta.json"

    if not bin_path.exists():
        raise FileNotFoundError(f"Data file not found: {bin_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    flip_times = np.fromfile(str(bin_path), dtype="<f4").astype(np.float64)
    grid_size = meta["grid_size"]
    flip_times_3d = flip_times.reshape((grid_size, grid_size, grid_size))

    return flip_times_3d, meta


def main():
    parser = argparse.ArgumentParser(description="Extract isosurface meshes")
    parser.add_argument("--resolution", type=int, default=150)
    parser.add_argument("--levels", type=int, default=5,
                        help="Number of isosurface levels to extract")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "renders" / "_meshes"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.resolution}^3 simulation data...")
    flip_times_3d, meta = load_binary_data(args.resolution, data_dir)
    theta_range = tuple(meta["theta_range"])
    t_max = meta["metadata"]["t_max"]

    # Compute isosurface levels spanning the flip-time range
    finite_times = flip_times_3d[np.isfinite(flip_times_3d)]
    t_min_data = float(np.percentile(finite_times, 5))
    t_max_data = float(np.percentile(finite_times, 95))

    levels = np.linspace(t_min_data, t_max_data, args.levels)
    print(f"Extracting {len(levels)} isosurfaces at levels: "
          f"{[f'{l:.2f}' for l in levels]}")

    # Also save level metadata for the Blender script
    level_info = []

    for i, level in enumerate(levels):
        try:
            mesh = extract_isosurface(flip_times_3d, level=level,
                                      theta_range=theta_range)
            obj_path = output_dir / f"iso_{i:02d}_{level:.2f}.obj"
            save_mesh_obj(str(obj_path), mesh["vertices"], mesh["faces"])
            normalized_time = (level - levels[0]) / (levels[-1] - levels[0])
            level_info.append({
                "index": i,
                "level": float(level),
                "normalized": float(normalized_time),
                "obj_file": obj_path.name,
                "num_vertices": int(mesh["vertices"].shape[0]),
                "num_faces": int(mesh["faces"].shape[0]),
            })
            print(f"  Level {level:.2f}s: {mesh['vertices'].shape[0]:,} vertices, "
                  f"{mesh['faces'].shape[0]:,} faces -> {obj_path.name}")
        except ValueError as e:
            print(f"  Level {level:.2f}s: skipped ({e})")

    # Save metadata for the Blender script
    info_path = output_dir / "levels.json"
    with open(info_path, "w") as f:
        json.dump({
            "resolution": args.resolution,
            "theta_range": list(theta_range),
            "t_max": t_max,
            "levels": level_info,
        }, f, indent=2)

    print(f"\n{len(level_info)} meshes saved to {output_dir}/")
    print(f"Level metadata saved to {info_path}")


if __name__ == "__main__":
    main()
