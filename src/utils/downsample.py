"""Downsample large memmap simulation results for the web viewer.

For resolutions that use memmap output (>= 700^3), the raw data is too large
for the Three.js viewer.  This module provides nearest-neighbor downsampling
from an arbitrary source resolution to a target resolution (default 200^3),
producing a JSON file that the viewer can load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.io import save_results_json


def downsample_memmap_to_json(
    memmap_path: str | Path,
    source_resolution: int,
    target_resolution: int = 200,
    theta_min: float = -170.0,
    theta_max: float = 170.0,
    output_path: str | Path | None = None,
) -> Path:
    """Nearest-neighbor downsample from source memmap to target resolution JSON.

    Maps each target voxel (i, j, k) to the nearest source voxel using index
    scaling, reads only those values from the memmap (no full load needed).

    Parameters
    ----------
    memmap_path : str or Path
        Path to the source ``.npy`` memmap file of shape ``(source_resolution^3,)``.
    source_resolution : int
        Number of points per axis in the source data.
    target_resolution : int
        Number of points per axis in the output (default 200).
    theta_min : float
        Lower bound of the angle range in degrees.
    theta_max : float
        Upper bound of the angle range in degrees.
    output_path : str or Path, optional
        Output JSON path. Defaults to
        ``data/simulation_{source_resolution}_ds{target_resolution}_gpu.json``.

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    if output_path is None:
        output_path = Path(
            f"data/simulation_{source_resolution}_ds{target_resolution}_gpu.json"
        )
    output_path = Path(output_path)

    source_total = source_resolution ** 3
    source_data = np.memmap(
        str(memmap_path), dtype=np.float64, mode="r", shape=(source_total,)
    )

    target_total = target_resolution ** 3
    scale = source_resolution / target_resolution
    target_n = target_resolution

    # Build target flat indices -> source flat indices via nearest-neighbor
    target_flat = np.arange(target_total)
    target_i = target_flat // (target_n * target_n)
    target_j = (target_flat // target_n) % target_n
    target_k = target_flat % target_n

    # Map target grid indices to source grid indices
    source_i = np.clip(np.round(target_i * scale).astype(np.intp), 0, source_resolution - 1)
    source_j = np.clip(np.round(target_j * scale).astype(np.intp), 0, source_resolution - 1)
    source_k = np.clip(np.round(target_k * scale).astype(np.intp), 0, source_resolution - 1)

    source_flat_indices = (
        source_i * source_resolution * source_resolution
        + source_j * source_resolution
        + source_k
    )

    downsampled_flip_times = np.array(source_data[source_flat_indices])

    num_flipped = int(np.sum(~np.isnan(downsampled_flip_times)))

    metadata = {
        "source_resolution": source_resolution,
        "target_resolution": target_resolution,
        "downsampled": True,
        "source_file": str(memmap_path),
        "num_flipped": num_flipped,
        "fraction_flipped": num_flipped / target_total,
    }

    save_results_json(
        output_path,
        grid_size=target_resolution,
        theta_range=(theta_min, theta_max),
        flip_times=downsampled_flip_times,
        metadata=metadata,
    )

    return output_path
