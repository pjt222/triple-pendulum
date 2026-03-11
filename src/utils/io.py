"""JSON data I/O for triple pendulum simulation results.

Handles serialisation of NumPy arrays (including NaN values) to a
portable JSON format and deserialisation back into NumPy arrays.
NaN values are written as JSON ``null`` and restored on load.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def save_results_json(
    path: str | Path,
    grid_size: int,
    theta_range: tuple[float, float],
    flip_times: npt.NDArray[np.float64],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save simulation results to a JSON file.

    Parameters
    ----------
    path : str or Path
        Destination file path.  Parent directories are created if needed.
    grid_size : int
        Number of sample points per axis (the "n" in an n**3 grid).
    theta_range : tuple of float
        ``(theta_min, theta_max)`` in degrees.
    flip_times : np.ndarray
        Flip-time values as a 1-D array of shape ``(n**3,)`` or a 3-D
        array of shape ``(n, n, n)``.  ``np.nan`` means the pendulum
        never flipped during the simulation window.
    metadata : dict, optional
        Extra metadata to store alongside the results.  Common keys:
        ``dt``, ``t_max``, ``computation_time_seconds``, ``date``.
        If *metadata* is provided but ``date`` is missing, the current
        UTC timestamp (ISO 8601) is added automatically.  Counts for
        ``total_points`` and ``total_flipped`` are always (re)computed
        from *flip_times*.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flat_flip_times = np.asarray(flip_times, dtype=np.float64).ravel()

    total_points = int(flat_flip_times.size)
    finite_mask = np.isfinite(flat_flip_times)
    total_flipped = int(finite_mask.sum())

    # Build metadata, filling in automatic fields.
    combined_metadata: dict[str, Any] = dict(metadata) if metadata else {}
    combined_metadata.setdefault("date", datetime.now(timezone.utc).isoformat())
    combined_metadata["total_points"] = total_points
    combined_metadata["total_flipped"] = total_flipped

    # Convert flip times: NaN -> None (becomes JSON null).
    serializable_flip_times: list[float | None] = [
        None if np.isnan(value) else float(value) for value in flat_flip_times
    ]

    document: dict[str, Any] = {
        "grid_size": grid_size,
        "theta_range": list(theta_range),
        "flip_times": serializable_flip_times,
        "metadata": combined_metadata,
    }

    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(document, file_handle)


def load_results_json(path: str | Path) -> dict[str, Any]:
    """Load simulation results from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to a JSON file previously written by :func:`save_results_json`.

    Returns
    -------
    dict
        Dictionary with keys ``"grid_size"``, ``"theta_range"``,
        ``"flip_times"`` (as a 1-D NumPy array with ``np.nan`` for
        entries that were ``null`` in JSON), and ``"metadata"``.
    """
    with open(path, encoding="utf-8") as file_handle:
        document: dict[str, Any] = json.load(file_handle)

    raw_flip_times: list[float | None] = document["flip_times"]
    restored_flip_times = np.array(
        [np.nan if value is None else value for value in raw_flip_times],
        dtype=np.float64,
    )

    return {
        "grid_size": document["grid_size"],
        "theta_range": document["theta_range"],
        "flip_times": restored_flip_times,
        "metadata": document.get("metadata", {}),
    }
