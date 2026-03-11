"""Data I/O for triple pendulum simulation results.

Supports three storage backends:

* **JSON** -- portable, human-readable; best for small grids (≤ 40³).
* **Memmap** -- NumPy memory-mapped files with a JSON metadata sidecar;
  ideal for large grids that do not fit in RAM.
* **HDF5** -- compressed, self-describing datasets via *h5py*; suited for
  archival storage of multiple arrays (flip times, Lyapunov exponents, …).

NaN values represent initial conditions where the pendulum never flipped
during the simulation window.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# h5py is an optional dependency -- the HDF5 helpers gracefully degrade
# to informative errors when it is not installed.
try:
    import h5py  # type: ignore[import-untyped]
except ImportError:
    h5py = None


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


# ---------------------------------------------------------------------------
# Memmap I/O  (Issue #30)
# ---------------------------------------------------------------------------


def save_results_memmap(
    path: str | Path,
    flip_times_3d: npt.NDArray[np.float64],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a 3-D flip-times array as a NumPy ``.npy`` file with a JSON sidecar.

    Two files are created:

    * ``{path}.npy``  -- the raw array, loadable as a memory-mapped file.
    * ``{path}_meta.json`` -- a JSON metadata sidecar.

    Parameters
    ----------
    path : str or Path
        Base path **without** extension (e.g. ``"data/sim_200"``).
    flip_times_3d : np.ndarray
        3-D array of shape ``(n, n, n)`` containing flip-time values.
    metadata : dict, optional
        Arbitrary metadata to store in the sidecar.  ``shape`` and ``dtype``
        are always added automatically so the array can be reconstructed
        from the sidecar alone.
    """
    base_path = Path(path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    array_path = base_path.with_suffix(".npy")
    sidecar_path = Path(f"{base_path}_meta.json")

    flip_times_array = np.asarray(flip_times_3d, dtype=np.float64)
    np.save(str(array_path), flip_times_array)

    sidecar_metadata: dict[str, Any] = dict(metadata) if metadata else {}
    sidecar_metadata["shape"] = list(flip_times_array.shape)
    sidecar_metadata["dtype"] = str(flip_times_array.dtype)

    with open(sidecar_path, "w", encoding="utf-8") as sidecar_file:
        json.dump(sidecar_metadata, sidecar_file)


def load_results_memmap(path: str | Path) -> dict[str, Any]:
    """Load a memory-mapped flip-times array and its metadata sidecar.

    Parameters
    ----------
    path : str or Path
        Base path used in :func:`save_results_memmap` (without extension).

    Returns
    -------
    dict
        ``"flip_times"`` is a **read-only** memory-mapped ``np.ndarray``.
        ``"metadata"`` contains the sidecar JSON contents.
    """
    base_path = Path(path)
    array_path = base_path.with_suffix(".npy")
    sidecar_path = Path(f"{base_path}_meta.json")

    flip_times_memmap: npt.NDArray[np.float64] = np.load(
        str(array_path), mmap_mode="r"
    )

    sidecar_metadata: dict[str, Any] = {}
    if sidecar_path.exists():
        with open(sidecar_path, encoding="utf-8") as sidecar_file:
            sidecar_metadata = json.load(sidecar_file)

    return {
        "flip_times": flip_times_memmap,
        "metadata": sidecar_metadata,
    }


def create_memmap_output(
    path: str | Path,
    shape: tuple[int, ...],
    dtype: npt.DTypeLike = np.float64,
) -> np.memmap:
    """Create an empty, writable memory-mapped array for incremental writes.

    This is useful when running a long simulation that fills the voxel grid
    slice-by-slice — results are flushed to disk continuously so progress
    is not lost if the process is interrupted.

    Parameters
    ----------
    path : str or Path
        Destination ``.npy``-compatible file path.
    shape : tuple of int
        Array dimensions, e.g. ``(200, 200, 200)``.
    dtype : dtype-like, optional
        Element type (default ``np.float64``).

    Returns
    -------
    np.memmap
        A writable memory-mapped array backed by *path*.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writable_memmap: np.memmap = np.memmap(
        str(output_path), dtype=dtype, mode="w+", shape=shape
    )
    return writable_memmap


# ---------------------------------------------------------------------------
# HDF5 I/O  (Issue #31)
# ---------------------------------------------------------------------------


def _require_h5py() -> None:
    """Raise a clear error when h5py is not installed."""
    if h5py is None:
        raise ImportError(
            "h5py is required for HDF5 I/O but is not installed. "
            "Install it with:  pip install h5py"
        )


def save_results_hdf5(
    path: str | Path,
    datasets: dict[str, npt.NDArray[Any]],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save one or more arrays to a gzip-compressed HDF5 file.

    Parameters
    ----------
    path : str or Path
        Destination ``.h5`` file path.
    datasets : dict
        Mapping of dataset names to NumPy arrays, e.g.
        ``{"flip_times": arr, "lyapunov": arr2}``.
    metadata : dict, optional
        Scalar metadata stored as HDF5 attributes on the root group.
        Values must be JSON-serialisable scalars (str, int, float, bool)
        or lists thereof.  Nested dicts are serialised to JSON strings so
        they survive the HDF5 attribute round-trip.
    """
    _require_h5py()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(output_path), "w") as hdf5_file:
        for dataset_name, array_data in datasets.items():
            hdf5_file.create_dataset(
                dataset_name,
                data=np.asarray(array_data),
                compression="gzip",
            )

        if metadata:
            for attribute_key, attribute_value in metadata.items():
                # h5py attributes do not support nested dicts — serialise
                # complex values as JSON strings.
                if isinstance(attribute_value, dict):
                    hdf5_file.attrs[attribute_key] = json.dumps(attribute_value)
                else:
                    hdf5_file.attrs[attribute_key] = attribute_value


def load_results_hdf5(
    path: str | Path,
    dataset_names: list[str] | None = None,
    lazy: bool = False,
) -> dict[str, Any]:
    """Load datasets and metadata from an HDF5 file.

    Parameters
    ----------
    path : str or Path
        Path to an ``.h5`` file written by :func:`save_results_hdf5`.
    dataset_names : list of str, optional
        Specific datasets to load.  ``None`` (default) loads every dataset
        in the file.
    lazy : bool, optional
        If ``True``, return the open :class:`h5py.File` object instead of
        reading arrays into memory.  The caller is responsible for closing
        the file.  Metadata is still read eagerly.

    Returns
    -------
    dict
        * When *lazy* is ``False`` (default): each requested dataset name
          maps to a NumPy array, plus a ``"metadata"`` key with a dict of
          the root-group HDF5 attributes.
        * When *lazy* is ``True``: a ``"file"`` key holds the open
          :class:`h5py.File`, and ``"metadata"`` holds the attributes dict.
    """
    _require_h5py()

    hdf5_file = h5py.File(str(path), "r")

    # Read metadata attributes eagerly in both modes.
    stored_metadata: dict[str, Any] = {}
    for attribute_key, attribute_value in hdf5_file.attrs.items():
        # Try to deserialise JSON strings that were dicts at save time.
        if isinstance(attribute_value, str):
            try:
                parsed_value = json.loads(attribute_value)
                if isinstance(parsed_value, dict):
                    attribute_value = parsed_value
            except (json.JSONDecodeError, ValueError):
                pass
        stored_metadata[attribute_key] = attribute_value

    if lazy:
        return {
            "file": hdf5_file,
            "metadata": stored_metadata,
        }

    # Eager mode: read requested datasets into memory, then close.
    names_to_load: list[str] = (
        dataset_names if dataset_names is not None else list(hdf5_file.keys())
    )

    loaded_results: dict[str, Any] = {"metadata": stored_metadata}
    for dataset_name in names_to_load:
        loaded_results[dataset_name] = np.array(hdf5_file[dataset_name])

    hdf5_file.close()
    return loaded_results
