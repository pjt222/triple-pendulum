"""Data I/O for triple pendulum simulation results.

Supports four storage backends:

* **JSON** -- portable, human-readable; best for small grids (≤ 40³).
* **Binary** -- raw little-endian Float32 with a JSON metadata sidecar;
  designed for fast browser loading (zero-parse ``Float32Array`` wrapping).
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

# orjson is an optional fast-path for JSON serialization (~10x faster
# than stdlib json for large numeric arrays).
try:
    import orjson
except ImportError:
    orjson = None

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
    *,
    grid_type: str = "cube",
    positions: npt.NDArray[np.float64] | None = None,
    grid_params: dict[str, Any] | None = None,
) -> None:
    """Save simulation results to a JSON file.

    Parameters
    ----------
    path : str or Path
        Destination file path.  Parent directories are created if needed.
    grid_size : int
        Number of sample points per axis (the "n" in an n**3 grid).
        For sphere grids this is the number of radial shells.
    theta_range : tuple of float
        ``(theta_min, theta_max)`` in degrees.
    flip_times : np.ndarray
        Flip-time values as a 1-D array.  ``np.nan`` means the pendulum
        never flipped during the simulation window.
    metadata : dict, optional
        Extra metadata to store alongside the results.  Common keys:
        ``dt``, ``t_max``, ``computation_time_seconds``, ``date``.
        If *metadata* is provided but ``date`` is missing, the current
        UTC timestamp (ISO 8601) is added automatically.  Counts for
        ``total_points`` and ``total_flipped`` are always (re)computed
        from *flip_times*.
    grid_type : str
        Either ``"cube"`` (default) or ``"sphere"``.  When ``"sphere"``,
        additional fields (``grid_type``, ``grid_params``, ``positions``,
        ``total_points``) are written to the JSON document.
    positions : np.ndarray, optional
        Explicit viewer coordinates of shape ``(M, 3)``, normalised to
        ``[-1, 1]``.  Written as a flat interleaved array
        ``[x0, y0, z0, x1, y1, z1, ...]``.  Required for sphere grids.
    grid_params : dict, optional
        Sphere-specific parameters (``resolution``, ``r_max``,
        ``num_shells``, etc.).
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

    # Prepare positions list (sphere only).
    positions_list: list[float] | None = None
    if positions is not None:
        # Round to 4 decimal places to control JSON size while preserving
        # more than enough precision for point-cloud visualisation.
        rounded_positions = np.round(
            np.asarray(positions, dtype=np.float64).ravel(), decimals=4
        )
        positions_list = rounded_positions.tolist()

    # Convert flip times: NaN -> None (becomes JSON null).
    # orjson handles numpy arrays natively and is ~10x faster for large arrays.
    if orjson is not None:
        flip_list = flat_flip_times.tolist()
        nan_indices = np.flatnonzero(np.isnan(flat_flip_times))
        for idx in nan_indices:
            flip_list[idx] = None

        document: dict[str, Any] = {
            "grid_size": grid_size,
            "theta_range": list(theta_range),
            "flip_times": flip_list,
            "metadata": combined_metadata,
        }

        # Sphere-specific fields.
        if grid_type != "cube":
            document["grid_type"] = grid_type
            document["total_points"] = total_points
        if grid_params is not None:
            document["grid_params"] = grid_params
        if positions_list is not None:
            document["positions"] = positions_list

        with open(output_path, "wb") as file_handle:
            file_handle.write(orjson.dumps(document))
    else:
        serializable_flip_times: list[float | None] = [
            None if np.isnan(value) else float(value) for value in flat_flip_times
        ]

        document = {
            "grid_size": grid_size,
            "theta_range": list(theta_range),
            "flip_times": serializable_flip_times,
            "metadata": combined_metadata,
        }

        if grid_type != "cube":
            document["grid_type"] = grid_type
            document["total_points"] = total_points
        if grid_params is not None:
            document["grid_params"] = grid_params
        if positions_list is not None:
            document["positions"] = positions_list

        with open(output_path, "w", encoding="utf-8") as file_handle:
            json.dump(document, file_handle)


def save_results_binary(
    path: str | Path,
    grid_size: int,
    theta_range: tuple[float, float],
    flip_times: npt.NDArray[np.float64],
    metadata: dict[str, Any] | None = None,
    *,
    grid_type: str = "cube",
    grid_params: dict[str, Any] | None = None,
) -> None:
    """Save simulation results as a raw binary file with a JSON sidecar.

    Two files are created:

    * ``{path}``  -- raw little-endian Float32 array of flip-time values.
      ``NaN`` values are preserved natively via IEEE 754 representation.
    * ``{path}.meta.json`` -- JSON metadata sidecar containing grid
      parameters and simulation metadata.

    This format is designed for fast browser-side loading: the viewer can
    fetch the ``.bin`` file as an ``ArrayBuffer`` and wrap it directly in
    a ``Float32Array`` with zero parsing overhead.

    Parameters
    ----------
    path : str or Path
        Destination file path.  Should end in ``.bin``.  Parent directories
        are created if needed.  The metadata sidecar is written to
        ``{path}.meta.json``.
    grid_size : int
        Number of sample points per axis (the "n" in an n**3 grid).
        For sphere grids this is the number of radial shells.
    theta_range : tuple of float
        ``(theta_min, theta_max)`` in degrees.
    flip_times : np.ndarray
        Flip-time values as a 1-D array.  ``np.nan`` means the pendulum
        never flipped during the simulation window.
    metadata : dict, optional
        Extra metadata to store in the sidecar.  Common keys:
        ``dt``, ``t_max``, ``computation_time_seconds``, ``date``.
        If *metadata* is provided but ``date`` is missing, the current
        UTC timestamp (ISO 8601) is added automatically.  Counts for
        ``total_points`` and ``total_flipped`` are always (re)computed
        from *flip_times*.
    grid_type : str
        Either ``"cube"`` (default) or ``"sphere"``.  Stored in the
        sidecar so the viewer knows how to reconstruct positions.
    grid_params : dict, optional
        Grid-specific parameters (e.g. sphere ``resolution``, ``r_max``,
        ``num_shells``).  Stored in the sidecar.

    Notes
    -----
    The ``positions`` parameter accepted by :func:`save_results_json` is
    intentionally omitted here.  For sphere grids, the viewer reconstructs
    positions client-side from ``grid_params``, avoiding the need to
    transfer a large coordinate array.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sidecar_path = Path(f"{output_path}.meta.json")

    flat_flip_times = np.asarray(flip_times, dtype=np.float64).ravel()

    total_points = int(flat_flip_times.size)
    finite_mask = np.isfinite(flat_flip_times)
    total_flipped = int(finite_mask.sum())

    # Write the raw binary data as little-endian float32.
    # NaN values are preserved natively in IEEE 754 float32.
    flip_times_float32 = flat_flip_times.astype("<f4")
    flip_times_float32.tofile(str(output_path))

    # Build sidecar metadata.
    combined_metadata: dict[str, Any] = dict(metadata) if metadata else {}
    combined_metadata.setdefault("date", datetime.now(timezone.utc).isoformat())
    combined_metadata["total_points"] = total_points
    combined_metadata["total_flipped"] = total_flipped

    sidecar_document: dict[str, Any] = {
        "grid_size": grid_size,
        "theta_range": list(theta_range),
        "grid_type": grid_type,
        "metadata": combined_metadata,
    }

    if grid_params is not None:
        sidecar_document["grid_params"] = grid_params

    with open(sidecar_path, "w", encoding="utf-8") as sidecar_file:
        json.dump(sidecar_document, sidecar_file)


def load_results_binary(path: str | Path) -> dict[str, Any]:
    """Load simulation results from a binary file and its JSON sidecar.

    Parameters
    ----------
    path : str or Path
        Path to a ``.bin`` file written by :func:`save_results_binary`.
        The metadata sidecar is expected at ``{path}.meta.json``.

    Returns
    -------
    dict
        Dictionary with keys ``"grid_size"``, ``"theta_range"``,
        ``"flip_times"`` (as a 1-D NumPy float64 array with ``np.nan``
        preserved), ``"metadata"``, ``"grid_type"`` (``"cube"`` or
        ``"sphere"``), ``"positions"`` (always ``None`` -- binary format
        omits positions), and ``"grid_params"`` (dict or ``None``).
    """
    binary_path = Path(path)
    sidecar_path = Path(f"{binary_path}.meta.json")

    # Read raw little-endian float32 values and upcast to float64.
    flip_times_float32 = np.fromfile(str(binary_path), dtype="<f4")
    flip_times = flip_times_float32.astype(np.float64)

    # Read sidecar metadata.
    with open(sidecar_path, encoding="utf-8") as sidecar_file:
        sidecar_document: dict[str, Any] = json.load(sidecar_file)

    return {
        "grid_size": sidecar_document["grid_size"],
        "theta_range": sidecar_document["theta_range"],
        "flip_times": flip_times,
        "metadata": sidecar_document.get("metadata", {}),
        "grid_type": sidecar_document.get("grid_type", "cube"),
        "positions": None,
        "grid_params": sidecar_document.get("grid_params"),
    }


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
        entries that were ``null`` in JSON), ``"metadata"``,
        ``"grid_type"`` (``"cube"`` or ``"sphere"``), ``"positions"``
        (NumPy array of shape ``(M, 3)`` or ``None``), and
        ``"grid_params"`` (dict or ``None``).
    """
    with open(path, encoding="utf-8") as file_handle:
        document: dict[str, Any] = json.load(file_handle)

    raw_flip_times: list[float | None] = document["flip_times"]
    restored_flip_times = np.array(
        [np.nan if value is None else value for value in raw_flip_times],
        dtype=np.float64,
    )

    # Sphere-specific fields (backward-compatible: missing = cube).
    grid_type: str = document.get("grid_type", "cube")

    restored_positions: npt.NDArray[np.float64] | None = None
    if "positions" in document:
        raw_positions: list[float] = document["positions"]
        restored_positions = np.array(raw_positions, dtype=np.float64).reshape(-1, 3)

    grid_params: dict[str, Any] | None = document.get("grid_params")

    return {
        "grid_size": document["grid_size"],
        "theta_range": document["theta_range"],
        "flip_times": restored_flip_times,
        "metadata": document.get("metadata", {}),
        "grid_type": grid_type,
        "positions": restored_positions,
        "grid_params": grid_params,
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
