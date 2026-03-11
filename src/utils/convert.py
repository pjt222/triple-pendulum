"""Format conversion and downsampling for triple pendulum simulation data.

Provides bidirectional conversion between the three storage backends
(JSON, memmap/npy, HDF5) and nearest-neighbor downsampling that preserves
fractal boundary structure in flip-time grids.

Conversions
-----------
* :func:`json_to_memmap` -- JSON -> ``.npy`` + metadata sidecar
* :func:`memmap_to_json` -- ``.npy`` + metadata sidecar -> JSON
* :func:`json_to_hdf5` -- JSON -> gzip-compressed HDF5
* :func:`hdf5_to_json` -- HDF5 dataset -> JSON

Downsampling
------------
* :func:`downsample_grid` -- reduce an ``(n, n, n)`` grid to
  ``(target_n, target_n, target_n)`` using nearest-neighbor sampling
  so flip-time boundaries stay sharp.

CLI
---
Run as a module for quick one-off conversions::

    python -m src.utils.convert input.json output.npy
    python -m src.utils.convert input.h5 output.json --dataset flip_times
    python -m src.utils.convert input.json output.json --downsample 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.utils.io import (
    load_results_hdf5,
    load_results_json,
    load_results_memmap,
    save_results_hdf5,
    save_results_json,
    save_results_memmap,
)

# h5py is optional -- mirror the guarded import pattern from io.py.
try:
    import h5py  # type: ignore[import-untyped]
except ImportError:
    h5py = None


# ---------------------------------------------------------------------------
# JSON <-> Memmap
# ---------------------------------------------------------------------------


def json_to_memmap(
    json_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Convert a JSON simulation file to a NumPy memmap (``.npy`` + sidecar).

    The 1-D ``flip_times`` array stored in the JSON file is reshaped into
    a 3-D ``(n, n, n)`` cube (where *n* is the ``grid_size`` field) and
    saved via :func:`~src.utils.io.save_results_memmap`.

    Parameters
    ----------
    json_path : str or Path
        Path to a JSON file written by :func:`~src.utils.io.save_results_json`.
    output_path : str or Path
        Base path for the output files **without** extension.  Two files are
        created: ``{output_path}.npy`` and ``{output_path}_meta.json``.

    Returns
    -------
    Path
        The resolved *output_path* (without extension) so callers know where
        the files landed.
    """
    json_results = load_results_json(json_path)

    grid_size: int = json_results["grid_size"]
    flat_flip_times: npt.NDArray[np.float64] = json_results["flip_times"]
    stored_metadata: dict[str, Any] = json_results["metadata"]

    flip_times_3d = flat_flip_times.reshape(grid_size, grid_size, grid_size)

    # Carry forward the original JSON metadata so nothing is lost.
    combined_metadata: dict[str, Any] = dict(stored_metadata)
    combined_metadata["grid_size"] = grid_size
    combined_metadata["theta_range"] = json_results["theta_range"]

    resolved_output_path = Path(output_path)
    # Strip .npy if the caller accidentally included it, since
    # save_results_memmap appends the extension itself.
    if resolved_output_path.suffix == ".npy":
        resolved_output_path = resolved_output_path.with_suffix("")

    save_results_memmap(resolved_output_path, flip_times_3d, metadata=combined_metadata)

    return resolved_output_path


def memmap_to_json(
    memmap_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Convert a memmap (``.npy`` + sidecar) to a JSON simulation file.

    Parameters
    ----------
    memmap_path : str or Path
        Base path used when the memmap was saved (without extension).
    output_path : str or Path
        Destination JSON file path.

    Returns
    -------
    Path
        The resolved *output_path*.
    """
    memmap_results = load_results_memmap(memmap_path)

    flip_times_array: npt.NDArray[np.float64] = np.array(memmap_results["flip_times"])
    sidecar_metadata: dict[str, Any] = memmap_results["metadata"]

    # Determine grid_size from the array shape (first axis of the cube).
    grid_size: int = flip_times_array.shape[0]

    # Recover theta_range from the sidecar if available; fall back to a
    # sentinel so the JSON is still structurally valid.
    theta_range = sidecar_metadata.pop(
        "theta_range", [-170.0, 170.0]
    )

    # Remove fields that save_results_json recomputes automatically.
    sidecar_metadata.pop("grid_size", None)
    sidecar_metadata.pop("total_points", None)
    sidecar_metadata.pop("total_flipped", None)

    resolved_output_path = Path(output_path)

    save_results_json(
        resolved_output_path,
        grid_size=grid_size,
        theta_range=tuple(theta_range),
        flip_times=flip_times_array,
        metadata=sidecar_metadata,
    )

    return resolved_output_path


# ---------------------------------------------------------------------------
# JSON <-> HDF5
# ---------------------------------------------------------------------------


def _require_h5py() -> None:
    """Raise a clear error when h5py is not installed."""
    if h5py is None:
        raise ImportError(
            "h5py is required for HDF5 conversion but is not installed. "
            "Install it with:  pip install h5py"
        )


def json_to_hdf5(
    json_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Convert a JSON simulation file to a gzip-compressed HDF5 file.

    The flip-times array is reshaped to 3-D and stored as the
    ``"flip_times"`` dataset.  All JSON metadata is preserved as HDF5
    root-group attributes.

    Parameters
    ----------
    json_path : str or Path
        Path to a JSON file written by :func:`~src.utils.io.save_results_json`.
    output_path : str or Path
        Destination ``.h5`` file path.

    Returns
    -------
    Path
        The resolved *output_path*.
    """
    _require_h5py()

    json_results = load_results_json(json_path)

    grid_size: int = json_results["grid_size"]
    flat_flip_times: npt.NDArray[np.float64] = json_results["flip_times"]
    stored_metadata: dict[str, Any] = json_results["metadata"]

    flip_times_3d = flat_flip_times.reshape(grid_size, grid_size, grid_size)

    # Include grid parameters in the HDF5 attributes.
    hdf5_metadata: dict[str, Any] = dict(stored_metadata)
    hdf5_metadata["grid_size"] = grid_size
    hdf5_metadata["theta_range"] = json_results["theta_range"]

    resolved_output_path = Path(output_path)

    save_results_hdf5(
        resolved_output_path,
        datasets={"flip_times": flip_times_3d},
        metadata=hdf5_metadata,
    )

    return resolved_output_path


def hdf5_to_json(
    hdf5_path: str | Path,
    output_path: str | Path,
    dataset_name: str = "flip_times",
) -> Path:
    """Convert an HDF5 dataset to a JSON simulation file.

    Parameters
    ----------
    hdf5_path : str or Path
        Path to an ``.h5`` file written by
        :func:`~src.utils.io.save_results_hdf5`.
    output_path : str or Path
        Destination JSON file path.
    dataset_name : str, optional
        Name of the HDF5 dataset to extract (default ``"flip_times"``).

    Returns
    -------
    Path
        The resolved *output_path*.
    """
    _require_h5py()

    hdf5_results = load_results_hdf5(hdf5_path, dataset_names=[dataset_name])

    flip_times_array: npt.NDArray[np.float64] = hdf5_results[dataset_name]
    stored_metadata: dict[str, Any] = hdf5_results["metadata"]

    # Determine grid_size from the array (first axis of the cube).
    grid_size: int = flip_times_array.shape[0]

    # Recover theta_range from HDF5 attributes if available.
    theta_range = stored_metadata.pop("theta_range", [-170.0, 170.0])
    if isinstance(theta_range, np.ndarray):
        theta_range = theta_range.tolist()

    # Remove fields that save_results_json recomputes automatically.
    stored_metadata.pop("grid_size", None)
    stored_metadata.pop("total_points", None)
    stored_metadata.pop("total_flipped", None)

    resolved_output_path = Path(output_path)

    save_results_json(
        resolved_output_path,
        grid_size=grid_size,
        theta_range=tuple(theta_range),
        flip_times=flip_times_array,
        metadata=stored_metadata,
    )

    return resolved_output_path


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------


def downsample_grid(
    flip_times_3d: npt.NDArray[np.float64],
    target_n: int,
) -> npt.NDArray[np.float64]:
    """Downsample a 3-D flip-time grid using nearest-neighbor sampling.

    Nearest-neighbor (stride-based) indexing is used instead of
    interpolation so that the sharp fractal boundaries between chaotic
    and non-chaotic regions are preserved.  Interpolating flip-time
    values would blur those boundaries.

    Parameters
    ----------
    flip_times_3d : np.ndarray
        Source array of shape ``(n, n, n)``.
    target_n : int
        Desired number of points per axis in the output.  Must be less
        than or equal to the source size.

    Returns
    -------
    np.ndarray
        Downsampled array of shape ``(target_n, target_n, target_n)``.

    Raises
    ------
    ValueError
        If *target_n* is larger than the source grid or is not positive.
    """
    source_n = flip_times_3d.shape[0]

    if target_n <= 0:
        raise ValueError(
            f"target_n must be a positive integer, got {target_n}"
        )
    if target_n > source_n:
        raise ValueError(
            f"target_n ({target_n}) exceeds source grid size ({source_n}). "
            "Upsampling is not supported."
        )
    if target_n == source_n:
        return flip_times_3d.copy()

    # Compute evenly spaced indices into the source grid.  np.linspace
    # with integer rounding gives the best coverage of the original range.
    sample_indices = np.round(
        np.linspace(0, source_n - 1, target_n)
    ).astype(int)

    # Use np.ix_ to build an open mesh for fancy indexing along all three
    # axes simultaneously.
    downsampled_grid = flip_times_3d[np.ix_(sample_indices, sample_indices, sample_indices)]

    return np.asarray(downsampled_grid, dtype=flip_times_3d.dtype)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_EXTENSION_TO_FORMAT: dict[str, str] = {
    ".json": "json",
    ".npy": "memmap",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
}


def _detect_format(file_path: Path) -> str:
    """Return the format string for a file based on its extension.

    Raises
    ------
    ValueError
        If the extension is not recognised.
    """
    extension = file_path.suffix.lower()
    detected_format = _EXTENSION_TO_FORMAT.get(extension)
    if detected_format is None:
        supported_extensions = ", ".join(sorted(_EXTENSION_TO_FORMAT.keys()))
        raise ValueError(
            f"Unrecognised file extension '{extension}'. "
            f"Supported extensions: {supported_extensions}"
        )
    return detected_format


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    argument_parser = argparse.ArgumentParser(
        prog="python -m src.utils.convert",
        description=(
            "Convert triple-pendulum simulation data between formats "
            "(JSON, memmap/npy, HDF5) and optionally downsample the grid."
        ),
    )
    argument_parser.add_argument(
        "input",
        type=str,
        help="Input file path (.json, .npy, .h5)",
    )
    argument_parser.add_argument(
        "output",
        type=str,
        help="Output file path (.json, .npy, .h5)",
    )
    argument_parser.add_argument(
        "--dataset",
        type=str,
        default="flip_times",
        help="HDF5 dataset name to read (default: flip_times)",
    )
    argument_parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        metavar="N",
        help="Downsample to an (N, N, N) grid before writing output",
    )
    return argument_parser


def _load_as_3d(
    input_path: Path,
    input_format: str,
    dataset_name: str,
) -> tuple[npt.NDArray[np.float64], int, dict[str, Any]]:
    """Load from any supported format and return (3d_array, grid_size, metadata)."""
    if input_format == "json":
        json_results = load_results_json(input_path)
        grid_size = json_results["grid_size"]
        flip_times_3d = json_results["flip_times"].reshape(
            grid_size, grid_size, grid_size
        )
        metadata = json_results["metadata"]
        metadata["theta_range"] = json_results["theta_range"]
        return flip_times_3d, grid_size, metadata

    if input_format == "memmap":
        memmap_base_path = input_path.with_suffix("")
        memmap_results = load_results_memmap(memmap_base_path)
        flip_times_array = np.array(memmap_results["flip_times"])
        grid_size = flip_times_array.shape[0]
        metadata = memmap_results["metadata"]
        return flip_times_array, grid_size, metadata

    if input_format == "hdf5":
        _require_h5py()
        hdf5_results = load_results_hdf5(
            input_path, dataset_names=[dataset_name]
        )
        flip_times_array = hdf5_results[dataset_name]
        grid_size = flip_times_array.shape[0]
        metadata = hdf5_results["metadata"]
        return flip_times_array, grid_size, metadata

    raise ValueError(f"Unsupported input format: {input_format}")


def _save_from_3d(
    flip_times_3d: npt.NDArray[np.float64],
    grid_size: int,
    metadata: dict[str, Any],
    output_path: Path,
    output_format: str,
) -> None:
    """Save a 3-D array to any supported format."""
    if output_format == "json":
        theta_range = metadata.pop("theta_range", [-170.0, 170.0])
        if isinstance(theta_range, np.ndarray):
            theta_range = theta_range.tolist()
        # Remove fields that save_results_json recomputes.
        metadata.pop("grid_size", None)
        metadata.pop("total_points", None)
        metadata.pop("total_flipped", None)
        save_results_json(
            output_path,
            grid_size=grid_size,
            theta_range=tuple(theta_range),
            flip_times=flip_times_3d,
            metadata=metadata,
        )
        return

    if output_format == "memmap":
        memmap_base_path = output_path.with_suffix("")
        memmap_metadata = dict(metadata)
        memmap_metadata["grid_size"] = grid_size
        save_results_memmap(memmap_base_path, flip_times_3d, metadata=memmap_metadata)
        return

    if output_format == "hdf5":
        _require_h5py()
        hdf5_metadata = dict(metadata)
        hdf5_metadata["grid_size"] = grid_size
        save_results_hdf5(
            output_path,
            datasets={"flip_times": flip_times_3d},
            metadata=hdf5_metadata,
        )
        return

    raise ValueError(f"Unsupported output format: {output_format}")


def main(cli_args: list[str] | None = None) -> None:
    """Entry point for the CLI.

    Parameters
    ----------
    cli_args : list of str, optional
        Command-line arguments to parse.  ``None`` (default) reads from
        ``sys.argv``.
    """
    argument_parser = _build_argument_parser()
    parsed_args = argument_parser.parse_args(cli_args)

    input_path = Path(parsed_args.input)
    output_path = Path(parsed_args.output)

    input_format = _detect_format(input_path)
    output_format = _detect_format(output_path)

    flip_times_3d, grid_size, metadata = _load_as_3d(
        input_path, input_format, parsed_args.dataset
    )

    # Optional downsampling step.
    if parsed_args.downsample is not None:
        target_n: int = parsed_args.downsample
        original_n = grid_size
        flip_times_3d = downsample_grid(flip_times_3d, target_n)
        grid_size = target_n
        metadata["downsampled_from"] = original_n
        metadata["downsampled_to"] = target_n

    _save_from_3d(flip_times_3d, grid_size, metadata, output_path, output_format)

    print(
        f"Converted {input_path} ({input_format}) -> "
        f"{output_path} ({output_format}), "
        f"grid_size={grid_size}"
    )


if __name__ == "__main__":
    main()
