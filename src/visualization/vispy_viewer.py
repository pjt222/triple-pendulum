"""Interactive Vispy volumetric renderer for triple pendulum chaos data.

Renders the 3-D flip-time voxel grid as a volume with a magma-inspired
colormap matching :mod:`colormap`.  Supports orbit/zoom/pan via a
turntable camera, threshold filtering, boundary-only display, and
high-resolution screenshot capture.

Requires the ``vispy`` package (optional dependency).  Install with::

    pip install vispy

A compatible backend (PyQt5, PyQt6, PySide2, PySide6, or pyglet) must
also be available for the interactive event loop.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# Guard vispy import -- it is an optional dependency.
try:
    from vispy import app, scene  # type: ignore[import-untyped]
    from vispy.color import Colormap  # type: ignore[import-untyped]

    _HAS_VISPY = True
except ImportError:
    _HAS_VISPY = False

# Local imports for boundary detection and data loading.
from src.visualization.volume_render import extract_boundary_mask
from src.visualization.colormap import COLOR_STOPS
from src.utils.io import load_results_json, load_results_memmap


# ---------------------------------------------------------------------------
# Colormap construction
# ---------------------------------------------------------------------------

# RGBA stops matching the magma-inspired ramp defined in colormap.py.
# Each entry is (normalised position, R, G, B, A) with values in [0, 1].
# Alpha ramps from semi-transparent at the fast-flip end to opaque at the
# slow-flip end, giving depth cues in volumetric rendering.
_VOLUME_COLORMAP_STOPS: list[tuple[float, float, float, float, float]] = [
    # position,   R,     G,     B,     A
    (0.00, 0.10, 0.00, 0.20, 0.05),  # very dark indigo  (fastest flip)
    (0.15, 0.30, 0.00, 0.53, 0.20),  # deep purple
    (0.35, 0.72, 0.08, 0.30, 0.45),  # crimson-red
    (0.55, 0.93, 0.28, 0.08, 0.65),  # orange
    (0.78, 1.00, 0.78, 0.10, 0.85),  # warm yellow
    (1.00, 1.00, 1.00, 1.00, 1.00),  # white             (slowest flip / t_max)
]


def _build_vispy_colormap() -> "Colormap":
    """Build a Vispy :class:`~vispy.color.Colormap` from the chaos colour stops.

    The returned colormap maps normalised flip-time values in ``[0, 1]``
    to RGBA colours on the magma-inspired ramp.

    Returns
    -------
    vispy.color.Colormap
        A Vispy colormap instance suitable for use with ``Volume`` visuals.
    """
    # Vispy's Colormap accepts a list of RGBA control-point colours and an
    # optional list of normalised positions (``controls``).
    positions = [stop[0] for stop in _VOLUME_COLORMAP_STOPS]
    rgba_colors = [
        (stop[1], stop[2], stop[3], stop[4])
        for stop in _VOLUME_COLORMAP_STOPS
    ]

    colormap = Colormap(colors=rgba_colors, controls=positions, interpolation="linear")
    return colormap


# ---------------------------------------------------------------------------
# ChaosVolumeViewer
# ---------------------------------------------------------------------------


def _require_vispy() -> None:
    """Raise a clear error when vispy is not installed."""
    if not _HAS_VISPY:
        raise ImportError(
            "vispy is required for ChaosVolumeViewer but is not installed.  "
            "Install it with:  pip install vispy\n"
            "You also need a compatible GUI backend such as PyQt5:\n"
            "  pip install PyQt5"
        )


class ChaosVolumeViewer:
    """Interactive volumetric viewer for triple pendulum chaos data.

    Renders a 3-D flip-time array as a Vispy ``Volume`` visual with the
    magma-inspired colormap.  Provides a turntable camera for orbit,
    zoom, and pan, plus methods for threshold filtering, boundary-only
    display, and screenshot capture.

    Parameters
    ----------
    flip_times_3d : np.ndarray
        3-D array of shape ``(n, n, n)`` with flip-time values.
        ``np.nan`` marks voxels that never flipped.
    t_max : float
        Maximum simulation time used for normalisation (default 15.0).
        Flip times >= *t_max* are clamped to the top of the colour ramp.
    theta_range : tuple of float
        ``(theta_min, theta_max)`` in degrees (default ``(-170, 170)``).
        Used only for axis labelling / metadata -- the volume visual
        itself operates in voxel-index space.
    """

    def __init__(
        self,
        flip_times_3d: npt.NDArray[np.float64],
        t_max: float = 15.0,
        theta_range: tuple[float, float] = (-170.0, 170.0),
    ) -> None:
        _require_vispy()

        self.t_max = t_max
        self.theta_range = theta_range

        # Store the original data (with NaN) for later manipulation.
        self._original_data = np.array(flip_times_3d, dtype=np.float64, copy=True)

        # Prepare the volume data: replace NaN with 0 (mapped to
        # transparent by the colormap) and normalise to [0, 1].
        self._volume_data = self._normalise_volume(self._original_data)

        # Build the canvas and view.
        self._canvas = scene.SceneCanvas(
            keys="interactive",
            title="Triple Pendulum Chaos — Vispy Volume Viewer",
            size=(1280, 720),
            show=False,
        )
        self._view = self._canvas.central_widget.add_view()

        # Build the colormap.
        self._colormap = _build_vispy_colormap()

        # Add the Volume visual.
        self._volume_visual = scene.visuals.Volume(
            self._volume_data,
            cmap=self._colormap,
            method="translucent",
            relative_step_size=0.8,
            parent=self._view.scene,
        )

        # Centre the camera on the volume.
        grid_shape = self._volume_data.shape
        centre_position = (
            grid_shape[0] / 2.0,
            grid_shape[1] / 2.0,
            grid_shape[2] / 2.0,
        )

        self._view.camera = scene.TurntableCamera(
            center=centre_position,
            fov=45,
            elevation=30,
            azimuth=45,
            distance=max(grid_shape) * 1.8,
        )

    # ----- internal helpers ------------------------------------------------

    def _normalise_volume(
        self,
        flip_times_3d: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float32]:
        """Normalise flip-time data to ``[0, 1]`` for the volume visual.

        NaN values (never-flipped) are set to 0 so the colormap renders
        them as fully transparent.

        Parameters
        ----------
        flip_times_3d : np.ndarray
            Raw flip-time array (may contain NaN).

        Returns
        -------
        np.ndarray
            Float32 array in ``[0, 1]`` with the same shape.
        """
        normalised = np.copy(flip_times_3d)

        # Replace NaN with 0 (will be transparent).
        nan_mask = ~np.isfinite(normalised)
        normalised[nan_mask] = 0.0

        # Clamp and normalise finite values.
        normalised = np.clip(normalised, 0.0, self.t_max)
        normalised = normalised / self.t_max

        return normalised.astype(np.float32)

    def _update_volume(self, new_data: npt.NDArray[np.float32]) -> None:
        """Push new data to the volume visual.

        Parameters
        ----------
        new_data : np.ndarray
            Normalised float32 volume of shape ``(n, n, n)`` in ``[0, 1]``.
        """
        self._volume_visual.set_data(new_data)
        self._canvas.update()

    # ----- public API ------------------------------------------------------

    def set_threshold(self, min_t: float, max_t: float) -> None:
        """Filter visible voxels by flip-time range.

        Voxels with flip times outside ``[min_t, max_t]`` are set to 0
        (transparent).  The original data is preserved so thresholds can
        be changed repeatedly.

        Parameters
        ----------
        min_t : float
            Minimum flip time to display (seconds).
        max_t : float
            Maximum flip time to display (seconds).
        """
        filtered_data = np.copy(self._original_data)

        # Mask out-of-range values (including NaN, which fails all
        # comparisons).
        outside_range = ~(
            (filtered_data >= min_t) & (filtered_data <= max_t)
        )
        filtered_data[outside_range] = np.nan

        normalised_data = self._normalise_volume(filtered_data)
        self._update_volume(normalised_data)

    def set_opacity_scale(self, scale: float) -> None:
        """Adjust overall opacity of the volume rendering.

        This modifies the alpha channel of all colormap stops
        proportionally and rebuilds the volume's colormap.

        Parameters
        ----------
        scale : float
            Opacity multiplier in ``[0, 1]``.  ``1.0`` uses the default
            alpha values; ``0.5`` halves all alphas; ``0.0`` makes the
            volume fully transparent.
        """
        clamped_scale = float(np.clip(scale, 0.0, 1.0))

        scaled_rgba_colors = [
            (stop[1], stop[2], stop[3], stop[4] * clamped_scale)
            for stop in _VOLUME_COLORMAP_STOPS
        ]
        positions = [stop[0] for stop in _VOLUME_COLORMAP_STOPS]

        new_colormap = Colormap(
            colors=scaled_rgba_colors,
            controls=positions,
            interpolation="linear",
        )
        self._colormap = new_colormap
        self._volume_visual.cmap = new_colormap
        self._canvas.update()

    def show_boundary_only(self, threshold: float | None = None) -> None:
        """Display only boundary voxels identified by gradient analysis.

        Uses :func:`~src.visualization.volume_render.extract_boundary_mask`
        to compute a gradient-based boundary mask.  Non-boundary voxels
        are set to transparent.

        Parameters
        ----------
        threshold : float or None
            Gradient magnitude cutoff passed to
            :func:`extract_boundary_mask`.  ``None`` uses the adaptive
            default (mean + 1 * std of non-zero gradients).
        """
        boundary_mask = extract_boundary_mask(
            self._original_data,
            threshold=threshold,
            method="gradient",
        )

        boundary_data = np.copy(self._original_data)
        boundary_data[~boundary_mask] = np.nan

        normalised_data = self._normalise_volume(boundary_data)
        self._update_volume(normalised_data)

    def screenshot(
        self,
        path: str | Path,
        size: tuple[int, int] = (1920, 1080),
    ) -> None:
        """Capture a high-resolution screenshot of the current view.

        Parameters
        ----------
        path : str or Path
            Output file path (e.g. ``"renders/volume.png"``).  Parent
            directories are created if needed.
        size : tuple of int
            ``(width, height)`` in pixels for the output image
            (default ``(1920, 1080)``).
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Temporarily resize the canvas for the desired output resolution.
        original_size = self._canvas.size
        self._canvas.size = size
        self._canvas.show(visible=False)
        self._canvas.app.process_events()

        rendered_image = self._canvas.render()

        # Restore original size.
        self._canvas.size = original_size

        # Use vispy's I/O or fall back to a simple writer.
        try:
            from vispy.io import write_png  # type: ignore[import-untyped]

            write_png(str(output_path), rendered_image)
        except ImportError:
            # Fallback: save via PIL/Pillow if available.
            try:
                from PIL import Image  # type: ignore[import-untyped]

                image = Image.fromarray(rendered_image)
                image.save(str(output_path))
            except ImportError:
                raise ImportError(
                    "Cannot save screenshot: neither vispy.io.write_png "
                    "nor Pillow is available.  Install Pillow with:\n"
                    "  pip install Pillow"
                )

    def run(self) -> None:
        """Start the Vispy event loop and display the interactive viewer.

        This call blocks until the window is closed.
        """
        self._canvas.show()
        app.run()


# ---------------------------------------------------------------------------
# Factory / convenience functions
# ---------------------------------------------------------------------------


def _auto_load_data(
    data_path: str | Path,
) -> npt.NDArray[np.float64]:
    """Load flip-time data from a file, auto-detecting the format.

    Supported formats:

    * ``.json`` -- loaded via :func:`~src.utils.io.load_results_json`
      and reshaped to 3-D.
    * ``.npy`` -- loaded via :func:`~src.utils.io.load_results_memmap`
      (base path without extension).
    * Bare path (no extension) -- assumed to be a memmap base path.

    Parameters
    ----------
    data_path : str or Path
        Path to the data file.

    Returns
    -------
    np.ndarray
        3-D array of shape ``(n, n, n)`` with flip-time values.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    ValueError
        If the file format cannot be determined.
    """
    file_path = Path(data_path)

    if file_path.suffix == ".json":
        results = load_results_json(file_path)
        grid_size = results["grid_size"]
        flip_times_flat = results["flip_times"]
        flip_times_3d = flip_times_flat.reshape(grid_size, grid_size, grid_size)
        return flip_times_3d

    elif file_path.suffix == ".npy":
        # Strip the .npy extension to get the base path for memmap loader.
        base_path = file_path.with_suffix("")
        results = load_results_memmap(base_path)
        flip_times = np.array(results["flip_times"])
        if flip_times.ndim == 3:
            return flip_times
        # If somehow flat, try to infer grid size.
        grid_size = round(flip_times.size ** (1.0 / 3.0))
        return flip_times.reshape(grid_size, grid_size, grid_size)

    elif file_path.suffix == "":
        # No extension -- try memmap loader directly.
        npy_path = file_path.with_suffix(".npy")
        if npy_path.exists():
            results = load_results_memmap(file_path)
            flip_times = np.array(results["flip_times"])
            if flip_times.ndim == 3:
                return flip_times
            grid_size = round(flip_times.size ** (1.0 / 3.0))
            return flip_times.reshape(grid_size, grid_size, grid_size)

    raise ValueError(
        f"Cannot auto-detect format for {data_path!r}.  "
        f"Supported formats: .json, .npy (memmap)."
    )


def create_volume_viewer(
    data_path: str | Path,
    t_max: float = 15.0,
) -> ChaosVolumeViewer:
    """Load data and create an interactive :class:`ChaosVolumeViewer`.

    Auto-detects the file format (JSON or memmap ``.npy``) and
    constructs the viewer.

    Parameters
    ----------
    data_path : str or Path
        Path to the simulation results file.
    t_max : float
        Maximum simulation time for colour normalisation (default 15.0).

    Returns
    -------
    ChaosVolumeViewer
        A configured viewer instance ready for :meth:`~ChaosVolumeViewer.run`.
    """
    _require_vispy()

    flip_times_3d = _auto_load_data(data_path)
    viewer = ChaosVolumeViewer(flip_times_3d, t_max=t_max)
    return viewer


def render_screenshot(
    flip_times_3d: npt.NDArray[np.float64],
    output_path: str | Path = "renders/volume.png",
    elevation: float = 30.0,
    azimuth: float = 45.0,
    size: tuple[int, int] = (1920, 1080),
) -> None:
    """Render a non-interactive screenshot of the chaos volume.

    Creates a viewer, positions the camera, captures a screenshot, and
    closes the canvas.  Intended for batch rendering without a display.

    Parameters
    ----------
    flip_times_3d : np.ndarray
        3-D array of shape ``(n, n, n)`` with flip-time values.
    output_path : str or Path
        Destination file path (default ``"renders/volume.png"``).
    elevation : float
        Camera elevation angle in degrees (default 30).
    azimuth : float
        Camera azimuth angle in degrees (default 45).
    size : tuple of int
        ``(width, height)`` in pixels (default ``(1920, 1080)``).
    """
    _require_vispy()

    viewer = ChaosVolumeViewer(flip_times_3d)

    # Position the camera for the requested viewpoint.
    viewer._view.camera.elevation = elevation
    viewer._view.camera.azimuth = azimuth

    viewer.screenshot(output_path, size=size)

    # Clean up the canvas.
    viewer._canvas.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    _require_vispy()

    default_data_path = "data/triple_pendulum_results.json"

    if len(sys.argv) > 1:
        data_file_path = sys.argv[1]
    else:
        data_file_path = default_data_path

    data_path = Path(data_file_path)
    if not data_path.exists():
        # Try adding .npy extension for memmap files.
        npy_variant = data_path.with_suffix(".npy")
        if npy_variant.exists():
            data_path = data_path  # use base path for memmap loader
        else:
            print(f"Error: data file not found: {data_file_path}")
            print(f"Usage: python {sys.argv[0]} [path/to/data.json|path/to/data]")
            sys.exit(1)

    print(f"Loading data from {data_path} ...")
    viewer = create_volume_viewer(data_path)

    print("Launching interactive viewer.")
    print("  - Mouse drag to orbit")
    print("  - Scroll to zoom")
    print("  - Shift+drag to pan")
    viewer.run()
