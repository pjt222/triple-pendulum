"""Animated 2D slice sweeps through the 3D flip-time volume.

Creates GIF animations and static galleries that sweep through slices of
the triple pendulum's initial-condition space along each theta axis.
Each frame shows a 2D heatmap of flip times for a fixed value of one
angle, using the ``chaos_magma`` colourmap from
:mod:`src.visualization.colormap`.

Typical usage::

    from src.visualization.slice_animation import create_slice_sweep
    import numpy as np

    flip_times = np.load("data/flip_times_40.npy")
    create_slice_sweep(flip_times, axis=2)

Requires matplotlib (with the Agg backend for headless rendering).
Optionally uses imageio for simpler GIF export; falls back to
matplotlib's PillowWriter when imageio is not available.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from src.visualization.colormap import get_matplotlib_cmap

if TYPE_CHECKING:
    pass  # reserved for future typing imports

# Optional dependency: imageio produces simpler GIF output.
try:
    import imageio.v2 as imageio  # type: ignore[import-untyped]
    _HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio  # type: ignore[import-untyped,no-redef]
        _HAS_IMAGEIO = True
    except ImportError:
        _HAS_IMAGEIO = False


# ---------------------------------------------------------------------------
# Axis metadata
# ---------------------------------------------------------------------------

_AXIS_LABELS: list[str] = ["\u03b81", "\u03b82", "\u03b83"]
"""Human-readable labels for the three pendulum angle axes."""


def _remaining_axes(fixed_axis: int) -> tuple[int, int]:
    """Return the two axis indices that are *not* the fixed axis.

    Parameters
    ----------
    fixed_axis : int
        The axis held constant (0, 1, or 2).

    Returns
    -------
    tuple of int
        The two remaining axis indices, in ascending order.
    """
    all_axes = [0, 1, 2]
    all_axes.remove(fixed_axis)
    return (all_axes[0], all_axes[1])


def _index_to_angle(
    index: int,
    grid_resolution: int,
    theta_range: tuple[float, float],
) -> float:
    """Convert a grid index along one axis to the corresponding angle in degrees.

    Parameters
    ----------
    index : int
        Grid index (0-based).
    grid_resolution : int
        Total number of grid points along this axis.
    theta_range : tuple of float
        ``(theta_min, theta_max)`` in degrees.

    Returns
    -------
    float
        Angle in degrees corresponding to *index*.
    """
    theta_min, theta_max = theta_range
    if grid_resolution <= 1:
        return (theta_min + theta_max) / 2.0
    return theta_min + index * (theta_max - theta_min) / (grid_resolution - 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_slice(
    flip_times_3d: npt.NDArray[np.float64],
    axis: int,
    index: int,
    t_max: float = 15.0,
) -> npt.NDArray[np.float64]:
    """Extract a 2D slice from the 3D flip-time volume.

    Parameters
    ----------
    flip_times_3d : np.ndarray
        3-D array of shape ``(n1, n2, n3)`` with flip-time values.
        ``np.nan`` marks voxels that never flipped.
    axis : int
        Axis along which to slice (0, 1, or 2).
    index : int
        Index position along *axis* at which to extract the slice.
    t_max : float
        Maximum simulation time.  Currently unused but reserved for
        future normalisation options (default 15.0).

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_row, n_col)`` extracted from the volume.
        The row axis corresponds to the lower of the two remaining axes
        and the column axis to the higher.

    Raises
    ------
    ValueError
        If *axis* is not 0, 1, or 2, or if *index* is out of range.
    """
    if axis not in (0, 1, 2):
        raise ValueError(
            f"axis must be 0, 1, or 2, got {axis!r}"
        )
    if not (0 <= index < flip_times_3d.shape[axis]):
        raise ValueError(
            f"index {index} is out of range for axis {axis} "
            f"with size {flip_times_3d.shape[axis]}"
        )

    slice_selector: list[slice | int] = [slice(None)] * 3
    slice_selector[axis] = index
    slice_2d: npt.NDArray[np.float64] = flip_times_3d[tuple(slice_selector)]

    return np.asarray(slice_2d, dtype=np.float64)


def create_slice_sweep(
    flip_times_3d: npt.NDArray[np.float64],
    axis: int = 2,
    t_max: float = 15.0,
    theta_range: tuple[float, float] = (-170.0, 170.0),
    output_path: str | Path = "renders/slice_sweep.gif",
    fps: int = 10,
    dpi: int = 150,
) -> Path:
    """Create an animated GIF sweeping through all slices along one axis.

    Each frame is a 2D heatmap of flip times at a fixed angle along the
    chosen axis, coloured with the ``chaos_magma`` colourmap.

    Parameters
    ----------
    flip_times_3d : np.ndarray
        3-D array of shape ``(n1, n2, n3)`` with flip-time values.
    axis : int
        Axis to sweep along (0, 1, or 2).  Default 2 (theta-3).
    t_max : float
        Maximum simulation time used for colour normalisation
        (default 15.0).
    theta_range : tuple of float
        ``(theta_min, theta_max)`` in degrees (default ``(-170, 170)``).
    output_path : str or Path
        Destination file path for the GIF (default
        ``"renders/slice_sweep.gif"``).
    fps : int
        Frames per second for the animation (default 10).
    dpi : int
        Resolution in dots per inch (default 150).

    Returns
    -------
    Path
        The absolute path to the written GIF file.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    chaos_colormap = get_matplotlib_cmap("chaos_magma")
    num_slices_along_axis = flip_times_3d.shape[axis]
    row_axis, col_axis = _remaining_axes(axis)
    fixed_axis_label = _AXIS_LABELS[axis]
    row_axis_label = _AXIS_LABELS[row_axis]
    col_axis_label = _AXIS_LABELS[col_axis]

    # Spatial extent for imshow (left, right, bottom, top) in degrees.
    theta_min, theta_max = theta_range
    image_extent = [theta_min, theta_max, theta_min, theta_max]

    if _HAS_IMAGEIO:
        _create_sweep_imageio(
            flip_times_3d=flip_times_3d,
            axis=axis,
            num_slices_along_axis=num_slices_along_axis,
            chaos_colormap=chaos_colormap,
            t_max=t_max,
            theta_range=theta_range,
            image_extent=image_extent,
            fixed_axis_label=fixed_axis_label,
            row_axis_label=row_axis_label,
            col_axis_label=col_axis_label,
            output_file=output_file,
            fps=fps,
            dpi=dpi,
        )
    else:
        _create_sweep_pillow(
            flip_times_3d=flip_times_3d,
            axis=axis,
            num_slices_along_axis=num_slices_along_axis,
            chaos_colormap=chaos_colormap,
            t_max=t_max,
            theta_range=theta_range,
            image_extent=image_extent,
            fixed_axis_label=fixed_axis_label,
            row_axis_label=row_axis_label,
            col_axis_label=col_axis_label,
            output_file=output_file,
            fps=fps,
            dpi=dpi,
        )

    return output_file.resolve()


def create_multi_axis_sweep(
    flip_times_3d: npt.NDArray[np.float64],
    t_max: float = 15.0,
    theta_range: tuple[float, float] = (-170.0, 170.0),
    output_dir: str | Path = "renders/",
    fps: int = 10,
) -> list[Path]:
    """Create sweep animations for all three axes.

    Saves one GIF per axis named ``slice_sweep_theta1.gif``,
    ``slice_sweep_theta2.gif``, and ``slice_sweep_theta3.gif`` inside
    *output_dir*.

    Parameters
    ----------
    flip_times_3d : np.ndarray
        3-D array of shape ``(n1, n2, n3)`` with flip-time values.
    t_max : float
        Maximum simulation time for colour normalisation (default 15.0).
    theta_range : tuple of float
        ``(theta_min, theta_max)`` in degrees (default ``(-170, 170)``).
    output_dir : str or Path
        Directory for the output GIFs (default ``"renders/"``).
    fps : int
        Frames per second (default 10).

    Returns
    -------
    list of Path
        Absolute paths to the three generated GIF files.
    """
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    generated_paths: list[Path] = []
    for axis_index in range(3):
        axis_name = f"theta{axis_index + 1}"
        gif_filename = f"slice_sweep_{axis_name}.gif"
        gif_path = create_slice_sweep(
            flip_times_3d,
            axis=axis_index,
            t_max=t_max,
            theta_range=theta_range,
            output_path=output_directory / gif_filename,
            fps=fps,
        )
        generated_paths.append(gif_path)

    return generated_paths


def create_slice_gallery(
    flip_times_3d: npt.NDArray[np.float64],
    axis: int = 2,
    num_slices: int = 6,
    t_max: float = 15.0,
    theta_range: tuple[float, float] = (-170.0, 170.0),
    output_path: str | Path = "renders/slice_gallery.png",
    dpi: int = 200,
) -> Path:
    """Create a static gallery of evenly-spaced slices in a 2x3 grid.

    Parameters
    ----------
    flip_times_3d : np.ndarray
        3-D array of shape ``(n1, n2, n3)`` with flip-time values.
    axis : int
        Axis to slice along (0, 1, or 2).  Default 2 (theta-3).
    num_slices : int
        Number of slices to display (default 6).  These are evenly
        distributed across the axis.
    t_max : float
        Maximum simulation time for colour normalisation (default 15.0).
    theta_range : tuple of float
        ``(theta_min, theta_max)`` in degrees (default ``(-170, 170)``).
    output_path : str or Path
        Destination file path for the PNG (default
        ``"renders/slice_gallery.png"``).
    dpi : int
        Resolution in dots per inch (default 200).

    Returns
    -------
    Path
        The absolute path to the written PNG file.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    chaos_colormap = get_matplotlib_cmap("chaos_magma")
    axis_size = flip_times_3d.shape[axis]
    row_axis, col_axis = _remaining_axes(axis)
    fixed_axis_label = _AXIS_LABELS[axis]
    row_axis_label = _AXIS_LABELS[row_axis]
    col_axis_label = _AXIS_LABELS[col_axis]

    theta_min, theta_max = theta_range
    image_extent = [theta_min, theta_max, theta_min, theta_max]

    # Select evenly-spaced indices across the axis.
    slice_indices = np.linspace(0, axis_size - 1, num_slices, dtype=int)

    # Layout: 2 rows x 3 columns (or adjust if num_slices differs).
    num_rows = 2
    num_cols = (num_slices + num_rows - 1) // num_rows  # ceiling division

    figure, axes_grid = plt.subplots(
        num_rows, num_cols,
        figsize=(4.5 * num_cols, 4.0 * num_rows),
        constrained_layout=True,
    )
    axes_flat = np.asarray(axes_grid).ravel()

    # Common normalisation so all subplots share the same colour scale.
    colour_normalisation = plt.Normalize(vmin=0.0, vmax=t_max)

    last_image_handle = None
    for subplot_index, slice_index in enumerate(slice_indices):
        current_axes = axes_flat[subplot_index]
        slice_data = render_slice(flip_times_3d, axis, int(slice_index))

        fixed_angle_degrees = _index_to_angle(
            int(slice_index), axis_size, theta_range
        )

        image_handle = current_axes.imshow(
            slice_data,
            origin="lower",
            cmap=chaos_colormap,
            norm=colour_normalisation,
            extent=image_extent,
            aspect="equal",
        )
        current_axes.set_title(
            f"{fixed_axis_label} = {fixed_angle_degrees:+.1f}\u00b0",
            fontsize=10,
        )
        current_axes.set_xlabel(f"{col_axis_label} (\u00b0)", fontsize=9)
        current_axes.set_ylabel(f"{row_axis_label} (\u00b0)", fontsize=9)
        current_axes.tick_params(labelsize=8)
        last_image_handle = image_handle

    # Hide any unused subplots.
    for unused_index in range(num_slices, len(axes_flat)):
        axes_flat[unused_index].set_visible(False)

    # Shared colorbar.
    if last_image_handle is not None:
        colorbar = figure.colorbar(
            last_image_handle,
            ax=axes_flat[:num_slices].tolist(),
            label="Flip time (s)",
            shrink=0.8,
        )
        colorbar.ax.tick_params(labelsize=8)

    figure.suptitle(
        f"Slice gallery along {fixed_axis_label}",
        fontsize=13,
        fontweight="bold",
    )

    figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    return output_file.resolve()


# ---------------------------------------------------------------------------
# Internal: GIF creation backends
# ---------------------------------------------------------------------------


def _render_frame_to_array(
    flip_times_3d: npt.NDArray[np.float64],
    axis: int,
    slice_index: int,
    chaos_colormap: object,
    t_max: float,
    image_extent: list[float],
    fixed_axis_label: str,
    row_axis_label: str,
    col_axis_label: str,
    theta_range: tuple[float, float],
    dpi: int,
) -> npt.NDArray[np.uint8]:
    """Render a single slice frame and return it as an RGBA uint8 array.

    Parameters
    ----------
    flip_times_3d : np.ndarray
        Full 3-D flip-time volume.
    axis : int
        Fixed axis.
    slice_index : int
        Index along the fixed axis.
    chaos_colormap : matplotlib colormap
        Colourmap to use for the heatmap.
    t_max : float
        Maximum simulation time for colour normalisation.
    image_extent : list of float
        ``[left, right, bottom, top]`` for ``imshow``.
    fixed_axis_label : str
        Label for the fixed axis (e.g. ``"theta3"``).
    row_axis_label : str
        Label for the vertical axis.
    col_axis_label : str
        Label for the horizontal axis.
    theta_range : tuple of float
        ``(theta_min, theta_max)`` in degrees.
    dpi : int
        Resolution for the rendered frame.

    Returns
    -------
    np.ndarray
        RGBA image array of shape ``(height, width, 4)`` with uint8 dtype.
    """
    slice_data = render_slice(flip_times_3d, axis, slice_index)
    axis_size = flip_times_3d.shape[axis]
    fixed_angle_degrees = _index_to_angle(slice_index, axis_size, theta_range)

    figure, frame_axes = plt.subplots(figsize=(6, 5), constrained_layout=True)
    colour_normalisation = plt.Normalize(vmin=0.0, vmax=t_max)

    image_handle = frame_axes.imshow(
        slice_data,
        origin="lower",
        cmap=chaos_colormap,
        norm=colour_normalisation,
        extent=image_extent,
        aspect="equal",
    )
    frame_axes.set_title(
        f"{fixed_axis_label} = {fixed_angle_degrees:+.1f}\u00b0",
        fontsize=12,
    )
    frame_axes.set_xlabel(f"{col_axis_label} (\u00b0)")
    frame_axes.set_ylabel(f"{row_axis_label} (\u00b0)")

    figure.colorbar(image_handle, ax=frame_axes, label="Flip time (s)")

    # Render to in-memory RGBA buffer.
    figure.canvas.draw()
    frame_width, frame_height = figure.canvas.get_width_height()
    rgba_buffer = np.frombuffer(
        figure.canvas.buffer_rgba(), dtype=np.uint8
    ).reshape(frame_height, frame_width, 4)
    # Copy so the buffer survives figure closure.
    frame_image = rgba_buffer.copy()
    plt.close(figure)

    return frame_image


def _create_sweep_imageio(
    flip_times_3d: npt.NDArray[np.float64],
    axis: int,
    num_slices_along_axis: int,
    chaos_colormap: object,
    t_max: float,
    theta_range: tuple[float, float],
    image_extent: list[float],
    fixed_axis_label: str,
    row_axis_label: str,
    col_axis_label: str,
    output_file: Path,
    fps: int,
    dpi: int,
) -> None:
    """Write a GIF using imageio (preferred backend).

    Parameters
    ----------
    flip_times_3d : np.ndarray
        Full 3-D flip-time volume.
    axis : int
        Fixed axis.
    num_slices_along_axis : int
        Number of frames (slices along the axis).
    chaos_colormap : matplotlib colormap
        Colourmap for the heatmap.
    t_max : float
        Maximum simulation time.
    theta_range : tuple of float
        ``(theta_min, theta_max)`` in degrees.
    image_extent : list of float
        ``[left, right, bottom, top]`` for ``imshow``.
    fixed_axis_label : str
        Label for the fixed axis.
    row_axis_label : str
        Label for the vertical axis.
    col_axis_label : str
        Label for the horizontal axis.
    output_file : Path
        Destination GIF path.
    fps : int
        Frames per second.
    dpi : int
        Resolution for each frame.
    """
    frame_duration_seconds = 1.0 / fps
    rendered_frames: list[npt.NDArray[np.uint8]] = []

    for slice_index in range(num_slices_along_axis):
        frame_image = _render_frame_to_array(
            flip_times_3d=flip_times_3d,
            axis=axis,
            slice_index=slice_index,
            chaos_colormap=chaos_colormap,
            t_max=t_max,
            image_extent=image_extent,
            fixed_axis_label=fixed_axis_label,
            row_axis_label=row_axis_label,
            col_axis_label=col_axis_label,
            theta_range=theta_range,
            dpi=dpi,
        )
        # imageio expects RGB, not RGBA.
        rendered_frames.append(frame_image[:, :, :3])

    imageio.mimsave(
        str(output_file),
        rendered_frames,
        duration=frame_duration_seconds,
        loop=0,
    )


def _create_sweep_pillow(
    flip_times_3d: npt.NDArray[np.float64],
    axis: int,
    num_slices_along_axis: int,
    chaos_colormap: object,
    t_max: float,
    theta_range: tuple[float, float],
    image_extent: list[float],
    fixed_axis_label: str,
    row_axis_label: str,
    col_axis_label: str,
    output_file: Path,
    fps: int,
    dpi: int,
) -> None:
    """Write a GIF using matplotlib's PillowWriter (fallback backend).

    Parameters
    ----------
    flip_times_3d : np.ndarray
        Full 3-D flip-time volume.
    axis : int
        Fixed axis.
    num_slices_along_axis : int
        Number of frames (slices along the axis).
    chaos_colormap : matplotlib colormap
        Colourmap for the heatmap.
    t_max : float
        Maximum simulation time.
    theta_range : tuple of float
        ``(theta_min, theta_max)`` in degrees.
    image_extent : list of float
        ``[left, right, bottom, top]`` for ``imshow``.
    fixed_axis_label : str
        Label for the fixed axis.
    row_axis_label : str
        Label for the vertical axis.
    col_axis_label : str
        Label for the horizontal axis.
    output_file : Path
        Destination GIF path.
    fps : int
        Frames per second.
    dpi : int
        Resolution for each frame.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    figure, animation_axes = plt.subplots(
        figsize=(6, 5), constrained_layout=True
    )
    colour_normalisation = plt.Normalize(vmin=0.0, vmax=t_max)

    # Initialise with the first slice.
    initial_slice = render_slice(flip_times_3d, axis, 0)
    image_handle = animation_axes.imshow(
        initial_slice,
        origin="lower",
        cmap=chaos_colormap,
        norm=colour_normalisation,
        extent=image_extent,
        aspect="equal",
    )
    figure.colorbar(image_handle, ax=animation_axes, label="Flip time (s)")

    row_axis, col_axis = _remaining_axes(axis)
    col_axis_label_text = _AXIS_LABELS[col_axis]
    row_axis_label_text = _AXIS_LABELS[row_axis]
    animation_axes.set_xlabel(f"{col_axis_label_text} (\u00b0)")
    animation_axes.set_ylabel(f"{row_axis_label_text} (\u00b0)")

    axis_size = flip_times_3d.shape[axis]

    def update_frame(frame_index: int) -> list:
        """Update the heatmap data and title for the current frame."""
        slice_data = render_slice(flip_times_3d, axis, frame_index)
        image_handle.set_data(slice_data)
        fixed_angle_degrees = _index_to_angle(
            frame_index, axis_size, theta_range
        )
        animation_axes.set_title(
            f"{fixed_axis_label} = {fixed_angle_degrees:+.1f}\u00b0",
            fontsize=12,
        )
        return [image_handle]

    animation = FuncAnimation(
        figure,
        update_frame,
        frames=num_slices_along_axis,
        interval=1000 // fps,
        blit=True,
    )

    pillow_writer = PillowWriter(fps=fps)
    animation.save(str(output_file), writer=pillow_writer, dpi=dpi)
    plt.close(figure)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Create animated slice sweeps of the 3D flip-time volume."
    )
    parser.add_argument(
        "input_file",
        help="Path to a .npy file containing the 3D flip-time array.",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Axis to sweep along (default: 2).",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=15.0,
        help="Maximum simulation time for colour normalisation (default: 15.0).",
    )
    parser.add_argument(
        "--theta-min",
        type=float,
        default=-170.0,
        help="Minimum theta in degrees (default: -170).",
    )
    parser.add_argument(
        "--theta-max",
        type=float,
        default=170.0,
        help="Maximum theta in degrees (default: 170).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: renders/slice_sweep.gif).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second (default: 10).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution in dots per inch (default: 150).",
    )
    parser.add_argument(
        "--all-axes",
        action="store_true",
        help="Create sweep animations for all three axes.",
    )
    parser.add_argument(
        "--gallery",
        action="store_true",
        help="Create a static slice gallery instead of an animation.",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=6,
        help="Number of slices for the gallery (default: 6).",
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    flip_times_volume = np.load(str(input_path))
    if flip_times_volume.ndim != 3:
        print(
            f"Error: expected a 3D array, got shape {flip_times_volume.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    theta_range_arg = (args.theta_min, args.theta_max)

    if args.all_axes:
        output_dir = args.output if args.output else "renders/"
        generated_files = create_multi_axis_sweep(
            flip_times_volume,
            t_max=args.t_max,
            theta_range=theta_range_arg,
            output_dir=output_dir,
            fps=args.fps,
        )
        for generated_path in generated_files:
            print(f"Saved: {generated_path}")
    elif args.gallery:
        output_path = args.output if args.output else "renders/slice_gallery.png"
        gallery_path = create_slice_gallery(
            flip_times_volume,
            axis=args.axis,
            num_slices=args.num_slices,
            t_max=args.t_max,
            theta_range=theta_range_arg,
            output_path=output_path,
            dpi=args.dpi,
        )
        print(f"Saved: {gallery_path}")
    else:
        output_path = args.output if args.output else "renders/slice_sweep.gif"
        sweep_path = create_slice_sweep(
            flip_times_volume,
            axis=args.axis,
            t_max=args.t_max,
            theta_range=theta_range_arg,
            output_path=output_path,
            fps=args.fps,
            dpi=args.dpi,
        )
        print(f"Saved: {sweep_path}")
