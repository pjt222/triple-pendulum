"""Magma-inspired colormap for triple pendulum chaos visualisation.

Maps flip-time values to colours on a dark-purple -> red -> orange ->
yellow -> white ramp.  Voxels that never flipped (NaN) are fully
transparent so they disappear in additive-blending renderers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    pass  # reserved for future typing imports


# ---------------------------------------------------------------------------
# Colour stops: (normalised position, R, G, B) with values in [0, 1].
# Position 0.0 = fastest flip, 1.0 = slowest flip (or t_max).
# ---------------------------------------------------------------------------
COLOR_STOPS: list[tuple[float, float, float, float]] = [
    # position,   R,     G,     B
    (0.00, 0.10, 0.00, 0.20),  # very dark indigo  (fastest flip)
    (0.15, 0.30, 0.00, 0.53),  # deep purple
    (0.35, 0.72, 0.08, 0.30),  # crimson-red
    (0.55, 0.93, 0.28, 0.08),  # orange
    (0.78, 1.00, 0.78, 0.10),  # warm yellow
    (1.00, 1.00, 1.00, 1.00),  # white             (slowest flip / t_max)
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _interpolate_color_stops(
    normalized_times: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Linearly interpolate RGB from *COLOR_STOPS* for normalised values in [0, 1].

    Parameters
    ----------
    normalized_times : np.ndarray
        1-D array of values in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(normalized_times), 3)`` with RGB in [0, 1].
    """
    stop_positions = np.array([stop[0] for stop in COLOR_STOPS])
    stop_red_values = np.array([stop[1] for stop in COLOR_STOPS])
    stop_green_values = np.array([stop[2] for stop in COLOR_STOPS])
    stop_blue_values = np.array([stop[3] for stop in COLOR_STOPS])

    red_channel = np.interp(normalized_times, stop_positions, stop_red_values)
    green_channel = np.interp(normalized_times, stop_positions, stop_green_values)
    blue_channel = np.interp(normalized_times, stop_positions, stop_blue_values)

    return np.column_stack([red_channel, green_channel, blue_channel])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flip_time_to_color(
    flip_time: npt.NDArray[np.float64],
    t_max: float = 15.0,
) -> npt.NDArray[np.float64]:
    """Convert flip-time values to RGBA colours.

    Parameters
    ----------
    flip_time : np.ndarray
        1-D array of flip times (seconds).  ``np.nan`` means the
        pendulum never flipped.
    t_max : float
        Maximum simulation time used for normalisation (default 15.0).
        Flip times >= *t_max* are mapped to the top of the colour ramp.

    Returns
    -------
    np.ndarray
        Float array of shape ``(N, 4)`` with RGBA values in ``[0, 1]``.
        NaN entries receive ``alpha = 0`` (fully transparent) and a very
        dark base colour.
    """
    flat_times = np.asarray(flip_time, dtype=np.float64).ravel()
    num_points = flat_times.size

    finite_mask = np.isfinite(flat_times)

    # Normalise finite values to [0, 1], clamping at t_max.
    clamped_times = np.where(finite_mask, np.minimum(flat_times, t_max), 0.0)
    normalized_times = np.where(finite_mask, clamped_times / t_max, 0.0)

    rgb_values = _interpolate_color_stops(normalized_times)

    alpha_channel = np.where(finite_mask, 1.0, 0.0)

    rgba_array = np.empty((num_points, 4), dtype=np.float64)
    rgba_array[:, :3] = rgb_values
    rgba_array[:, 3] = alpha_channel

    # For NaN entries, set a very dark base colour so they are invisible
    # even if alpha blending is not honoured.
    nan_indices = ~finite_mask
    rgba_array[nan_indices, 0] = 0.02
    rgba_array[nan_indices, 1] = 0.0
    rgba_array[nan_indices, 2] = 0.05

    return rgba_array


def flip_time_to_hex(
    flip_time: npt.NDArray[np.float64],
    t_max: float = 15.0,
) -> list[str]:
    """Convert flip-time values to hex colour strings for JavaScript.

    Parameters
    ----------
    flip_time : np.ndarray
        1-D array of flip times (seconds).  ``np.nan`` entries map to
        ``"#050013"`` (near-black).
    t_max : float
        Maximum simulation time for normalisation (default 15.0).

    Returns
    -------
    list of str
        Hex colour strings in ``"#RRGGBB"`` format, one per input value.
    """
    rgba_colors = flip_time_to_color(flip_time, t_max=t_max)
    rgb_uint8 = np.clip(rgba_colors[:, :3] * 255, 0, 255).astype(np.uint8)
    hex_strings: list[str] = [
        f"#{red:02x}{green:02x}{blue:02x}"
        for red, green, blue in rgb_uint8
    ]
    return hex_strings


def get_matplotlib_cmap(name: str = "chaos_magma"):
    """Build and register a matplotlib ``LinearSegmentedColormap``.

    The colourmap is registered globally so that subsequent calls to
    ``matplotlib.pyplot.get_cmap(name)`` or ``plt.colormaps[name]`` return it.

    Parameters
    ----------
    name : str
        Name under which the colourmap is registered (default
        ``"chaos_magma"``).

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The constructed colourmap instance.

    Raises
    ------
    ImportError
        If matplotlib is not installed (it is an optional dependency).
    """
    try:
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for get_matplotlib_cmap(). "
            "Install it with:  pip install matplotlib"
        ) from exc

    stop_positions = [stop[0] for stop in COLOR_STOPS]
    red_values = [stop[1] for stop in COLOR_STOPS]
    green_values = [stop[2] for stop in COLOR_STOPS]
    blue_values = [stop[3] for stop in COLOR_STOPS]

    color_dict = {
        "red": [(pos, val, val) for pos, val in zip(stop_positions, red_values)],
        "green": [(pos, val, val) for pos, val in zip(stop_positions, green_values)],
        "blue": [(pos, val, val) for pos, val in zip(stop_positions, blue_values)],
    }

    colormap = LinearSegmentedColormap(name, segmentdata=color_dict, N=256)

    # Register so plt.get_cmap(name) works globally.
    try:
        plt.colormaps.register(colormap, name=name, force=True)
    except TypeError:
        # Older matplotlib versions do not support the `force` keyword.
        try:
            plt.colormaps.register(colormap, name=name)
        except ValueError:
            # Already registered from a prior call -- that is fine.
            pass

    return colormap
