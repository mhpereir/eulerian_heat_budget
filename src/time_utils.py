"""
Utilities for working with regular time axes.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _as_time_values(time: xr.DataArray | xr.Variable | np.ndarray) -> np.ndarray:
    values = np.asarray(time)
    if values.ndim != 1:
        raise ValueError("Time coordinate must be one-dimensional.")
    return values


def infer_timestep_seconds(time: xr.DataArray | xr.Variable | np.ndarray) -> float:
    values = _as_time_values(time)
    if values.size < 2:
        raise ValueError("At least two time points are required to infer a time step.")

    delta = values[1] - values[0]
    if np.issubdtype(np.asarray(delta).dtype, np.timedelta64):
        return float(delta / np.timedelta64(1, "s"))
    return float(delta)


def require_regular_time(time: xr.DataArray | xr.Variable | np.ndarray) -> float:
    values = _as_time_values(time)
    if values.size < 2:
        raise ValueError("At least two time points are required for regularity checks.")

    diffs = np.diff(values)
    first = diffs[0]

    if np.issubdtype(np.asarray(first).dtype, np.timedelta64):
        diffs_seconds = diffs / np.timedelta64(1, "s")
        first_seconds = float(first / np.timedelta64(1, "s"))
    else:
        diffs_seconds = diffs.astype(float)
        first_seconds = float(first)

    if not np.allclose(diffs_seconds, first_seconds):
        raise ValueError("Time coordinate must be regular (constant time step)")

    return first_seconds


def samples_for_duration(
    time: xr.DataArray | xr.Variable | np.ndarray,
    duration_seconds: float,
) -> int:
    if duration_seconds <= 0:
        raise ValueError("Duration must be positive.")

    dt_seconds = require_regular_time(time)
    samples = duration_seconds / dt_seconds
    rounded_samples = round(samples)
    if not np.isclose(samples, rounded_samples):
        raise ValueError(
            "Requested duration must be an integer multiple of the dataset time step."
        )
    return max(1, int(rounded_samples))


def integrate_rate_over_time(rate: xr.DataArray, time: xr.DataArray | None = None) -> xr.DataArray:
    time_coord = rate["time"] if time is None else time
    dt_seconds = require_regular_time(time_coord)
    return rate * dt_seconds
