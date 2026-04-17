from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import plot_results


def _time_dataset(freq: str, periods: int) -> xr.Dataset:
    time = pd.date_range("2000-01-01", periods=periods, freq=freq)
    return xr.Dataset(coords={"time": time})


def test_timeseries_default_smoothing_maps_twenty_four_hours_to_four_samples_for_six_hour_data():
    ds_budget = _time_dataset("6h", 8)

    window = plot_results._rolling_window_samples(ds_budget, 24)

    assert window == 4


def test_daily_total_is_cadence_neutral_for_hourly_and_six_hour_inputs():
    hourly_time = pd.date_range("2000-01-01", periods=48, freq="1h")
    six_hour_time = pd.date_range("2000-01-01", periods=8, freq="6h")

    hourly_rate = xr.DataArray(
        np.full(hourly_time.size, 1 / 86400),
        dims=("time",),
        coords={"time": hourly_time},
    )
    six_hour_rate = xr.DataArray(
        np.full(six_hour_time.size, 1 / 86400),
        dims=("time",),
        coords={"time": six_hour_time},
    )

    hourly_daily = plot_results._daily_total(hourly_rate)
    six_hour_daily = plot_results._daily_total(six_hour_rate)

    np.testing.assert_allclose(hourly_daily.values, [1.0, 1.0])
    np.testing.assert_allclose(six_hour_daily.values, [1.0, 1.0])
