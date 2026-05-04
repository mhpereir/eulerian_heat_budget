from pathlib import Path
import sys

import pandas as pd
import pytest
import xarray as xr

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.time_utils import (
    infer_timestep_seconds,
    integrate_rate_over_time,
    require_regular_time,
    samples_for_duration,
    select_phase_by_utc_hour,
)


def test_infer_timestep_seconds_for_hourly_time_axis():
    time = xr.DataArray(pd.date_range("2000-01-01", periods=4, freq="1h"), dims=("time",))

    assert infer_timestep_seconds(time) == 3600.0
    assert require_regular_time(time) == 3600.0


def test_samples_for_duration_for_six_hour_time_axis():
    time = xr.DataArray(pd.date_range("2000-01-01", periods=5, freq="6h"), dims=("time",))

    assert infer_timestep_seconds(time) == 21600.0
    assert samples_for_duration(time, 24 * 3600) == 4


def test_samples_for_duration_rejects_irregular_time_axis():
    time = xr.DataArray(
        pd.to_datetime(
            [
                "2000-01-01T00:00:00",
                "2000-01-01T06:00:00",
                "2000-01-01T15:00:00",
            ]
        ),
        dims=("time",),
    )

    with pytest.raises(ValueError, match="regular"):
        require_regular_time(time)


def test_integrate_rate_over_time_uses_actual_timestep():
    time = xr.DataArray(pd.date_range("2000-01-01", periods=4, freq="6h"), dims=("time",))
    rate = xr.DataArray([1.0, 1.0, 1.0, 1.0], dims=("time",), coords={"time": time})

    integrated = integrate_rate_over_time(rate)

    assert integrated.values.tolist() == [21600.0] * 4


def test_select_phase_by_utc_hour_selects_expected_hours():
    time = pd.date_range("2000-01-01", periods=24, freq="1h")
    ds = xr.Dataset(coords={"time": time})

    for phase in range(6):
        selected = select_phase_by_utc_hour(ds, stride_hours=6, phase=phase)

        assert selected["time"].dt.hour.values.tolist() == [
            phase,
            phase + 6,
            phase + 12,
            phase + 18,
        ]
        assert require_regular_time(selected["time"]) == 21600.0


def test_select_phase_by_utc_hour_rejects_invalid_stride():
    ds = xr.Dataset(coords={"time": pd.date_range("2000-01-01", periods=4, freq="1h")})

    with pytest.raises(ValueError, match="stride_hours must be positive"):
        select_phase_by_utc_hour(ds, stride_hours=0, phase=0)


def test_select_phase_by_utc_hour_rejects_invalid_phase():
    ds = xr.Dataset(coords={"time": pd.date_range("2000-01-01", periods=4, freq="1h")})

    with pytest.raises(ValueError, match="0 <= phase < stride_hours"):
        select_phase_by_utc_hour(ds, stride_hours=6, phase=6)


def test_select_phase_by_utc_hour_requires_time_coordinate():
    ds = xr.Dataset(coords={"x": [1, 2, 3]})

    with pytest.raises(ValueError, match="time coordinate"):
        select_phase_by_utc_hour(ds, stride_hours=6, phase=0)


def test_select_phase_by_utc_hour_requires_three_samples():
    ds = xr.Dataset(coords={"time": pd.date_range("2000-01-01", periods=12, freq="1h")})

    with pytest.raises(ValueError, match="at least three timestamps"):
        select_phase_by_utc_hour(ds, stride_hours=6, phase=0)
