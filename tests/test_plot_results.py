import warnings
from unittest import mock

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg", force=True)

from src import plot_results


def _short_budget_dataset() -> xr.Dataset:
    time = np.arange(
        "1941-06-01T01:00:00",
        "1941-06-02T23:00:00",
        dtype="datetime64[h]",
    )
    phase = np.linspace(0.0, 2.0 * np.pi, time.size)
    return xr.Dataset(
        {
            "domain_volume": ("time", 1.0e15 + 1.0e12 * np.sin(phase)),
            "T_domain_avg": ("time", 280.0 + np.cos(phase)),
            "d_dt_T": ("time", 1.0e11 * np.sin(phase)),
            "dV_dt": ("time", 1.0e8 * np.cos(phase)),
            "dT_dt": ("time", 1.0e11 * np.sin(phase + 0.2)),
            "advection_term": ("time", 1.0e11 * np.sin(phase + 0.4)),
            "adiabatic_term": ("time", 1.0e11 * np.sin(phase + 0.6)),
            "diabatic_term": ("time", 1.0e11 * np.sin(phase + 0.8)),
            "advection_error": ("time", np.full(time.size, 1.0e9)),
        },
        coords={"time": time},
    )


def test_short_budget_plots_are_warning_free(tmp_path):
    dataset = _short_budget_dataset()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_results.plot_budget_terms_hourly(dataset, 1, str(tmp_path))
        plot_results.plot_budget_terms_hourly(dataset, 24, str(tmp_path))
        plot_results.plot_budget_terms_day_bin(dataset, str(tmp_path))

    expected = {
        "budget_terms_timeseries_hourly_smoothwindow_1.png",
        "budget_terms_timeseries_hourly_smoothwindow_24.png",
        "budget_terms_timeseries_daily.png",
    }
    assert {path.name for path in tmp_path.glob("*.png")} == expected
    assert all((tmp_path / name).stat().st_size > 0 for name in expected)


def test_daily_plot_uses_major_date_grids_not_daily_boundary_lines(tmp_path):
    dataset = _short_budget_dataset()

    with mock.patch("matplotlib.axes.Axes.axvline") as axvline:
        plot_results.plot_budget_terms_day_bin(dataset, str(tmp_path))

    axvline.assert_not_called()
