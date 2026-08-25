import numpy as np
import pytest
import xarray as xr

from src import plot_diagnostics


@pytest.mark.filterwarnings("error")
def test_fig1_mass_continuity_handles_empty_quantile_bins(tmp_path):
    time = np.arange(40)
    volume_tendency = xr.DataArray(
        np.repeat([0.0, 1.0], 20),
        dims="time",
        coords={"time": time},
    )
    advection = xr.Dataset(
        {
            "net_mass_advection": xr.DataArray(
                np.repeat([0.25, 1.25], 20),
                dims="time",
                coords={"time": time},
            )
        }
    )

    plot_diagnostics.fig1_mass_continuity(
        volume_tendency,
        advection,
        str(tmp_path),
    )

    for filename in (
        "fig1_mass_continuity.png",
        "fig1.3_mass_continuity_binned_residual.png",
    ):
        assert (tmp_path / filename).stat().st_size > 0


@pytest.mark.filterwarnings("error")
def test_nanmean_or_nan_preserves_nonempty_nanmean_semantics():
    assert np.isnan(plot_diagnostics._nanmean_or_nan(np.array([])))
    assert np.isnan(plot_diagnostics._nanmean_or_nan(np.array([np.nan, np.nan])))
    assert plot_diagnostics._nanmean_or_nan(np.array([1.0, np.nan, 3.0])) == 2.0
