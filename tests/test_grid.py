import sys

PROJECT_ROOT = "/home/mhpereir/eulerian_heat_budget"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pytest
import xarray as xr

from src import config
from src.grid import (
    determine_domain,
    get_horizontal_cell_areas,
    get_vertical_cell_areas,
    get_cell_volumes,
)


def _make_dataset(*, level, lat, lon) -> xr.Dataset:
    """
    Build a minimal dataset with coordinates only.

    Conventions used in tests:
      - level is provided as pressure-level *centers* (can be decreasing)
      - lat/lon are provided as *cell start* coordinates (length N_start)
        so implied cell count is N_start-1
    """
    return xr.Dataset(
        coords={
            "level": xr.DataArray(np.asarray(level, dtype=float), dims=("level",)),
            "lat": xr.DataArray(np.asarray(lat, dtype=float), dims=("lat",)),
            "lon": xr.DataArray(np.asarray(lon, dtype=float), dims=("lon",)),
        }
    )


def test_determine_domain_crops_to_cells_and_sets_bounds(monkeypatch):
    ds = _make_dataset(
        level=[100000.0, 90000.0, 80000.0],
        lat=[0.0, 1.0, 2.0, 3.0, 4.0],       # N_start=5 => N_cells=4
        lon=[10.0, 11.0, 12.0, 13.0, 14.0],  # N_start=5 => N_cells=4
    )

    monkeypatch.setattr(config, "margin", 1)
    dom = determine_domain(ds)

    # After cropping: N_cells_keep = (N_start-1) - 2*margin = 4 - 2 = 2
    assert dom.sizes["lat"] == 2
    assert dom.sizes["lon"] == 2
    assert dom.sizes["level"] == 3


    # lat/lon become cell centers (Option A)
    np.testing.assert_allclose(dom["lat"].values, np.array([1.5, 2.5]))
    np.testing.assert_allclose(dom["lon"].values, np.array([11.5, 12.5]))

    # bounds live on the same dims
    np.testing.assert_allclose(dom["lat_start"].values, np.array([1.0, 2.0]))
    np.testing.assert_allclose(dom["lat_end"].values,   np.array([2.0, 3.0]))
    np.testing.assert_allclose(dom["lon_start"].values, np.array([11.0, 12.0]))
    np.testing.assert_allclose(dom["lon_end"].values,   np.array([12.0, 13.0]))

    # attrs store true domain edges
    assert dom.attrs["lat_min"] == 1.0
    assert dom.attrs["lat_max"] == 3.0
    assert dom.attrs["lon_min"] == 11.0
    assert dom.attrs["lon_max"] == 13.0

    # traceability ids exist on the same dims
    np.testing.assert_array_equal(dom["lat_cell_id"].values, np.array([1, 2]))
    np.testing.assert_array_equal(dom["lon_cell_id"].values, np.array([1, 2]))

    # margin too large should error
    monkeypatch.setattr(config, "margin", 3)
    with pytest.raises(ValueError, match="margin is too large"):
        determine_domain(ds)


def test_horizontal_cell_areas_shape_and_positive(monkeypatch):
    ds = _make_dataset(
        level=[100000.0, 90000.0, 80000.0],
        lat=[0.0, 10.0, 20.0, 30.0],      # N_start=4 => N_cells=3
        lon=[100.0, 110.0, 120.0],        # N_start=3 => N_cells=2
    )
    monkeypatch.setattr(config, "margin", 0)
    dom = determine_domain(ds)

    A = get_horizontal_cell_areas(dom)
    assert A.dims == ("lat", "lon")
    assert A.shape == (3, 2)
    assert A.name == "top"
    assert A.attrs["units"] == "m2"
    assert np.isfinite(A.values).all()
    assert np.all(A.values > 0.0)


def test_horizontal_cell_areas_analytic_regular_grid(monkeypatch):
    ds = _make_dataset(
        level=[100000.0, 90000.0, 80000.0],
        lat=[0.0, 10.0, 20.0],          # cells: [0-10], [10-20]
        lon=[100.0, 110.0, 120.0],      # cells: [100-110], [110-120]
    )
    monkeypatch.setattr(config, "margin", 0)
    dom = determine_domain(ds)

    A = get_horizontal_cell_areas(dom)

    # expected from spherical area formula using bounds
    lat_s = np.deg2rad(np.array([0.0, 10.0]))
    lat_e = np.deg2rad(np.array([10.0, 20.0]))
    d_sin_lat = np.abs(np.sin(lat_e) - np.sin(lat_s))  # (2,)

    lon_s = np.deg2rad(np.array([100.0, 110.0]))
    lon_e = np.deg2rad(np.array([110.0, 120.0]))
    d_lon = np.abs(lon_e - lon_s)  # (2,)

    expected = (config.R_earth ** 2) * d_sin_lat[:, None] * d_lon[None, :]
    np.testing.assert_allclose(A.values, expected, rtol=1e-12, atol=0.0)


def test_vertical_cell_areas_shapes_and_symmetries(monkeypatch):
    ds = _make_dataset(
        level=[100000.0, 90000.0, 80000.0, 70000.0],  # N_cells=4
        lat=[0.0, 10.0, 20.0, 30.0],    # N_cells=3
        lon=[100.0, 110.0, 120.0],      # N_cells=2
    )
    monkeypatch.setattr(config, "margin", 0)
    dom = determine_domain(ds)

    walls = get_vertical_cell_areas(dom)

    assert walls["east"].dims == ("level", "lat")
    assert walls["west"].dims == ("level", "lat")
    assert walls["south"].dims == ("level", "lon")
    assert walls["north"].dims == ("level", "lon")

    assert walls["east"].shape == (4, 3)
    assert walls["west"].shape == (4, 3)
    assert walls["south"].shape == (4, 2)
    assert walls["north"].shape == (4, 2)

    for wall in ("east", "west", "south", "north"):
        arr = walls[wall]
        assert arr.attrs["units"] == "m*Pa"
        assert np.isfinite(arr.values).all()
        assert np.all(arr.values > 0.0)

    # east and west are geometrically identical here (same dy, dp)
    np.testing.assert_allclose(walls["east"].values, walls["west"].values, rtol=1e-12, atol=0.0)


def test_vertical_cell_areas_analytic_regular_grid(monkeypatch):
    # Use a simple regular grid so analytic expectations are clear.
    ds = _make_dataset(
        level=[100000.0, 90000.0, 80000.0],   # centers => dp=10000 Pa via edges
        lat=[0.0, 10.0, 20.0, 30.0],
        lon=[100.0, 110.0, 120.0],
    )
    monkeypatch.setattr(config, "margin", 0)
    dom = determine_domain(ds)

    walls = get_vertical_cell_areas(dom)

    # dp from centers: edges [105k, 95k, 85k, 75k] => dp=[10k,10k,10k]
    expected_dp = np.array([10000.0, 10000.0, 10000.0])

    # dy for each lat cell: dphi = 10 deg for each of 3 cells
    expected_dphi = np.deg2rad(np.array([10.0, 10.0, 10.0]))
    expected_dy = config.R_earth * expected_dphi
    expected_east = expected_dp[:, None] * expected_dy[None, :]

    # dlon for lon cells: 10 deg for each of 2 cells
    expected_dlon = np.deg2rad(np.array([10.0, 10.0]))

    # south/north edges are true domain edges from attrs (0 and 30 here)
    south_edge_lat = np.deg2rad(0.0)
    north_edge_lat = np.deg2rad(30.0)
    expected_dx_south = config.R_earth * np.cos(south_edge_lat) * expected_dlon
    expected_dx_north = config.R_earth * np.cos(north_edge_lat) * expected_dlon

    expected_south = expected_dp[:, None] * expected_dx_south[None, :]
    expected_north = expected_dp[:, None] * expected_dx_north[None, :]

    np.testing.assert_allclose(walls["east"].values, expected_east, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(walls["west"].values, expected_east, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(walls["south"].values, expected_south, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(walls["north"].values, expected_north, rtol=1e-12, atol=0.0)

    # Because cos(lat) differs at 0 vs 30, south != north
    assert not np.allclose(walls["south"].values, walls["north"].values)


def test_cell_volumes_strict_identity(monkeypatch):
    ds = _make_dataset(
        level=[100000.0, 90000.0, 80000.0],
        lat=[0.0, 10.0, 20.0, 30.0],
        lon=[100.0, 110.0, 120.0],
    )
    monkeypatch.setattr(config, "margin", 0)
    dom = determine_domain(ds)

    V = get_cell_volumes(dom).astype("float64")
    A = get_horizontal_cell_areas(dom).astype("float64")  # "top"
    dp = np.abs(dom["p_end"] - dom["p_start"]).astype("float64")

    lhs = float(V.sum())
    rhs = float(A.sum() * dp.sum())
    assert np.isclose(lhs, rhs, rtol=1e-10, atol=0.0)


def test_cell_volumes_trapezoid_wall_sanity(monkeypatch):
    ds = _make_dataset(
        level=[100000.0, 90000.0, 80000.0],
        lat=[0.0, 10.0, 20.0, 30.0],
        lon=[100.0, 110.0, 120.0],
    )
    monkeypatch.setattr(config, "margin", 0)
    dom = determine_domain(ds)

    V = get_cell_volumes(dom).astype("float64")
    walls = get_vertical_cell_areas(dom)

    lhs = float(V.sum())

    south_total = float(walls["south"].sum())  # m*Pa
    north_total = float(walls["north"].sum())  # m*Pa

    # Use true domain edges from bounds (robust even if attrs change)
    phi_min = np.deg2rad(float(dom["lat_start"].isel(lat=0)))
    phi_max = np.deg2rad(float(dom["lat_end"].isel(lat=-1)))
    Ly = config.R_earth * abs(phi_max - phi_min)

    rhs = 0.5 * (south_total + north_total) * Ly

    # This is a trapezoid-in-lat approximation -> keep tolerance loose
    assert np.isclose(lhs, rhs, rtol=1e-1, atol=0.0)
