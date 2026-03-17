import sys

PROJECT_ROOT = "/home/mhpereir/eulerian_heat_budget_surface"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt

from src import config
from src.specs import DomainRequest
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


def _make_request(*, 
                  bbox: tuple[float, float, float, float],
                  margin_n: int,
    ) -> DomainRequest:
    
    return DomainRequest(
        bbox=bbox,
        margin_n=margin_n,
        zg_top_pressure=500*100,
        zg_bottom="pressure_level",
        zg_bottom_pressure=1000*100,
        allow_bottom_overflow=True,
        in_surface_variables=False,
    )


def test_determine_domain_crops_to_cells_and_sets_bounds():

    bbox = (0.0, 6.0, 10.0, 15.0) # lat_min, lat_max, lon_min, lon_max
    ds = _make_dataset(
        level=[1000*100, 900*100, 800*100, 700*100, 600*100, 500*100],  # N_cells=4
        lat=np.linspace(0,6,7,   endpoint=True),
        lon=np.linspace(10,15,6, endpoint=True), 
    )

    print(ds["lat"].values) # [0. 1.2 2.4 3.6 4.8 6.]
    print(ds["lon"].values) # [10. 11.25 12.5 13.75 15.]

    dom, halo, spec = determine_domain(ds, _make_request(bbox=bbox, margin_n=1))

    # After cropping: N_cells_keep = (N_start-1) - 2*margin = 4 - 2 = 2
    assert dom.sizes["lat"] == (7-1) - 2
    assert dom.sizes["lon"] == (6-1) - 2
    assert dom.sizes["level"] == 6

    assert halo.sizes["lat"] == (7-1) 
    assert halo.sizes["lon"] == (6-1)
    assert halo.sizes["level"] == 6

    
    # bounds live on the same dims
    np.testing.assert_allclose(dom["lat_start"].values, np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(dom["lat_end"].values,   np.array([2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_allclose(dom["lon_start"].values, np.array([11.0, 12.0, 13.0]))
    np.testing.assert_allclose(dom["lon_end"].values,   np.array([12.0, 13.0, 14.0]))


    # lat/lon become cell centers
    np.testing.assert_allclose(dom["lat"].values, np.array([1.5, 2.5, 3.5, 4.5]))
    np.testing.assert_allclose(dom["lon"].values, np.array([11.5, 12.5, 13.5]))

    
    # attrs store true domain edges
    assert dom.attrs["lat_min"] == 1.0
    assert dom.attrs["lat_max"] == 5.0
    assert dom.attrs["lon_min"] == 11.0
    assert dom.attrs["lon_max"] == 14.0

    assert spec.lat_min == 1.0
    assert spec.lat_max == 5.0
    assert spec.lon_min == 11.0
    assert spec.lon_max == 14.0
    assert spec.zg_top_pressure == 500*100
    assert spec.zg_bottom == "pressure_level"
    assert spec.zg_bottom_pressure == 1000*100
    assert spec.allow_bottom_overflow is True

    # traceability ids exist on the same dims
    np.testing.assert_array_equal(dom["lat_cell_id"].values, np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(dom["lon_cell_id"].values, np.array([1, 2, 3]))

    # margin too large should error
    with pytest.raises(ValueError, match="too large"):
        determine_domain(ds, _make_request(bbox=bbox, margin_n=3))


def test_horizontal_cell_areas_shape_and_positive():
    bbox=(0,40,100,130)
    ds = _make_dataset(
        level=[1000*100, 900*100, 800*100, 700*100, 600*100, 500*100],
        lat=[0.0, 10.0, 20.0, 30.0, 40.0],      # N_start=5 => N_cells=4
        lon=[100.0, 110.0, 120.0, 130.0],        # N_start=4 => N_cells=3
    )
    dom, _, _ = determine_domain(ds, _make_request(bbox=bbox, margin_n=1))

    A = get_horizontal_cell_areas(dom)
    assert A.dims == ("lat", "lon")
    assert A.shape == (2, 1)
    assert A.name == "A_horizontal"
    assert A.attrs["units"] == "m2"
    assert np.isfinite(A.values).all()
    assert np.all(A.values > 0.0)


def test_horizontal_cell_areas_analytic_regular_grid():
    bbox=(0,40,100,130)
    ds = _make_dataset(
        level=[1000*100, 900*100, 800*100, 700*100, 600*100, 500*100],
        lat=[0.0, 10.0, 20.0, 30.0, 40.0],      # N_start=5 => N_cells=4
        lon=[100.0, 110.0, 120.0, 130.0],        # N_start=4 => N_cells=3
    )
    dom, halo, _ = determine_domain(ds, _make_request(bbox=bbox, margin_n=1))

    A = get_horizontal_cell_areas(dom)

    # expected from spherical area formula using bounds
    lat_s = np.deg2rad(np.array([10.0, 20.0]))
    lat_e = np.deg2rad(np.array([20.0, 30.0]))
    d_sin_lat = np.abs(np.sin(lat_e) - np.sin(lat_s))  # (2,)

    lon_s = np.deg2rad(np.array([110.0]))
    lon_e = np.deg2rad(np.array([120.0]))
    d_lon = np.abs(lon_e - lon_s)  # (1,)

    expected = (config.R_earth ** 2) * d_sin_lat[:, None] * d_lon[None, :]
    np.testing.assert_allclose(A.values, expected, rtol=1e-12, atol=0.0)


def test_vertical_cell_areas_shapes_and_symmetries():
    bbox=(0,40,100,130)
    ds = _make_dataset(
        level=[1000*100, 900*100, 800*100, 700*100, 600*100, 500*100],
        lat=[0.0, 10.0, 20.0, 30.0, 40.0],      # N_start=5 => N_cells=4
        lon=[100.0, 110.0, 120.0, 130.0],        # N_start=4 => N_cells=3
    )
    dom, halo, _ = determine_domain(ds, _make_request(bbox=bbox, margin_n=1))

    walls = get_vertical_cell_areas(halo)

    assert walls["A_east"].dims == ("level", "lat")
    assert walls["A_west"].dims == ("level", "lat")
    assert walls["A_south"].dims == ("level", "lon")
    assert walls["A_north"].dims == ("level", "lon")

    assert walls["A_east"].shape == (6, 4)
    assert walls["A_west"].shape == (6, 4)
    assert walls["A_south"].shape == (6, 3)
    assert walls["A_north"].shape == (6, 3)

    for wall in ("A_east", "A_west", "A_south", "A_north"):
        arr = walls[wall]
        assert arr.attrs["units"] == "m*Pa"
        assert np.isfinite(arr.values).all()
        assert np.all(arr.values > 0.0)

    # east and west are geometrically identical here (same dy, dp)
    np.testing.assert_allclose(walls["A_east"].values, walls["A_west"].values, rtol=1e-12, atol=0.0)


def test_vertical_cell_areas_analytic_regular_grid():
    # Use a simple regular grid so analytic expectations are clear.
    bbox=(0,40,100,130)
    ds = _make_dataset(
        level=[1000*100, 900*100, 800*100, 700*100, 600*100, 500*100],
        lat=[0.0, 10.0, 20.0, 30.0, 40.0],      # N_start=5 => N_cells=4
        lon=[100.0, 110.0, 120.0, 130.0],        # N_start=4 => N_cells=3
    )
    dom, halo, _ = determine_domain(ds, _make_request(bbox=bbox, margin_n=1))

    walls = get_vertical_cell_areas(halo)

    # dp from centers: edges [105k, 95k, 85k, 75k] => dp=[10k,10k,10k]
    expected_dp = np.array([10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0])

    # dy for each lat cell: dphi = 10 deg for each of 4 cells
    expected_dphi = np.deg2rad(np.array([10.0, 10.0, 10.0, 10.0]))
    expected_dy = config.R_earth * expected_dphi
    expected_east = expected_dp[:, None] * expected_dy[None, :]

    # dlon for lon cells: 10 deg for each of 3 cells
    expected_dlon = np.deg2rad(np.array([10.0, 10.0, 10.0]))

    # south/north edges are true domain edges from attrs (0 and 30 here)
    south_edge_lat = np.deg2rad(0.0)
    north_edge_lat = np.deg2rad(40.0)
    expected_dx_south = config.R_earth * np.cos(south_edge_lat) * expected_dlon
    expected_dx_north = config.R_earth * np.cos(north_edge_lat) * expected_dlon

    expected_south = expected_dp[:, None] * expected_dx_south[None, :]
    expected_north = expected_dp[:, None] * expected_dx_north[None, :]

    np.testing.assert_allclose(walls["A_east"].values, expected_east, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(walls["A_west"].values, expected_east, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(walls["A_south"].values, expected_south, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(walls["A_north"].values, expected_north, rtol=1e-12, atol=0.0)

    # Because cos(lat) differs at 0 vs 30, south != north
    assert not np.allclose(walls["A_south"].values, walls["A_north"].values)


def test_cell_volumes_strict_identity():
    bbox=(0,40,100,130)
    ds = _make_dataset(
        level=[1000*100, 900*100, 800*100, 700*100, 600*100, 500*100],
        lat=[0.0, 10.0, 20.0, 30.0, 40.0],      # N_start=5 => N_cells=4
        lon=[100.0, 110.0, 120.0, 130.0],        # N_start=4 => N_cells=3
    )
    dom, halo, _ = determine_domain(ds, _make_request(bbox=bbox, margin_n=1))

    V = get_cell_volumes(halo).astype("float64")
    A = get_horizontal_cell_areas(halo).astype("float64")
    dp = np.abs(halo["p_end"] - halo["p_start"]).astype("float64")

    lhs = float(V.sum())
    rhs = float(A.sum() * dp.sum())
    assert np.isclose(lhs, rhs, rtol=1e-10, atol=0.0)


def test_cell_volumes_trapezoid_wall_sanity():
    bbox=(0,40,100,130)
    ds = _make_dataset(
        level=[1000*100, 900*100, 800*100, 700*100, 600*100, 500*100],
        lat=[0.0, 10.0, 20.0, 30.0, 40.0],      # N_start=5 => N_cells=4
        lon=[100.0, 110.0, 120.0, 130.0],        # N_start=4 => N_cells=3
    )
    dom, halo, _ = determine_domain(ds, _make_request(bbox=bbox, margin_n=1))

    V = get_cell_volumes(dom).astype("float64")
    walls = get_vertical_cell_areas(dom).astype("float64")

    lhs = float(V.sum())

    south_total = float(walls["A_south"].sum())  # m*Pa
    north_total = float(walls["A_north"].sum())  # m*Pa

    # Use true domain edges from bounds (robust even if attrs change)
    phi_min = np.deg2rad(float(dom["lat_start"].isel(lat=0)))
    phi_max = np.deg2rad(float(dom["lat_end"].isel(lat=-1)))
    Ly = config.R_earth * abs(phi_max - phi_min)

    rhs = 0.5 * (south_total + north_total) * Ly

    # This is a trapezoid-in-lat approximation -> keep tolerance loose
    assert np.isclose(lhs, rhs, rtol=1e-1, atol=0.0)


def test_determine_domain_halo_core_aligns_by_isel_1_minus1():
    """
    Ensures ds_domain is exactly the interior of ds_halo, i.e.
    ds_halo.isel(lat=slice(1,-1), lon=slice(1,-1)) aligns with ds_domain.

    Assumes determine_domain returns (ds_domain, ds_halo, spec).
    Assumes determine_domain creates cell-centered lat/lon and also attaches
    *_start/_end coords (lat_start/lat_end/lon_start/lon_end).
    """
    bbox=(0,40,100,130)
    ds = _make_dataset(
        level=[1000*100, 900*100, 800*100, 700*100, 600*100, 500*100],
        lat=[0.0, 10.0, 20.0, 30.0, 40.0],      # N_start=5 => N_cells=4
        lon=[100.0, 110.0, 120.0, 130.0],        # N_start=4 => N_cells=3
    )
    ds_domain, ds_halo, _spec = determine_domain(ds, _make_request(bbox=bbox, margin_n=1))

    core = ds_halo.isel(lat=slice(1, -1), lon=slice(1, -1))

    # 1) Dimension sizes must match
    assert ds_domain.sizes["lat"] == core.sizes["lat"]
    assert ds_domain.sizes["lon"] == core.sizes["lon"]

    # 2) Core coordinate alignment: cell centers must match exactly
    # (if you use float midpoints, allclose is safer than identical)
    xrt.assert_allclose(ds_domain["lat"], core["lat"])
    xrt.assert_allclose(ds_domain["lon"], core["lon"])

    # 3) Bounds coords should also match if present
    for name in ("lat_start", "lat_end"):
        if name in ds_domain.coords and name in core.coords:
            xrt.assert_allclose(ds_domain[name], core[name])

    for name in ("lon_start", "lon_end"):
        if name in ds_domain.coords and name in core.coords:
            xrt.assert_allclose(ds_domain[name], core[name])

    # 4) Optional: cell-id coords match (common in your grid conventions)
    for name in ("lat_cell_id", "lon_cell_id"):
        if name in ds_domain.coords and name in core.coords:
            # integer coords should be identical
            xrt.assert_identical(ds_domain[name], core[name])

    # 5) Optional: ensure ds_domain variables are a subset slice of ds_halo variables
    # (only checks variables that exist in both)
    common_vars = set(ds_domain.data_vars).intersection(core.data_vars)
    for v in common_vars:
        # This checks the actual data alignment, not just coords
        xrt.assert_allclose(ds_domain[v], core[v])