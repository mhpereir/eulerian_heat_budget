import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pytest
import xarray as xr

from src.specs import DomainRequest, SurfaceBehaviour
from src.grid import determine_domain
from src.weights import volume_weights, area_weights_vertical, area_weights_horizontal


def _make_dataset(*, level, lat, lon) -> xr.Dataset:
    """
    Minimal coord-only dataset, consistent with test_grid.py conventions:
      - level: pressure-level centers (Pa)
      - lat/lon: horizontal cell-center coords
    """
    return xr.Dataset(
        coords={
            "level": xr.DataArray(np.asarray(level, dtype=float), dims=("level",)),
            "lat": xr.DataArray(np.asarray(lat, dtype=float), dims=("lat",)),
            "lon": xr.DataArray(np.asarray(lon, dtype=float), dims=("lon",)),
        }
    )


def _make_request(
    *,
    bbox: tuple[float, float, float, float],
    zg_top_pressure: float,
    zg_bottom: str,
    zg_bottom_pressure: float | None,
    margin_n: int = 0,
) -> DomainRequest:
    return DomainRequest(
        bbox=bbox,
        margin_n=margin_n,
        zg_top_pressure=zg_top_pressure,
        zg_bottom=zg_bottom,  # "surface_pressure" | "pressure_level" #type:ignore
        zg_bottom_pressure=zg_bottom_pressure,
    )


def _make_surface_behaviour(*, allow_bottom_overflow: bool) -> SurfaceBehaviour:
    return SurfaceBehaviour(
        allow_bottom_overflow=allow_bottom_overflow,
        use_surface_variables=False,
        surface_variable_mode="none",
    )


def _attach_sp(dom: xr.Dataset, sp_vals: np.ndarray) -> xr.Dataset:
    """
    Attach a surface pressure field sp(time, lat, lon) to a domain dataset.
    """
    sp = xr.DataArray(
        sp_vals.astype(float),
        dims=("time", "lat", "lon"),
        coords={"time": [0], "lat": dom["lat"], "lon": dom["lon"]},
        name="sp",
        attrs={"units": "Pa"},
    )
    return dom.assign(sp=sp)


def test_volume_weights_surface_and_top_intersections_exact():
    """
    Construct a 2-layer column where:
      - surface intersects the bottom layer (fractional)
      - CV top pressure intersects the top layer (fractional)
    Expect exact fractions.
    """
    bbox=(0,5,15,20)
    ds = _make_dataset(
        level=[1000*100, 900*100, 800*100, 700*100],     # 4 levels
        lat=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],              # 5 cell
        lon=[15., 16., 17., 18., 19., 20.],              # 5 cell
    )

    # CV top at 850 hPa (cuts the upper layer), bottom at surface pressure
    req = _make_request(
        bbox=bbox,
        zg_top_pressure=790*100,
        zg_bottom="surface_pressure",
        zg_bottom_pressure=None,
        margin_n=1,
    )
    surface_spec = _make_surface_behaviour(allow_bottom_overflow=False)
    dom, halo, spec = determine_domain(ds, req)

    # Define sp such that it cuts the *bottom* layer halfway:
    # If bottom layer is [p_end, p_start] = [90000, 100000], pick sp=95000 => 0.5
    # The top layer is [80000, 90000], but CV top is 85000 so only [85000, 90000] included => 0.5
    dom = _attach_sp(
        dom,
        sp_vals=np.full((1, dom.sizes["lat"], dom.sizes["lon"]), 1000 * 100.0),
    )

    W = volume_weights(dom, spec, surface_spec)
    assert W.dims == ("time", "level", "lat", "lon")
    assert np.isfinite(W.values).all()

    # We expect both layers to be 0.5 in this setup
    np.testing.assert_allclose(W.isel(time=0, lat=0, lon=0).values, np.array([0.5, 1, 0.6, 0]), rtol=0, atol=1e-12)

    # Sanity: in non-overflow mode, weights must be within [0,1]
    assert float(W.min()) >= 0.0
    assert float(W.max()) <= 1.0


def test_area_weights_horizontal_binary_top_and_bottom():
    bbox=(0,4,10,14)
    ds = _make_dataset(
        level=[950*100, 850*100, 750*100],     # 3 levels
        lat=[0.0, 1.0, 2.0, 3.0, 4.0],     # 4 cells
        lon=[10.0, 11.0, 12.0, 13.0, 14.0],  # 4 cells
    )

    # Choose CV top=90000 and bottom=80000 so both faces exist as pressure-level boundaries
    req = _make_request(
        bbox=bbox,
        zg_top_pressure=800*100,
        zg_bottom="pressure_level",
        zg_bottom_pressure=900*100,
        margin_n=1,
    )
    dom, halo, spec = determine_domain(ds, req)

    # sp field over the current 3x3 cropped domain
    sp_vals = np.array([[
        [950*100, 850*100, 870*100],
        [750*100, 905*100, 799*100],
        [901*100, 800*100, 920*100],
    ]])
    dom = _attach_sp(dom, sp_vals=sp_vals)

    out = area_weights_horizontal(dom, spec)
    assert "W_top" in out
    assert "W_bottom" in out

    Wt = out["W_top"]
    Wb = out["W_bottom"]
    assert Wt.dims == ("time", "lat", "lon")
    assert Wb.dims == ("time", "lat", "lon")

    # W_top=1 where sp>900
    np.testing.assert_array_equal(Wt.values, (sp_vals > 800*100).astype(float))

    # W_bottom=1 where sp>800
    np.testing.assert_array_equal(Wb.values, (sp_vals > 900*100).astype(float))


def test_area_weights_vertical_uses_boundary_slices():
    """
    Ensure vertical wall weights use the halo-cell edge values that match the wall
    flux convention:
      - east/west average the two halo columns adjacent to the boundary
      - north/south average the two halo rows adjacent to the boundary
    The retained boundary points should line up with the interior domain cells.
    """
    bbox=(0,4,10,14)
    ds = _make_dataset(
        level=[950*100, 850*100],                  # 2 levels -> easier expected values
        lat=[0.0,  1.0,  2.0,  3.0,  4.0],
        lon=[10.0, 11.0, 12.0, 13.0, 14.0],
    )
    req = _make_request(
        bbox=bbox,
        zg_top_pressure=880*100,
        zg_bottom="surface_pressure",
        zg_bottom_pressure=None,
        margin_n=1,
    )
    surface_spec = _make_surface_behaviour(allow_bottom_overflow=True)
    dom, halo, spec = determine_domain(ds, req)

    # Halo surface pressure field on the 5x5 halo grid.
    # The wall pressures are taken from the halo-adjacent cell-edge values:
    #   west  = 0.5 * (col 0 + col 1) at interior halo rows
    #   east  = 0.5 * (col-2 + col-1) at interior halo rows
    #   south = 0.5 * (row 0 + row 1) at interior halo cols
    #   north = 0.5 * (row-2 + row-1) at interior halo cols
    sp_vals = np.array([[
        [900*100, 910*100, 930*100, 940*100, 950*100],
        [910*100, 920*100, 910*100, 920*100, 930*100],
        [920*100, 930*100, 890*100, 900*100, 910*100],
        [930*100, 940*100, 1200*100, 880*100, 870*100],
        [940*100, 950*100, 960*100, 970*100, 980*100],
    ]])
    halo = _attach_sp(halo, sp_vals=sp_vals)

    out = area_weights_vertical(halo, spec, surface_spec) 

    for k in ["W_east", "W_west", "W_south", "W_north"]:
        assert k in out

    We = out["W_east"]
    Ww = out["W_west"]
    Ws = out["W_south"]
    Wn = out["W_north"]

    # dims
    assert We.dims == ("time", "level", "lat")
    assert Ww.dims == ("time", "level", "lat")
    assert Ws.dims == ("time", "level", "lon")
    assert Wn.dims == ("time", "level", "lon")
    assert We.shape == (1, 2, 3)
    assert Ww.shape == (1, 2, 3)
    assert Ws.shape == (1, 2, 3)
    assert Wn.shape == (1, 2, 3)

    # Expected boundary pressures (Pa), after averaging the two halo cells adjacent
    # to each wall:
    #   west  = [91500, 92500, 93500]
    #   east  = [92500, 90500, 87500]
    #   south = [91500, 92000, 93000]
    #   north = [94500, 108000, 92500]
    #
    # With levels [950, 850] hPa, the layer bounds are [900,1000] and [800,900] hPa.
    # For the bottom layer, overflow is allowed, so raw bottom weights may exceed 1 (not tested right now)

    expected_We = np.array([[
        [0.25, 0.05, 0.00],
        [0.20, 0.20, 0.00],
    ]])

    expected_Ww = np.array([[
        [0.15, 0.25, 0.35],
        [0.20, 0.20, 0.20],
    ]])

    expected_Ws = np.array([[
        [0.15, 0.20, 0.30],
        [0.20, 0.20, 0.20],
    ]])

    expected_Wn = np.array([[
        [0.45, 1.80, 0.25],
        [0.20, 0.20, 0.20],
    ]])

    np.testing.assert_allclose(We.values, expected_We, rtol=0, atol=1e-12)
    np.testing.assert_allclose(Ww.values, expected_Ww, rtol=0, atol=1e-12)
    np.testing.assert_allclose(Ws.values, expected_Ws, rtol=0, atol=1e-12)
    np.testing.assert_allclose(Wn.values, expected_Wn, rtol=0, atol=1e-12)


def test_area_weights_vertical_uses_boundary_slices_no_bottom_overflow():
    """
    Ensure vertical wall weights use the halo-cell edge values that match the wall
    flux convention:
      - east/west average the two halo columns adjacent to the boundary
      - north/south average the two halo rows adjacent to the boundary
    The retained boundary points should line up with the interior domain cells.
    """
    bbox=(0,4,10,14)
    ds = _make_dataset(
        level=[950*100, 850*100],                  # 2 levels -> easier expected values
        lat=[0.0,  1.0,  2.0,  3.0,  4.0],
        lon=[10.0, 11.0, 12.0, 13.0, 14.0],
    )
    req = _make_request(
        bbox=bbox,
        zg_top_pressure=880*100,
        zg_bottom="surface_pressure",
        zg_bottom_pressure=None,
        margin_n=1,
    )
    surface_spec = _make_surface_behaviour(allow_bottom_overflow=False)
    dom, halo, spec = determine_domain(ds, req)

    # Halo surface pressure field on the 5x5 halo grid.
    # The wall pressures are taken from the halo-adjacent cell-edge values:
    #   west  = 0.5 * (col 0 + col 1) at interior halo rows
    #   east  = 0.5 * (col-2 + col-1) at interior halo rows
    #   south = 0.5 * (row 0 + row 1) at interior halo cols
    #   north = 0.5 * (row-2 + row-1) at interior halo cols
    sp_vals = np.array([[
        [900*100, 910*100, 930*100, 940*100, 950*100],
        [910*100, 920*100, 910*100, 920*100, 930*100],
        [920*100, 930*100, 890*100, 900*100, 910*100],
        [930*100, 940*100, 1200*100, 880*100, 870*100],
        [940*100, 950*100, 960*100, 970*100, 980*100],
    ]])
    halo = _attach_sp(halo, sp_vals=sp_vals)

    out = area_weights_vertical(halo, spec, surface_spec) 

    for k in ["W_east", "W_west", "W_south", "W_north"]:
        assert k in out

    We = out["W_east"]
    Ww = out["W_west"]
    Ws = out["W_south"]
    Wn = out["W_north"]

    # dims
    assert We.dims == ("time", "level", "lat")
    assert Ww.dims == ("time", "level", "lat")
    assert Ws.dims == ("time", "level", "lon")
    assert Wn.dims == ("time", "level", "lon")
    assert We.shape == (1, 2, 3)
    assert Ww.shape == (1, 2, 3)
    assert Ws.shape == (1, 2, 3)
    assert Wn.shape == (1, 2, 3)

    # Expected boundary pressures (Pa), after averaging the two halo cells adjacent
    # to each wall:
    #   west  = [91500, 92500, 93500]
    #   east  = [92500, 90500, 87500]
    #   south = [91500, 92000, 93000]
    #   north = [94500, 108000, 92500]
    #
    # With levels [950, 850] hPa, the layer bounds are [900,1000] and [800,900] hPa.
    # For the bottom layer, overflow is allowed, so raw bottom weights may exceed 1 (not tested right now)

    expected_We = np.array([[
        [0.25, 0.05, 0.00],
        [0.20, 0.20, 0.00],
    ]])

    expected_Ww = np.array([[
        [0.15, 0.25, 0.35],
        [0.20, 0.20, 0.20],
    ]])

    expected_Ws = np.array([[
        [0.15, 0.20, 0.30],
        [0.20, 0.20, 0.20],
    ]])

    expected_Wn = np.array([[
        [0.45, 1.00, 0.25],
        [0.20, 0.20, 0.20],
    ]])

    np.testing.assert_allclose(We.values, expected_We, rtol=0, atol=1e-12)
    np.testing.assert_allclose(Ww.values, expected_Ww, rtol=0, atol=1e-12)
    np.testing.assert_allclose(Ws.values, expected_Ws, rtol=0, atol=1e-12)
    np.testing.assert_allclose(Wn.values, expected_Wn, rtol=0, atol=1e-12)

    # Optional generic bounds checks (still useful)
    assert float(We.min()) >= 0.0 and float(We.max()) <= 1.0
    assert float(Ww.min()) >= 0.0 and float(Ww.max()) <= 1.0
    assert float(Ws.min()) >= 0.0 and float(Ws.max()) <= 1.0
    assert float(Wn.min()) >= 0.0 and float(Wn.max()) <= 1.0
