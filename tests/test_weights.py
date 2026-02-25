import sys

PROJECT_ROOT = "/home/mhpereir/eulerian_heat_budget"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pytest
import xarray as xr

from src.specs import DomainRequest
from src.grid import determine_domain
from src.weights import volume_weights, area_weights_vertical, area_weights_horizontal


def _make_dataset(*, level, lat, lon) -> xr.Dataset:
    """
    Minimal coord-only dataset, consistent with test_grid.py conventions:
      - level: pressure-level centers (Pa)
      - lat/lon: cell start coords (so N_cells = N_start-1)
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
    allow_bottom_overflow: bool,
    margin_n: int = 0,
) -> DomainRequest:
    return DomainRequest(
        bbox=bbox,
        margin_n=margin_n,
        zg_top_pressure=zg_top_pressure,
        zg_bottom=zg_bottom,  # "surface_pressure" | "pressure_level" #type:ignore
        zg_bottom_pressure=zg_bottom_pressure,
        allow_bottom_overflow=allow_bottom_overflow,
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
        allow_bottom_overflow=False,
        margin_n=1,
    )
    dom, halo, spec = determine_domain(ds, req)

    # Define sp such that it cuts the *bottom* layer halfway:
    # If bottom layer is [p_end, p_start] = [90000, 100000], pick sp=95000 => 0.5
    # The top layer is [80000, 90000], but CV top is 85000 so only [85000, 90000] included => 0.5
    dom = _attach_sp(dom, sp_vals=np.zeros((1, 3, 3)) + np.array([[[1000*100]]]))  # shape (time=1, lat=6, lon=6)

    W = volume_weights(dom, spec)["W_volume"]
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
        allow_bottom_overflow=False,
        margin_n=1,
    )
    dom, halo, spec = determine_domain(ds, req)

    # sp field: one point above top, one between faces, one below bottom etc.
    # shape (time=1, lat=2, lon=2)
    sp_vals = np.array([[
        [950*100, 850*100],
        [750*100, 905*100],
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
    Ensure W_east/W_west use lon boundary slices and W_north/W_south use lat boundary slices.
    This is a shape + value test for one simple level.
    """
    bbox=(0,4,10,14)
    ds = _make_dataset(
        level=[950*100, 850*100],              # 1 level -> easier expected values
        lat=[0.0, 1.0, 2.0, 3.0, 4.0],           # 2 cells => lat centers len=2 after determine_domain
        lon=[10.0, 11.0, 12.0, 13.0, 14.0],        # 2 cells => lon centers len=2
    )
    req = _make_request(
        bbox=bbox,
        zg_top_pressure=880*100,
        zg_bottom="surface_pressure",
        zg_bottom_pressure=None,
        allow_bottom_overflow=True,
        margin_n=1,
    )
    dom, halo, spec = determine_domain(ds, req)

    # Create sp(time,lat,lon) with distinct boundary values:
    # west boundary (lon=0) differs from east boundary (lon=-1)
    sp_vals = np.array([[
        [910*100, 930*100],  # lat0: west=910, east=930
        [920*100, 890*100],  # lat1: west=920, east=890
    ]])
    dom = _attach_sp(dom, sp_vals=sp_vals)

    out = area_weights_vertical(dom, spec)

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

    # Expectation given CV top = 880 hPa and two layers ~[900,1000] and ~[800,900]:
    # - level=950 hPa layer fraction = (sp - 900) / 100
    # - level=850 hPa layer fraction = 0
    #
    # East wall uses sp[:, :, -1] (lon=-1), West wall uses sp[:, :, 0] (lon=0)
    # South wall uses sp[:, 0, :] (lat=0), North wall uses sp[:, -1, :] (lat=-1)

    # level index 0 corresponds to 950 hPa in your descending input list
    # level index 1 corresponds to 850 hPa

    expected_We = np.array([[
        [0.30, 0.00],   # level=950: (930-900)/100=0.3, (890<900)/100=0.0 for lat0/lat1
        [0.20, 0.10],   # level=850: (900-880)/100=0.2, (890-880)/100=0.1 for lat0/lat1
    ]])

    expected_Ww = np.array([[
        [0.10, 0.20],   # level=950: (910-900)/100=0.1, (920-900)/100=0.2 for lat0/lat1
        [0.20, 0.20],   # level=850: (900-880)/100=0.2, (900-880)/100=0.2 for lat0/lat1
    ]])

    expected_Ws = np.array([[
        [0.10, 0.30],   # lat0 level=950: (910-900)/100=0.1, (930-900)/100=0.3 for lon0/lon1
        [0.20, 0.20],   # lat0 level=850: (900-880)/100=0.2, (900-880)/100=0.2 for lon0/lon1
    ]])

    expected_Wn = np.array([[
        [0.20, 0.00],   # lat1 level=950: (920-900)/100=0.2, (890<900)/100=0.0 for lon0/lon1
        [0.20, 0.10],   # lat1 level=850: (900-880)/100=0.2, (890-880)/100=0.1 for lon0/lon1
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


