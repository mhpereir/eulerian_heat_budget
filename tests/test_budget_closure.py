import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import numpy as np
import xarray as xr
import numpy.testing as npt

import pytest

from src import config

from src.specs import DomainRequest, SurfaceBehaviour
from src.grid import (
    determine_domain,
    get_horizontal_cell_areas,
    get_vertical_cell_areas,
)
from src.weights import area_weights_horizontal, area_weights_vertical
from src.terms import compute_advective_term, prepare_advective_faces


def _make_dataset_with_state(*, time, level, lat, lon, u0=0.0, v0=0.0, w0=0.0, T0=300.0, sp0=1000e2):
    """
    Build a minimal state dataset on the 'full-cell-start' horizontal convention
    expected by determine_domain().

    Parameters
    ----------
    lat, lon : arrays of cell-start coordinates
        So if len(lat)=N+1 and len(lon)=M+1, the domain has N x M cells before margining.
    level : pressure-level centers (Pa), descending
    """
    time = np.asarray(time)
    level = np.asarray(level, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    nt = time.size
    nz = level.size
    ny = lat.size
    nx = lon.size

    ds = xr.Dataset(
        coords={
            "time": ("time", time),
            "level": ("level", level),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        }
    )

    shape_4d = (nt, nz, ny, nx)
    shape_3d = (nt, ny, nx)

    ds["u"] = xr.DataArray(
        np.full(shape_4d, u0, dtype=np.float64),
        dims=("time", "level", "lat", "lon"),
    )
    ds["v"] = xr.DataArray(
        np.full(shape_4d, v0, dtype=np.float64),
        dims=("time", "level", "lat", "lon"),
    )
    ds["w"] = xr.DataArray(
        np.full(shape_4d, w0, dtype=np.float64),
        dims=("time", "level", "lat", "lon"),
    )
    ds["T"] = xr.DataArray(
        np.full(shape_4d, T0, dtype=np.float64),
        dims=("time", "level", "lat", "lon"),
    )
    ds["sp"] = xr.DataArray(
        np.full(shape_3d, sp0, dtype=np.float64),
        dims=("time", "lat", "lon"),
    )

    return ds


def _build_geometry_and_weights(ds_full):
    """
    Use a fixed-pressure-bottom control volume so the box is closed.
    """
    req = DomainRequest(
        bbox=(10.0, 40.0, 100.0, 130.0),
        margin_n=1,
        zg_top_pressure=500e2,
        zg_bottom="pressure_level",
        zg_bottom_pressure=1000e2,
    )
    surface_spec = SurfaceBehaviour(
        allow_bottom_overflow=False,
        use_surface_variables=False,
        surface_variable_mode="none",
    )

    ds_domain, ds_halo, spec = determine_domain(ds_full, req)

    ds_cell_areas = xr.merge(
        [
            get_vertical_cell_areas(ds_halo),
            xr.Dataset({"A_horizontal": get_horizontal_cell_areas(ds_domain)}),
        ],
        join="outer",
        compat="override",
    )

    

    ds_weights_areas = xr.merge(
        [
            area_weights_vertical(ds_halo, spec, surface_spec),
            area_weights_horizontal(ds_domain, spec),
        ],
        compat="no_conflicts",
    )

    return ds_domain, ds_halo, ds_cell_areas, ds_weights_areas, spec, surface_spec




def _build_geometry_and_weights_surface(ds_full):
    """
    Use a fixed-pressure-bottom control volume so the box is closed.
    """
    req = DomainRequest(
        bbox=(10.0, 40.0, 100.0, 130.0),
        margin_n=1,
        zg_top_pressure=500e2,
        zg_bottom="surface_pressure",
        zg_bottom_pressure=None,
    )
    surface_spec = SurfaceBehaviour(
        allow_bottom_overflow=False,
        use_surface_variables=False,
        surface_variable_mode="none",
    )

    ds_domain, ds_halo, spec = determine_domain(ds_full, req)

    ds_cell_areas = xr.merge(
        [
            get_vertical_cell_areas(ds_halo),
            xr.Dataset({"A_horizontal": get_horizontal_cell_areas(ds_domain)}),
        ],
        join="outer",
        compat="override",
    )

    

    ds_weights_areas = xr.merge(
        [
            area_weights_vertical(ds_halo, spec, surface_spec),
            area_weights_horizontal(ds_domain, spec),
        ],
        compat="no_conflicts",
    )

    return ds_domain, ds_halo, ds_cell_areas, ds_weights_areas, spec, surface_spec


def test_mass_advection_zero_flow_closed_to_machine_precision():
    """
    Exact null test:
    u = v = w = 0 everywhere, so net mass advection must be exactly zero.
    """
    ds_full = _make_dataset_with_state(
        time=np.array(["2000-01-01", "2000-01-01T01"], dtype="datetime64[h]"),
        level=[1000e2, 900e2, 800e2, 700e2, 600e2, 500e2],
        lat=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0],   # cell centers
        lon=[90.0, 100.0, 110.0, 120.0, 130.0, 140.0],
        u0=0.0,
        v0=0.0,
        w0=0.0,
        T0=300.0,
        sp0=1015e2,
    )

    ds_domain, ds_halo, ds_cell_areas, ds_weights_areas, spec, surface_spec = _build_geometry_and_weights(ds_full)

    ds_domain_adv_trim, ds_faces = prepare_advective_faces(
        ds_domain,
        ds_halo,
        spec,
        surface_spec,
        integral_diagnostics_flag=True,
    )

    out = compute_advective_term(
        ds_domain_adv_trim,
        ds_faces,
        ds_cell_areas,
        ds_weights_areas,
        spec,
        True,
    )

    npt.assert_allclose(
        out["net_mass_advection"].values,
        0.0,
        atol=1e-14,
        rtol=0.0,
    )
    npt.assert_allclose(
        out["net_heat_advection"].values,
        0.0,
        atol=1e-12,
        rtol=0.0,
    )


def test_mass_advection_uniform_zonal_flow_closed_to_machine_precision():
    """
    Nontrivial closed-flow test:
    u = constant, v = 0, w = 0, T = constant.

    In a fixed-pressure box, west/east fluxes should cancel exactly because:
      - face-normal velocity is identical on west/east boundaries
      - A_east == A_west in the current geometry construction
      - top/bottom fluxes vanish because w = 0
      - north/south fluxes vanish because v = 0
    """
    ds_full = _make_dataset_with_state(
        time=np.array(["2000-01-01", "2000-01-01T01"], dtype="datetime64[h]"),
        level=[1000e2, 900e2, 800e2, 700e2, 600e2, 500e2],
        lat=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0],   # cell centers
        lon=[90.0, 100.0, 110.0, 120.0, 130.0, 140.0],
        u0=7.5,
        v0=0.0,
        w0=0.0,
        T0=300.0,
        sp0=1015e2,
    )

    ds_domain, ds_halo, ds_cell_areas, ds_weights_areas, spec, surface_spec = _build_geometry_and_weights(ds_full)

    ds_domain_adv_trim, ds_faces = prepare_advective_faces(
        ds_domain,
        ds_halo,
        spec,
        surface_spec,
        integral_diagnostics_flag=True,
    )

    out = compute_advective_term(
        ds_domain_adv_trim,
        ds_faces,
        ds_cell_areas,
        ds_weights_areas,
        spec,
        True,
    )

    npt.assert_allclose(
        out["net_mass_advection"].values,
        0.0,
        atol=1e-12,
        rtol=0.0,
    )

    # With constant T, the net heat advection should also vanish if mass closes.
    npt.assert_allclose(
        out["net_heat_advection"].values,
        0.0,
        atol=1e-10,
        rtol=0.0,
    )


def test_mass_advection_uniform_meridional_flow_closed_to_machine_precision():
    """
    Nontrivial closed-flow test:
    u = constant, v = 0, w = 0, T = constant.

    In a fixed-pressure box, west/east fluxes should cancel exactly because:
      - face-normal velocity is identical on west/east boundaries
      - A_east == A_west in the current geometry construction
      - top/bottom fluxes vanish because w = 0
      - north/south fluxes vanish because v = 0
    """
    v0 = 7.5
    T0 = 300.0
    ds_full = _make_dataset_with_state(
        time=np.array(["2000-01-01", "2000-01-01T01"], dtype="datetime64[h]"),
        level=[1000e2, 900e2, 800e2, 700e2, 600e2, 500e2],
        lat=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0],   # cell centers
        lon=[90.0, 100.0, 110.0, 120.0, 130.0, 140.0],
        u0=0.0,
        v0=v0,
        w0=0.0,
        T0=T0,
        sp0=1015e2,
    )

    ds_domain, ds_halo, ds_cell_areas, ds_weights_areas, spec, surface_spec = _build_geometry_and_weights(ds_full)

    ds_domain_adv_trim, ds_faces = prepare_advective_faces(
        ds_domain,
        ds_halo,
        spec,
        surface_spec,
        integral_diagnostics_flag=True,
    )

    out = compute_advective_term(
        ds_domain_adv_trim,
        ds_faces,
        ds_cell_areas,
        ds_weights_areas,
        spec,
        True,
    )

    expected = (
        v0 * (
            (ds_cell_areas["A_north"] * ds_weights_areas["W_north"]).sum(dim=("level", "lon"))
            - (ds_cell_areas["A_south"] * ds_weights_areas["W_south"]).sum(dim=("level", "lon"))
        )
    ).astype("float64")

    npt.assert_allclose(
        out["net_mass_advection"].values,
        expected.values,
        rtol=1e-12,
        atol=0.0,
    )

    # With constant T, the net heat advection should also vanish if mass closes.
    npt.assert_allclose(
        out["net_heat_advection"].values,
        expected.values*T0,
        rtol=1e-10,
        atol=0.0,
    )




def test_mass_advection_uniform_meridional_and_compensating_vertical_flow_closed_to_machine_precision():
    """
    Nontrivial closed-flow test:
    u = 0, v = constant, w = constant, T = constant.

    In a fixed-pressure box, west/east fluxes should cancel exactly because:
      - face-normal velocity is identical on west/east boundaries
      - A_east == A_west in the current geometry construction
      - north/south fluxes is non-zero because v != 0 and A_south != A_north
      - bottom flux is zero because we're using surface pressure coordinates
      - top flux is built in order to cancel out the north/south net flux.  
    """
    u0 = 10.0
    v0 = 7.5
    T0 = 300.0

    ds_full = _make_dataset_with_state(
        time=np.array(["2000-01-01", "2000-01-01T01"], dtype="datetime64[h]"),
        level=[1000e2, 900e2, 800e2, 700e2, 600e2, 500e2],
        lat=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0],   # cell centers
        lon=[90.0, 100.0, 110.0, 120.0, 130.0, 140.0],
        u0=u0,
        v0=v0,
        w0=0.0,
        T0=T0,
        sp0=1000e2,
    )

    ds_domain, ds_halo, ds_cell_areas, ds_weights_areas, spec, surface_spec = _build_geometry_and_weights_surface(ds_full)

    F_ns = v0 * (
        (ds_cell_areas["A_north"] * ds_weights_areas["W_north"]).sum(dim=("level", "lon"))
        - (ds_cell_areas["A_south"] * ds_weights_areas["W_south"]).sum(dim=("level", "lon"))
    ) #m/s * m Pa = m2 Pa /s

    A_top_tot = (
        ds_cell_areas["A_horizontal"] * ds_weights_areas["W_top"]
    ).sum(dim=("lat", "lon")) # m2

    w_needed = (F_ns / A_top_tot)

    #ensure its constant with time;
    npt.assert_allclose(
        w_needed.values,
        w_needed.isel(time=0).values,
        atol=1e-14,
        rtol=0.0,
    )

    w0 = w_needed.isel(time=0).item()


    # now rebuild with the compensating vertical velocity
    ds_full = _make_dataset_with_state(
        time=np.array(["2000-01-01", "2000-01-01T01"], dtype="datetime64[h]"),
        level=[1000e2, 900e2, 800e2, 700e2, 600e2, 500e2],
        lat=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0],   # cell centers
        lon=[90.0, 100.0, 110.0, 120.0, 130.0, 140.0],
        u0=u0,
        v0=v0,
        w0=w0,
        T0=T0,
        sp0=1000e2,
    )

    ds_domain, ds_halo, ds_cell_areas, ds_weights_areas, spec, surface_spec = _build_geometry_and_weights_surface(ds_full)

    ds_domain_adv_trim, ds_faces = prepare_advective_faces(
        ds_domain,
        ds_halo,
        spec,
        surface_spec,
        integral_diagnostics_flag=True,
    )

    out = compute_advective_term(
        ds_domain_adv_trim,
        ds_faces,
        ds_cell_areas,
        ds_weights_areas,
        spec,
        True,
    )

    
    # u * A where u~10, A~1e10, flux~1e11. 1e11 * 1e-16 = 1e-5, so we expect mass advection to close to within ~1e-5 of zero due to numerical precision limits.
    npt.assert_allclose(
        out["net_mass_advection"].values,
        0.0,
        atol=1e-4,
        rtol=0.0,
    )

    for wall in ["west", "east", "south", "north", "top"]:
        npt.assert_allclose(
            ds_faces[f"uT_{wall}"].values,
            T0 * ds_faces[f"u_{wall}"].values,
            rtol=1e-12,
            atol=1e-10,
        )

    # Primary correctness check: with constant T, heat flux must be T0 times mass flux.
    npt.assert_allclose(
        out["net_heat_advection"].values,
        T0 * out["net_mass_advection"].values,
        rtol=1e-12,
        atol=1e-1,
    )

    # Secondary sanity check: the summed heat flux should also be near zero.
    npt.assert_allclose(
        out["net_heat_advection"].values,
        0.0,
        atol=3e-1,
        rtol=0.0,
    )
