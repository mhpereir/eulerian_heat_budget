'''
Docstring for eulerian_heat_budget.src.terms

Computes individual budget components:

- `compute_storage(T)`
- `compute_advective_term(U, T)`
- `compute_adiabatic_term(omega, T)`
- `compute_diabatic_term(S,A,C)` #residual term 

May call `integrals.py` internally.
'''

import xarray as xr
import numpy as np

from . import integrals, weights, config

from .specs import DomainSpec


def compute_domain_volume(ds_domain: xr.Dataset,
                            ds_cell_volumes: xr.DataArray,
                            ds_weights_volumes: xr.DataArray,
                            DomainSpecs: DomainSpec) -> xr.DataArray:
    
    #temporary array, to hold 1s
    constant_arr = xr.zeros_like(ds_cell_volumes) + 1 #config.g #4D
    out = integrals.volume_integral_pcoords(constant_arr, ds_cell_volumes, ds_weights_volumes)

    return out.rename("V")

def compute_time_derivative(da: xr.DataArray) -> xr.DataArray:
    if da.name is None:
        raise ValueError("Input DataArray must have a name")

    # centered difference: (f[i+1] - f[i-1]) / (t[i+1] - t[i-1]), on interior points
    num = (da.shift(time=-1) - da.shift(time=1)).isel(time=slice(1, -1))
    den = (da["time"].shift(time=-1) - da["time"].shift(time=1)).isel(time=slice(1, -1))

    # if time is datetime64, convert timedelta to seconds
    if np.issubdtype(den.dtype, np.timedelta64):
        den = den / np.timedelta64(1, "s")

    ddt = num / den
    ddt.name = f"d{da.name}_dt"
    return ddt
   
def compute_storage(T: xr.DataArray,
                    ds_cell_volumes: xr.DataArray,
                    ds_weights_volumes: xr.DataArray,
                    DomainSpecs: DomainSpec) -> xr.DataArray: #S
    r'''
    Inputs: T (temperature field, 4D: time, level, lat, lon)
    Outputs: S (storage term, 3D: time-2)
    
    math term: d/dt \int T dV
    '''
    # Compute local time tendency of temperature
    
    T_int = integrals.volume_integral_pcoords(T, ds_cell_volumes, ds_weights_volumes).astype("float64")
    T_int = T_int.reset_coords(drop=True)

    T_int.name = 'T'
    dT_dt      = compute_time_derivative(T_int)

    return dT_dt

def compute_advective_term(ds_domain: xr.Dataset, #A
                           ds_halo: xr.Dataset,
                           ds_cell_areas: xr.Dataset,
                           ds_weights_areas: xr.Dataset,
                           DomainSpecs: DomainSpec,
                           integral_diagnostics_flag: bool) -> xr.Dataset:
    r'''
    math term: -\int T U \cdot dA
    '''
    
    # Compute velocity at cell faces for advection term
    # top/bottom - vertical velocity at top/bottom walls needed for advective fluxes
    w_top = ds_halo['w'].sel(level=DomainSpecs.zg_top_pressure, method=None)  # vertical velocity at top wall
    T_top = ds_halo['T'].sel(level=DomainSpecs.zg_top_pressure, method=None)  # temperature at top wall
    wT_top = w_top * T_top
    if DomainSpecs.zg_bottom == "surface_pressure":
        w_bottom  = None
        T_bottom  = None
        wT_bottom = None
    elif DomainSpecs.zg_bottom == "pressure_level":
        w_bottom  = ds_halo['w'].sel(level=DomainSpecs.zg_bottom_pressure, method=None)  # vertical velocity at bottom wall (fixed pressure)
        T_bottom  = ds_halo['T'].sel(level=DomainSpecs.zg_bottom_pressure, method=None)  # temperature at bottom wall
        wT_bottom = w_bottom * T_bottom
    else:
        raise ValueError(f"Invalid zg_bottom mode: {DomainSpecs.zg_bottom}")
    
    # Compute normal velocity components at east/west/north/south walls for advective fluxes
    # also compute cell face T for advective fluxes
    # halo cells are needed to compute these at the domain boundaries
    # note lat/lon monotonic ascending;
    u_west = 0.5 * (ds_halo['u'].isel(lon=0) + ds_halo['u'].isel(lon=1))
    T_west = 0.5 * (ds_halo['T'].isel(lon=0) + ds_halo['T'].isel(lon=1))
    uT_west = u_west * T_west

    u_east = 0.5 * (ds_halo['u'].isel(lon=-1) + ds_halo['u'].isel(lon=-2))
    T_east = 0.5 * (ds_halo['T'].isel(lon=-1) + ds_halo['T'].isel(lon=-2))
    uT_east = u_east * T_east

    v_south = 0.5 * (ds_halo['v'].isel(lat=0) + ds_halo['v'].isel(lat=1))
    T_south = 0.5 * (ds_halo['T'].isel(lat=0) + ds_halo['T'].isel(lat=1))
    vT_south = v_south * T_south

    v_north = 0.5 * (ds_halo['v'].isel(lat=-1) + ds_halo['v'].isel(lat=-2))
    T_north = 0.5 * (ds_halo['T'].isel(lat=-1) + ds_halo['T'].isel(lat=-2))
    vT_north = v_north * T_north

    # Ensure ds_halo derived variables are cropped to ds_domain
    uT_west  = uT_west.sel(lat=ds_domain.lat)
    uT_east  = uT_east.sel(lat=ds_domain.lat)

    vT_south = vT_south.sel(lon=ds_domain.lon)
    vT_north = vT_north.sel(lon=ds_domain.lon)

    wT_top   = wT_top.sel(lat=ds_domain.lat, lon=ds_domain.lon)
    if wT_bottom is not None:
        wT_bottom = wT_bottom.sel(lat=ds_domain.lat, lon=ds_domain.lon)

    uT_west  = weights._drop_if_present(uT_west, ["lon", "lon_start", "lon_end", "lon_cell_id"])
    uT_east  = weights._drop_if_present(uT_east, ["lon", "lon_start", "lon_end", "lon_cell_id"])
    vT_north = weights._drop_if_present(vT_north, ["lat", "lat_start", "lat_end", "lat_cell_id"])
    vT_south = weights._drop_if_present(vT_south, ["lat", "lat_start", "lat_end", "lat_cell_id"])
    wT_top   = weights._drop_if_present(wT_top, ["p_mid", "p_start", "p_end", "p_cell_id"])
    if wT_bottom is not None:
        wT_bottom = weights._drop_if_present(wT_bottom, ["p_mid", "p_start", "p_end", "p_cell_id"])


    if integral_diagnostics_flag:
        # prep mass continuity diagnostic variables (same croppping & variable control)
        u_west = u_west.sel(lat=ds_domain.lat)
        u_east = u_east.sel(lat=ds_domain.lat)

        v_south = v_south.sel(lon=ds_domain.lon)
        v_north = v_north.sel(lon=ds_domain.lon)

        w_top  = w_top.sel(lat=ds_domain.lat, lon=ds_domain.lon)
        if w_bottom is not None:
            w_bottom = w_bottom.sel(lat=ds_domain.lat, lon=ds_domain.lon)


        u_west  = weights._drop_if_present(u_west, ["lon", "lon_start", "lon_end", "lon_cell_id"])
        u_east  = weights._drop_if_present(u_east, ["lon", "lon_start", "lon_end", "lon_cell_id"])
        v_north = weights._drop_if_present(v_north, ["lat", "lat_start", "lat_end", "lat_cell_id"])
        v_south = weights._drop_if_present(v_south, ["lat", "lat_start", "lat_end", "lat_cell_id"])
        w_top   = weights._drop_if_present(w_top, ["p_mid", "p_start", "p_end", "p_cell_id"])
        if w_bottom is not None:
            w_bottom = weights._drop_if_present(w_bottom, ["p_mid", "p_start", "p_end", "p_cell_id"])
        

    wall_list = ['west', 'east', 'south', 'north', 'top']
    ds_uT_integrands = xr.Dataset({
        'uT_west':  uT_west,
        'uT_east':  uT_east,
        'uT_south': vT_south,
        'uT_north': vT_north,
        'uT_top':   wT_top,
    })
    if w_bottom is not None and T_bottom is not None:
        ds_uT_integrands['uT_bottom'] = wT_bottom
        wall_list.append('bottom')


    if integral_diagnostics_flag:
        ds_u_integrands = xr.Dataset({
            'u_west':  u_west,
            'u_east':  u_east,
            'u_south': v_south,
            'u_north': v_north,
            'u_top':   w_top,
        })
        if w_bottom is not None and T_bottom is not None:
            ds_u_integrands['u_bottom'] = w_bottom

    # Net advection is constrained to a 1D time series.
    advection_walls = xr.zeros_like(ds_domain["time"], dtype="float64").rename("uT_integral")
    
    out = xr.Dataset(
        data_vars={
            "net_heat_advection": advection_walls,
        }
    )
    out["net_heat_advection"].attrs.update(
        {"long_name": "Net advected heat", "units": "K m2 Pa s-1"}
    )

    if integral_diagnostics_flag:
        mass_advection_walls = xr.zeros_like(ds_domain["time"], dtype="float64")
        out["net_mass_advection"] = mass_advection_walls
        out["net_mass_advection"].attrs.update(
            {"long_name": "Net advected mass", "units": "m2 Pa s-1"}
        )

    wall_sign = {
        "west":   -1.0, #winds are west to east, west wall's normal is west facing
        "east":   +1.0, #winds are west to east, east wall's normal is east facing
        "south":  -1.0, #winds are south to north, south wall's normal is south facing
        "north":  +1.0, #winds are south to north, north wall's normal is north facing
        "top":    -1.0, #omega is top to bottom, top wall's normal is top facing
        "bottom": +1.0,  # only used when fixed-pressure bottom is enabled
    }

    residual_num   = xr.zeros_like(ds_domain["time"], dtype="float64")
    residual_denom = xr.zeros_like(ds_domain["time"], dtype="float64")
    for wall in wall_list:
        if wall == 'top' or wall == 'bottom':
            cell_wall_name = 'A_horizontal'
        else:
            cell_wall_name = f'A_{wall}'

        adv_wall = integrals.area_integral(ds_uT_integrands[f'uT_{wall}'],
                                           ds_cell_areas[cell_wall_name],
                                           ds_weights_areas[f'W_{wall}'])
        
        adv_wall = adv_wall.astype("float64").reset_coords(drop=True)
        out["net_heat_advection"] = out["net_heat_advection"] + wall_sign[wall] * adv_wall
        out["flux_contribution_" + wall] = wall_sign[wall] * adv_wall

        if integral_diagnostics_flag:
            mass_adv_wall = integrals.area_integral(ds_u_integrands[f'u_{wall}'], #type: ignore condition guarded behind if-statements
                                                    ds_cell_areas[cell_wall_name],
                                                    ds_weights_areas[f'W_{wall}'])

            mass_adv_wall = mass_adv_wall.astype("float64").reset_coords(drop=True)
            out["net_mass_advection"] = out["net_mass_advection"] + wall_sign[wall] * mass_adv_wall #type: ignore
            out["mass_flux_contribution_" + wall] = wall_sign[wall] * mass_adv_wall

            residual_denom = residual_denom + np.abs(mass_adv_wall)

    if integral_diagnostics_flag:
        residual_num = np.abs(out["net_mass_advection"])

        eps= residual_num / residual_denom

        out["abs_mass_advection_residual_fraction"] = eps

        print(f"Residual from mass advection into domain through the surfaces: \n MEAN={eps.mean().values}, MAX={eps.max().values} of time series.")
    
    return out

def compute_adiabatic_term(ds_domain: xr.Dataset, 
                           ds_cell_volumes: xr.DataArray,
                           ds_weights_volumes: xr.DataArray,
                           DomainSpecs: DomainSpec) -> xr.DataArray: # C
    r'''
    math term: \int_{V(t)} \omega \frac{RT}{c_p p} dV
    '''
    # Compute adiabatic term (vertical motion)
    integrand = ds_domain['w'] * config.R_value * ds_domain['T'] / ds_domain['level'] / config.cp

    adiabatic_heating = integrals.volume_integral_pcoords(integrand, ds_cell_volumes, ds_weights_volumes)

    adiabatic_heating = adiabatic_heating.rename('adiabatic_heating')

    return adiabatic_heating

def compute_diabatic_term(
    S: xr.DataArray, #storage, dT_dt
    A: xr.DataArray, #advection, net_heat_advection
    C: xr.DataArray, #adiabatic, adiabatic_heating
) -> xr.DataArray: # D
    r'''
    math term: S = - A + C + D  => D = S + A - C
    '''
    # Compute diabatic term as residual
    return S + A - C



#extra terms

def compute_T_domain_average(T: xr.DataArray,
                             domain_volume: xr.DataArray,
                             ds_cell_volumes: xr.DataArray,
                             ds_weights_volumes: xr.DataArray,
                             DomainSpecs: DomainSpec) -> xr.DataArray:
    r'''
    math term: \int T dV / \int dV
    '''
    T_integral = integrals.volume_integral_pcoords(T, ds_cell_volumes, ds_weights_volumes)
    # volume_integral = compute_domain_volume(ds_domain=None, ds_cell_volumes=ds_cell_volumes, ds_weights_volumes=ds_weights_volumes, DomainSpecs=DomainSpecs)

    T_domain_avg = T_integral / domain_volume
    T_domain_avg.name = "T_domain_avg"

    return T_domain_avg