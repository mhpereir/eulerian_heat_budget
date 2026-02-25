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

from . import integrals, weights

from .specs import DomainSpec

def compute_storage(T): #S
    r'''
    Inputs: T (temperature field, 4D: time, level, lat, lon)
    Outputs: S (storage term, 3D: time-2)
    
    math term: d/dt \int T dV
    '''
    # Compute local time tendency of temperature
    
    pass

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
    if integral_diagnostics_flag:
        mass_advection_walls = xr.zeros_like(ds_domain["time"], dtype="float64").rename("u_integral")

    out = xr.Dataset(
    data_vars={
            "advective_heat": advection_walls,
            "advective_mass": mass_advection_walls,
        }
    )
    out["advective_heat"].attrs.update({"long_name": "...", "units": "W"})  # whatever you use
    out["advective_mass"].attrs.update({"long_name": "...", "units": "kg/s or Pa*m^2/s proxy"})

    wall_sign = {
        "west":  -1.0, #winds are west to east, west wall's normal is west facing
        "east":  +1.0, #winds are west to east, east wall's normal is east facing
        "south": -1.0, #winds are south to north, south wall's normal is south facing
        "north": +1.0, #winds are south to north, north wall's normal is north facing
        "top":   -1.0, #omega is 
        "bottom": +1.0,  # only used when fixed-pressure bottom is enabled
    }
    for wall in wall_list:
        if wall == 'top' or wall == 'bottom':
            cell_wall_name = 'A_horizontal'
        else:
            cell_wall_name = f'A_{wall}'

        adv_wall = integrals.area_integral(ds_uT_integrands[f'uT_{wall}'],
                                           ds_cell_areas[cell_wall_name],
                                           ds_weights_areas[f'W_{wall}'])
        
        adv_wall = adv_wall.astype("float64").reset_coords(drop=True)
        advection_walls = advection_walls + wall_sign[wall] * adv_wall

        if integral_diagnostics_flag:
            mass_adv_wall = integrals.area_integral(ds_u_integrands[f'u_{wall}'], #type: ignore condition guarded behind if-statements
                                                    ds_cell_areas[cell_wall_name],
                                                    ds_weights_areas[f'W_{wall}'])

            mass_adv_wall = mass_adv_wall.astype("float64").reset_coords(drop=True)
            mass_advection_walls = mass_advection_walls + wall_sign[wall] * mass_adv_wall #type: ignore

    advection_walls.attrs.update({
        "long_name": "Net advective heat flux through control-volume walls",
        "units": "K m2 Pa s-1",
    })

    if integral_diagnostics_flag:
        advection_walls = advection_walls.assign_coords(mass_advection=mass_advection_walls) # type: ignore

    return advection_walls

def compute_adiabatic_term(omega, T): # C
    r'''
    math term: -\int \omega * dT/dp dV 
    '''
    # Compute adiabatic term (vertical motion)
    pass

def compute_diabatic_term(S, A, C): # D
    r'''
    math term: D = S - A - C
    '''
    # Compute diabatic term as residual
    return S - A - C
