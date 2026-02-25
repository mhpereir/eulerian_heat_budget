import sys

PROJECT_ROOT = "/home/mhpereir/eulerian_heat_budget"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src import config, cli, specs, io, validate, grid, weights, integrals 

import numpy as np



def build_request_from_cli(args) -> specs.DomainRequest:
    bbox = (
        args.lat_min if args.lat_min is not None else config.DEFAULT_BBOX[0],
        args.lat_max if args.lat_max is not None else config.DEFAULT_BBOX[1],
        args.lon_min if args.lon_min is not None else config.DEFAULT_BBOX[2],
        args.lon_max if args.lon_max is not None else config.DEFAULT_BBOX[3],
    )
    return specs.DomainRequest(
        bbox=bbox,
        margin_n=args.margin_n if args.margin_n is not None else config.DEFAULT_MARGIN_N,
        zg_top_pressure=args.zg_top_pa if args.zg_top_pa is not None else config.DEFAULT_ZG_TOP_PA,
        zg_bottom=args.zg_bottom if args.zg_bottom is not None else config.DEFAULT_ZG_BOT_MODE,
        zg_bottom_pressure=args.zg_bottom_pa if args.zg_bottom_pa is not None else config.DEFAULT_ZG_BOT_PA,
        allow_bottom_overflow=args.allow_bottom_overflow if args.allow_bottom_overflow is not None else config.DEFAULT_ALLOW_BOTTOM_OVERFLOW,
    )



if __name__ == "__main__":
    args = cli.parse_args()
    request = build_request_from_cli(args)

    # Example usage
    ds_T = io.load_era5_T(f"{config.path_data}/T.nc")
    ds_u = io.load_era5_u(f"{config.path_data}/ux.nc")
    ds_v = io.load_era5_u(f"{config.path_data}/uy.nc")
    ds_omega = io.load_era5_omega(f"{config.path_data}/uz.nc")
    ds_sp = io.load_era5_sp(f"{config.path_data}/sfp.nc") #surface pressure in Pa
    # Merge datasets on common coordinates and variables
    ds_merged = io.load_era5_merge_dataset(ds_T, ds_u, ds_v, ds_omega, ds_sp)

    # Validate merged dataset against strict schema
    validate.validate_schema(ds_merged)

    # Determine domain extent based on grid and config margin
    ds_domain, ds_halo, domain_specs = grid.determine_domain(ds_merged, request)
    
    # Construct integand cell areas and weights for integration
    ds_horizontal_cell_areas = grid.get_horizontal_cell_areas(ds_domain).astype("float64")
    ds_vertical_cell_areas   = grid.get_vertical_cell_areas(ds_domain).astype("float64")

    ds_cell_volumes = grid.get_cell_volumes(ds_domain).astype("float64")

    ds_weights_horizontal = weights.area_weights_horizontal(ds_domain, domain_specs)
    ds_weights_vertical   = weights.area_weights_vertical(ds_domain, domain_specs)
    ds_weights_volume     = weights.volume_weights(ds_domain, domain_specs)

    # pre-compute differential terms for advective, and adiabatic components
    # advective term needs du/dx, dv/dy, dw/dp @ each wall (east, west, north, south, top, bottom)
    # adiabatic term needs dT/dp at cell centers 
    # these can be computed using finite differences and stored as new variables in the dataset for use in the budget calculations
    

    # Compute velocity at cell faces for advection term
    # top/bottom - vertical velocity at top/bottom walls needed for advective fluxes
    w_top = ds_domain['w'].sel(level=domain_specs.zg_top_pressure, method=None)  # vertical velocity at top wall
    T_top = ds_domain['T'].sel(level=domain_specs.zg_top_pressure, method=None)  # temperature at top wall
    if domain_specs.zg_bottom == "surface_pressure":
        w_bottom = None
    elif domain_specs.zg_bottom == "pressure_level":
        w_bottom = ds_domain['w'].sel(level=domain_specs.zg_bottom_pressure, method=None)  # vertical velocity at bottom wall (fixed pressure)
        T_bottom = ds_domain['T'].sel(level=domain_specs.zg_bottom_pressure, method=None)  # temperature at bottom wall
    else:
        raise ValueError(f"Invalid zg_bottom mode: {domain_specs.zg_bottom}")
    
    # Compute normal velocity components at east/west/north/south walls for advective fluxes
    # also compute cell face T for advective fluxes
    # halo cells are needed to compute these at the domain boundaries
    # note lat/lon monotonic ascending;
    u_west = 0.5 * (ds_halo['u'].isel(lon=slice(0)) + ds_halo['u'].isel(lon=slice(1)))
    T_west = 0.5 * (ds_halo['T'].isel(lon=slice(0)) + ds_halo['T'].isel(lon=slice(1)))

    u_east = 0.5 * (ds_halo['u'].isel(lon=slice(-1)) + ds_halo['u'].isel(lon=slice(-2)))
    T_east = 0.5 * (ds_halo['T'].isel(lon=slice(-1)) + ds_halo['T'].isel(lon=slice(-2)))

    v_south = 0.5 * (ds_halo['v'].isel(lat=slice(0)) + ds_halo['v'].isel(lat=slice(1)))
    T_south = 0.5 * (ds_halo['T'].isel(lat=slice(0)) + ds_halo['T'].isel(lat=slice(1)))

    v_north = 0.5 * (ds_halo['v'].isel(lat=slice(-1)) + ds_halo['v'].isel(lat=slice(-2)))
    T_north = 0.5 * (ds_halo['T'].isel(lat=slice(-1)) + ds_halo['T'].isel(lat=slice(-2)))

    