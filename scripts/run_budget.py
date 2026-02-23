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
    ds_sfp = io.load_era5_sfp(f"{config.path_data}/sfp.nc") #surface pressure
    # Merge datasets on common coordinates and variables
    ds_merged = io.load_era5_merge_dataset(ds_T, ds_u, ds_v, ds_omega, ds_sfp)

    # Validate merged dataset against strict schema
    validate.validate_dataset(ds_merged)

    # Determine domain extent based on grid and config margin
    ds_domain, domain_specs = grid.determine_domain(ds_merged, request)
    
    # # Construct integand cell weights and masks
    ds_horizontal_cell_areas = grid.get_horizontal_cell_areas(ds_domain).astype("float64")
    ds_vertical_cell_areas   = grid.get_vertical_cell_areas(ds_domain).astype("float64")

    ds_cell_volumes = grid.get_cell_volumes(ds_domain).astype("float64")

    print(ds_domain)

    print(ds_horizontal_cell_areas)

    print(ds_vertical_cell_areas['east'])

    print(ds_cell_volumes)

