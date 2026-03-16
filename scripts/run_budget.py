import sys

PROJECT_ROOT = "/home/mhpereir/eulerian_heat_budget"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config, cli, specs, io, validate, grid, budget

from src import plot_results

import xarray as xr
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
    ds_domain, ds_halo, DomainSpecs = grid.determine_domain(ds_merged, request)
    
    print('Proceeding with', DomainSpecs)

    result = budget.calculate_budget(ds_domain, ds_halo, DomainSpecs, integral_diagnostics_flag=True, plot_dir=config.DEFAULT_PLOTS_OUTPUT, plot_flag=True)


    plot_results.plot_budget_terms_hourly(result, smoothing_window=1, plot_dir=config.DEFAULT_PLOTS_OUTPUT)
    plot_results.plot_budget_terms_hourly(result, smoothing_window=24, plot_dir=config.DEFAULT_PLOTS_OUTPUT)
    plot_results.plot_budget_terms_day_bin(result, plot_dir=config.DEFAULT_PLOTS_OUTPUT)


    # testing to see if a constant temperature field, yields a net heat advection error comparable to the estimated advection error from mass continuity (delta_mass * T_scale)

    ds_domain_test = ds_domain.copy(deep=True)
    ds_domain_test['T'] = result.T_scale

    ds_halo_test = ds_halo.copy(deep=True)
    ds_halo_test['T'] = result.T_scale

    result_test = budget.calculate_budget(ds_domain_test, ds_halo_test, DomainSpecs, integral_diagnostics_flag=True, plot_dir=config.DEFAULT_PLOTS_OUTPUT+'_2', plot_flag=True)

    plot_results.plot_constant_T_results(result, result_test, plot_dir=config.DEFAULT_PLOTS_OUTPUT+'_2')

    print(result)