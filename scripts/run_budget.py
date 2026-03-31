import sys
import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import xarray as xr

from src import config, cli, specs, io, validate, grid, budget, run_outputs
from src import plot_results

import logging
from dask.distributed import Client
    

def build_request_from_cli(args) -> specs.DomainRequest:
    zg_bottom = args.zg_bottom if args.zg_bottom is not None else config.DEFAULT_ZG_BOT_MODE
    zg_bottom_pressure = (
        args.zg_bottom_pa if args.zg_bottom_pa is not None else config.DEFAULT_ZG_BOT_PA
    )
    if zg_bottom == "surface_pressure":
        zg_bottom_pressure = None

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
        zg_bottom=zg_bottom,
        zg_bottom_pressure=zg_bottom_pressure,
    )

def build_surface_behaviour_from_cli(args) -> specs.SurfaceBehaviour:

    if args.use_surface_variables if args.use_surface_variables is not None else config.DEFAULT_USE_SURFACE_VARIABLES:
        if (args.surface_variable_mode is None or args.surface_variable_mode == 'none') and config.DEFAULT_SURFACE_VARIABLE_MODE == 'none':
            raise ValueError("surface_variable_mode must be set when use_surface_variables is True")

    return specs.SurfaceBehaviour(
        allow_bottom_overflow=args.allow_bottom_overflow if args.allow_bottom_overflow is not None else config.DEFAULT_ALLOW_BOTTOM_OVERFLOW,
        use_surface_variables=args.use_surface_variables if args.use_surface_variables is not None else config.DEFAULT_USE_SURFACE_VARIABLES,
        surface_variable_mode=args.surface_variable_mode if args.surface_variable_mode is not None else config.DEFAULT_SURFACE_VARIABLE_MODE
    )


def build_data_source_from_cli(args) -> specs.DataSourceConfig:
    if args.data_source == "local_era5":
        return specs.DataSourceConfig(
            kind="local_era5",
            path_data=config.DEFAULT_LOCAL_PATH,
            time_start=args.time_start if args.time_start is not None else config.DEFAULT_TIME_START,
            time_end=args.time_end if args.time_end is not None else config.DEFAULT_TIME_END,
        )
    elif args.data_source == "arco_era5":
        return specs.DataSourceConfig(
            kind="arco_era5",
            arco_path=config.DEFAULT_ARCO_PATH,
            time_start=args.time_start if args.time_start is not None else config.DEFAULT_TIME_START,
            time_end=args.time_end if args.time_end is not None else config.DEFAULT_TIME_END,
        )
    else:
        raise ValueError(f"Unsupported data source: {args.data_source}")

def main() -> None:
    args         = cli.parse_args()
    request      = build_request_from_cli(args)
    SurfaceSpecs = build_surface_behaviour_from_cli(args)
    SourceCfg    = build_data_source_from_cli(args)
    run_paths    = run_outputs.prepare_run_paths(config.DEFAULT_PLOTS_OUTPUT)

    print(f"Saving plots to {run_paths.plot_dir}")

    ds_merged = io.load_dataset(SourceCfg, SurfaceSpecs)
    
    # Validate merged dataset against strict schema
    validate.validate_schema(ds_merged)

    # Determine domain extent based on grid and config margin
    ds_domain, ds_halo, DomainSpecs = grid.determine_domain(ds_merged, request, eager_loading=True)
    
    # Persist domain and halo datasets in memory to avoid redundant computations during budget calculations and plotting
    ds_domain = ds_domain
    ds_halo = ds_halo

    print('Proceeding with', DomainSpecs)

    metadata_path = run_outputs.write_run_info(
        run_paths,
        request=request,
        source_spec=SourceCfg,
        domain_spec=DomainSpecs,
        surface_behaviour=SurfaceSpecs,
        cli_args=vars(args),
    )
    print(f"Saved run metadata to {metadata_path}")

    result = budget.calculate_budget(
        ds_domain,
        ds_halo,
        DomainSpecs,
        SurfaceSpecs,
        integral_diagnostics_flag=True,
        plot_dir=run_paths.plot_dir,
        plot_flag=True,
    ) #already computed before returning

    plot_results.plot_budget_terms_hourly(result, smoothing_window=1, plot_dir=run_paths.plot_dir)
    plot_results.plot_budget_terms_hourly(result, smoothing_window=24, plot_dir=run_paths.plot_dir)
    plot_results.plot_budget_terms_day_bin(result, plot_dir=run_paths.plot_dir)

  
    # testing to see if a constant temperature field, yields a net heat advection error comparable to the estimated advection error from mass continuity (delta_mass * T_scale)

    os.makedirs(run_paths.plot_dir+'/constant_T', exist_ok=True)

    # ds_domain_test = ds_domain.copy(deep=True)
    # ds_domain_test['T'] = xr.full_like(ds_domain['T'], result.T_scale)

    ds_domain_test = ds_domain.assign(
        T = xr.full_like(ds_domain['T'], result.T_scale)
    )

    # ds_halo_test = ds_halo.copy(deep=True)
    # ds_halo_test['T'] = xr.full_like(ds_halo['T'], result.T_scale)

    ds_halo_test = ds_halo.assign(
        T = xr.full_like(ds_halo['T'], result.T_scale)
    )


    result_test = budget.calculate_budget(
        ds_domain_test, 
        ds_halo_test, 
        DomainSpecs, 
        SurfaceSpecs,
        integral_diagnostics_flag=True, 
        plot_dir=run_paths.plot_dir+'/constant_T', 
        plot_flag=True, 
        test_constant_T=True
    )
    
    plot_results.plot_budget_terms_day_bin(result_test, plot_dir=run_paths.plot_dir+'/constant_T')
    plot_results.plot_constant_T_results(result, result_test, plot_dir=run_paths.plot_dir+'/constant_T')

if __name__ == "__main__":
    
    logging.getLogger("distributed.shuffle._scheduler_plugin").setLevel(logging.ERROR)
    logging.getLogger("distributed.shuffle._core").setLevel(logging.ERROR)  

    client = Client(
        n_workers=4,
        threads_per_worker=1,
        processes=True,
        memory_limit="8GB",
    )

    print(client)
    print("Dashboard:", client.dashboard_link)

    main()

    
