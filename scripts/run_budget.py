import sys
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import xarray as xr

from src import config, cli, specs, io, validate, grid, budget, run_outputs
from src import plot_results

import logging
from dask.distributed import Client
    

from src import terms


@dataclass(frozen=True)
class ProductionOptions:
    output_dir: str
    init_manifest: bool
    start_year: int | None
    end_year: int | None
    year: int | None
    overwrite_output: bool

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


def build_data_source_from_cli(
    args,
    *,
    use_default_time_window: bool = True,
) -> specs.DataSourceConfig:
    time_start = args.time_start
    time_end = args.time_end
    if use_default_time_window:
        time_start = time_start if time_start is not None else config.DEFAULT_TIME_START
        time_end = time_end if time_end is not None else config.DEFAULT_TIME_END

    if args.data_source == "local_era5":
        return specs.DataSourceConfig(
            kind="local_era5",
            path_data=config.DEFAULT_LOCAL_PATH,
            time_start=time_start,
            time_end=time_end,
        )
    elif args.data_source == "arco_era5":
        return specs.DataSourceConfig(
            kind="arco_era5",
            arco_path=config.DEFAULT_ARCO_PATH,
            time_start=time_start,
            time_end=time_end,
        )
    else:
        raise ValueError(f"Unsupported data source: {args.data_source}")


def build_runtime_controls_from_cli(args) -> tuple[bool, bool]:
    diagnostic_plots = (
        args.diagnostic_plots
        if args.diagnostic_plots is not None
        else config.DEFAULT_DIAGNOSTIC_PLOTS
    )
    constant_temperature_test = (
        args.constant_temperature_test
        if args.constant_temperature_test is not None
        else config.DEFAULT_CONSTANT_TEMPERATURE_TEST
    )
    return diagnostic_plots, constant_temperature_test


def build_production_options_from_cli(args) -> ProductionOptions | None:
    if args.production_output_dir is None:
        if (
            args.init_production_manifest
            or args.production_start_year is not None
            or args.production_end_year is not None
            or args.overwrite_output
        ):
            raise ValueError("Production CLI arguments require --production-output-dir.")
        return None

    if args.init_production_manifest:
        if args.production_start_year is None or args.production_end_year is None:
            raise ValueError(
                "--init-production-manifest requires --production-start-year and --production-end-year."
            )
        if args.production_start_year > args.production_end_year:
            raise ValueError("--production-start-year cannot exceed --production-end-year.")
        return ProductionOptions(
            output_dir=args.production_output_dir,
            init_manifest=True,
            start_year=args.production_start_year,
            end_year=args.production_end_year,
            year=None,
            overwrite_output=args.overwrite_output,
        )

    if args.production_start_year is not None or args.production_end_year is not None:
        raise ValueError(
            "--production-start-year and --production-end-year are only valid with --init-production-manifest."
        )

    return ProductionOptions(
        output_dir=args.production_output_dir,
        init_manifest=False,
        start_year=None,
        end_year=None,
        year=run_outputs.resolve_production_year(
            time_start=args.time_start,
            time_end=args.time_end,
        ),
        overwrite_output=args.overwrite_output,
    )

def main() -> None:
    args = cli.parse_args()
    request = build_request_from_cli(args)
    SurfaceSpecs = build_surface_behaviour_from_cli(args)
    diagnostic_plots, constant_temperature_test = build_runtime_controls_from_cli(args)
    production_options = build_production_options_from_cli(args)
    git_provenance = run_outputs.resolve_git_provenance(PROJECT_ROOT)
    ad_hoc_run_paths: run_outputs.RunPaths | None = None
    production_paths: run_outputs.ProductionPaths | None = None
    plot_dir: str
    yearly_output_path: str | None = None

    if production_options is not None and production_options.init_manifest:
        SourceCfg = build_data_source_from_cli(args, use_default_time_window=False)
        production_paths = run_outputs.prepare_production_paths(production_options.output_dir)
        manifest_path = run_outputs.write_production_manifest(
            production_paths,
            production_start_year=production_options.start_year,  # type: ignore[arg-type]
            production_end_year=production_options.end_year,  # type: ignore[arg-type]
            request=request,
            source_spec=SourceCfg,
            surface_behaviour=SurfaceSpecs,
            git_provenance=git_provenance,
            cli_args=vars(args),
        )
        print(f"Saved production manifest to {manifest_path}")
        return

    SourceCfg = build_data_source_from_cli(args)

    if production_options is not None:
        production_paths = run_outputs.prepare_production_paths(
            production_options.output_dir,
            year=production_options.year,
        )
        manifest_path = run_outputs.require_production_manifest(production_paths)
        yearly_output_path = run_outputs.require_output_path(
            production_paths.output_path,
            overwrite=production_options.overwrite_output,
        )
        if production_paths.plot_dir is None:
            raise ValueError("Production yearly runs require a year-specific plot directory.")
        plot_dir = production_paths.plot_dir
        print(f"Using production manifest {manifest_path}")
        print(f"Saving yearly output to {yearly_output_path}")
        print(f"Saving plots to {plot_dir}")
    else:
        ad_hoc_run_paths = run_outputs.prepare_run_paths(config.DEFAULT_PLOTS_OUTPUT)
        plot_dir = ad_hoc_run_paths.plot_dir
        print(f"Saving plots to {plot_dir}")

    ds_merged = io.load_dataset(SourceCfg, SurfaceSpecs)
    
    # Validate merged dataset against strict schema
    validate.validate_schema(ds_merged)

    ds_bench = None
    if SourceCfg.kind == "arco_era5": #only available for arco era5 for now.
        benchmark_var_map = {
            "vertical_integral_of_eastward_heat_flux":  "Fx_heat",
            "vertical_integral_of_northward_heat_flux": "Fy_heat",
            "vertical_integral_of_eastward_mass_flux":  "Fx_mass",
            "vertical_integral_of_northward_mass_flux": "Fy_mass",
        }
        ds_bench = io.load_arco_benchmark_fluxes(SourceCfg, benchmark_var_map)



    # print(ds_bench)
    # print(ds_bench['Fx_mass'].sel(lon=360-130, lat=50, method='nearest').values) #type: ignore
    # print(ds_bench['Fx_heat'].sel(lon=360-130, lat=50, method='nearest').values) #type: ignore


    # Determine domain extent based on grid and config margin
    ds_domain, ds_halo, DomainSpecs = grid.determine_domain(ds_merged, request, eager_loading=True)

    print('Proceeding with', DomainSpecs)

    if production_options is None:
        if ad_hoc_run_paths is None:
            raise ValueError("Ad hoc runs require resolved run paths.")
        metadata_path = run_outputs.write_run_info(
            ad_hoc_run_paths,
            request=request,
            source_spec=SourceCfg,
            domain_spec=DomainSpecs,
            surface_behaviour=SurfaceSpecs,
            git_provenance=git_provenance,
            cli_args=vars(args),
        )
        print(f"Saved run metadata to {metadata_path}")

    result = budget.calculate_budget(
        ds_domain,
        ds_halo,
        DomainSpecs,
        SurfaceSpecs,
        integral_diagnostics_flag=True,
        plot_dir=plot_dir,
        plot_flag=diagnostic_plots,
        benchmark_ds=ds_bench
    ) #already computed before returning

    if production_options is not None:
        if yearly_output_path is None:
            raise ValueError("Production yearly runs require a resolved output path.")
        yearly_output_path = run_outputs.write_budget_result(
            result,
            yearly_output_path,
            overwrite=False,
        )
        print(f"Saved yearly output to {yearly_output_path}")

    if diagnostic_plots:
        plot_results.plot_budget_terms_hourly(result, smoothing_window=1, plot_dir=plot_dir)
        plot_results.plot_budget_terms_hourly(result, smoothing_window=24, plot_dir=plot_dir)
        plot_results.plot_budget_terms_day_bin(result, plot_dir=plot_dir)

    if constant_temperature_test:
        # testing to see if a constant temperature field, yields a net heat advection error comparable to the estimated advection error from mass continuity (delta_mass * T_scale)
        constant_t_plot_dir = plot_dir + '/constant_T'
        if diagnostic_plots:
            os.makedirs(constant_t_plot_dir, exist_ok=True)

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
            plot_dir=constant_t_plot_dir,
            plot_flag=diagnostic_plots,
            test_constant_T=True
        )

        if diagnostic_plots:
            plot_results.plot_budget_terms_day_bin(result_test, plot_dir=constant_t_plot_dir)
            plot_results.plot_constant_T_results(result, result_test, plot_dir=constant_t_plot_dir)

if __name__ == "__main__":
    
    logging.getLogger("distributed.shuffle._scheduler_plugin").setLevel(logging.ERROR)
    logging.getLogger("distributed.shuffle._core").setLevel(logging.ERROR)  

    client = Client(
        n_workers=4,
        threads_per_worker=1,
        processes=True,
        memory_limit="8GB",
    )

    # print(client)
    # print("Dashboard:", client.dashboard_link)

    main()

    
