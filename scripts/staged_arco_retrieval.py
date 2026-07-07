import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import run_budget
from src import cli, config, io, specs
from src_arco import cache, variables


def parse_args():
    parser = cli.build_arg_parser()
    parser.prog = "staged_arco_retrieval"
    parser.description = "Populate an indexed local ARCO ERA5 cache for offline budget runs."
    parser.add_argument(
        "--run-start-month-day",
        dest="run_start_month_day",
        type=str,
        default="05-01",
        help="Campaign season start month/day used with --production-start-year.",
    )
    parser.add_argument(
        "--run-end-month-day",
        dest="run_end_month_day",
        type=str,
        default="10-31",
        help="Campaign season end month/day used with --production-end-year.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_source not in (None, "arco_era5"):
        raise ValueError("staged_arco_retrieval.py can only retrieve from ARCO ERA5.")
    if args.staged_cache_root is None:
        raise ValueError("staged_arco_retrieval.py requires --staged-cache-root.")
    surface_variables_requested = (
        args.use_surface_variables
        if args.use_surface_variables is not None
        else config.DEFAULT_USE_SURFACE_VARIABLES
    )
    if surface_variables_requested:
        raise NotImplementedError(variables.SURFACE_VARIABLE_ERROR)

    request = run_budget.build_request_from_cli(args)
    surface_specs = run_budget.build_surface_behaviour_from_cli(args)
    variables.require_no_surface_variables(surface_specs)

    cache_root = cache.init_cache_root(args.staged_cache_root)
    print(f"[info] staged ARCO cache root: {cache_root}")

    for time_start, time_end in _iter_time_windows(args):
        source_cfg = specs.DataSourceConfig(
            kind="arco_era5",
            arco_path=config.DEFAULT_ARCO_PATH,
            time_start=time_start,
            time_end=time_end,
        )

        if cache.cache_has_coverage(
            cache_root,
            source_cfg,
            request,
            include_benchmark_variables=args.include_benchmark_variables,
        ):
            print(f"[info] cache already covers {time_start} to {time_end}; skipping")
            continue

        print(f"[info] staging ARCO ERA5 {time_start} to {time_end}")
        tile_path, tile = _stage_window_with_retry(
            cache_root,
            source_cfg,
            request,
            include_benchmark_variables=args.include_benchmark_variables,
        )
        print(
            "[info] wrote staged tile "
            f"{tile_path} sizes={dict(tile.sizes)} variables={list(tile.data_vars)}"
        )


def _stage_window_with_retry(
    cache_root: Path,
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    *,
    include_benchmark_variables: bool,
):
    max_attempts = config.DEFAULT_ARCO_OPEN_MAX_ATTEMPTS
    base_delay_seconds = config.DEFAULT_ARCO_OPEN_RETRY_BASE_DELAY_SECONDS

    for attempt in range(1, max_attempts + 1):
        try:
            tile = _build_tile_from_arco(
                source_cfg,
                request,
                include_benchmark_variables=include_benchmark_variables,
            )
            tile_path = cache.write_tile(
                cache_root,
                tile,
                source_cfg,
                request,
                include_benchmark_variables=include_benchmark_variables,
            )
            return tile_path, tile
        except Exception as exc:
            if not io._is_transient_arco_open_error(exc) or attempt == max_attempts:
                raise

            delay_seconds = base_delay_seconds * (2 ** (attempt - 1))
            print(
                f"ARCO stage/write attempt {attempt}/{max_attempts} failed with a transient error: {exc}. "
                f"Retrying in {delay_seconds:.0f} seconds..."
            )
            time.sleep(delay_seconds)

    raise RuntimeError("ARCO stage/write retry loop exhausted unexpectedly.")


def _iter_time_windows(args):
    if args.production_start_year is not None or args.production_end_year is not None:
        if args.production_start_year is None or args.production_end_year is None:
            raise ValueError(
                "Campaign staging requires both --production-start-year and --production-end-year."
            )
        if args.production_start_year > args.production_end_year:
            raise ValueError("--production-start-year cannot exceed --production-end-year.")

        for year in range(args.production_start_year, args.production_end_year + 1):
            yield (
                f"{year:04d}-{args.run_start_month_day}T00:00:00",
                f"{year:04d}-{args.run_end_month_day}T23:00:00",
            )
        return

    yield (
        args.time_start if args.time_start is not None else config.DEFAULT_TIME_START,
        args.time_end if args.time_end is not None else config.DEFAULT_TIME_END,
    )


def _build_tile_from_arco(
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    *,
    include_benchmark_variables: bool,
):
    var_map = dict(variables.ARCO_CORE_VAR_MAP)
    if include_benchmark_variables:
        var_map.update(variables.ARCO_BENCHMARK_VAR_MAP)

    ds = io._open_arco_zarr_with_retry(source_cfg)
    ds = ds[list(var_map.keys())]
    if source_cfg.time_start is not None or source_cfg.time_end is not None:
        ds = ds.sel(time=slice(source_cfg.time_start, source_cfg.time_end))
    ds = ds.rename(var_map)
    ds = io.standardize_era5_dataset(ds, source_cfg)
    return cache.build_arco_cache_tile(
        ds,
        request,
        include_benchmark_variables=include_benchmark_variables,
    )


def run() -> None:
    run_budget.configure_dask_runtime()
    main()


if __name__ == "__main__":
    run()
