import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import run_budget
from src import cli, config, io, specs
from src_arco import cache, variables


def _log_info(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[info] {timestamp} {message}", flush=True)


def _elapsed_seconds(start: float) -> str:
    return f"{time.monotonic() - start:.1f}s"


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
    parser.add_argument(
        "--stage-time-chunk",
        dest="stage_time_chunk",
        choices=("none", "day", "month"),
        default="month",
        help=(
            "Split each requested staging window into smaller ARCO fetch/write tiles. "
            "Use 'none' for the previous full-window behavior."
        ),
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
    _log_info(f"staged ARCO cache root: {cache_root}")

    stage_time_chunk = getattr(args, "stage_time_chunk", "month")
    _log_info(f"staging options: stage_time_chunk={stage_time_chunk}")
    for time_start, time_end in _iter_time_windows(args):
        chunk_windows = list(_iter_chunked_time_windows(time_start, time_end, stage_time_chunk))
        if len(chunk_windows) > 1:
            _log_info(
                f"staging window {time_start} to {time_end} "
                f"as {len(chunk_windows)} {stage_time_chunk} chunks"
            )
        for chunk_start, chunk_end in chunk_windows:
            source_cfg = specs.DataSourceConfig(
                kind="arco_era5",
                arco_path=config.DEFAULT_ARCO_PATH,
                time_start=chunk_start,
                time_end=chunk_end,
            )

            if cache.exact_tile_exists(
                cache_root,
                source_cfg,
                request,
                include_benchmark_variables=args.include_benchmark_variables,
            ):
                _log_info(f"exact staged tile already exists for {chunk_start} to {chunk_end}; skipping")
                continue

            _log_info(f"staging ARCO ERA5 {chunk_start} to {chunk_end}")
            tile_path, tile = _stage_window_with_retry(
                cache_root,
                source_cfg,
                request,
                include_benchmark_variables=args.include_benchmark_variables,
            )
            _log_info(
                f"wrote staged tile {tile_path} "
                f"sizes={dict(tile.sizes)} variables={list(tile.data_vars)}"
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
        attempt_started = time.monotonic()
        try:
            _log_info(
                f"ARCO stage/write attempt {attempt}/{max_attempts} "
                f"for {source_cfg.time_start} to {source_cfg.time_end} starting"
            )
            tile = _build_tile_from_arco(
                source_cfg,
                request,
                include_benchmark_variables=include_benchmark_variables,
            )
            _log_info(
                f"writing staged tile to local cache root {cache_root} "
                f"for {source_cfg.time_start} to {source_cfg.time_end}"
            )
            tile_path = cache.write_tile(
                cache_root,
                tile,
                source_cfg,
                request,
                include_benchmark_variables=include_benchmark_variables,
            )
            _log_info(
                f"ARCO stage/write attempt {attempt}/{max_attempts} completed "
                f"in {_elapsed_seconds(attempt_started)}"
            )
            return tile_path, tile
        except Exception as exc:
            if not io._is_transient_arco_open_error(exc) or attempt == max_attempts:
                _log_info(
                    f"ARCO stage/write attempt {attempt}/{max_attempts} failed "
                    f"after {_elapsed_seconds(attempt_started)} without retry: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise

            delay_seconds = base_delay_seconds * (2 ** (attempt - 1))
            _log_info(
                f"ARCO stage/write attempt {attempt}/{max_attempts} failed with a transient error: {exc}. "
                f"Elapsed={_elapsed_seconds(attempt_started)}. Retrying in {delay_seconds:.0f} seconds..."
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


def _iter_chunked_time_windows(time_start: str, time_end: str, chunk: str):
    if chunk == "none":
        yield time_start, time_end
        return
    if chunk not in {"day", "month"}:
        raise ValueError("--stage-time-chunk must be one of: none, day, month.")

    start = _parse_hourly_iso_time(time_start)
    end = _parse_hourly_iso_time(time_end)
    if start > end:
        raise ValueError(f"time_start={time_start!r} is after time_end={time_end!r}.")

    current = start
    while current <= end:
        boundary = _next_chunk_boundary(current, chunk)
        chunk_end = min(end, boundary - timedelta(hours=1))
        if chunk_end < current:
            raise ValueError(
                f"Could not build a non-empty {chunk} chunk starting at {current.isoformat()}."
            )
        yield _format_time(current), _format_time(chunk_end)
        current = chunk_end + timedelta(hours=1)


def _parse_hourly_iso_time(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    if parsed.minute != 0 or parsed.second != 0 or parsed.microsecond != 0:
        raise ValueError(
            "Staged ARCO time chunking expects hourly-aligned ISO times; "
            f"got {value!r}."
        )
    return parsed


def _next_chunk_boundary(current: datetime, chunk: str) -> datetime:
    if chunk == "day":
        return datetime(current.year, current.month, current.day) + timedelta(days=1)
    if current.month == 12:
        return datetime(current.year + 1, 1, 1)
    return datetime(current.year, current.month + 1, 1)


def _format_time(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _build_tile_from_arco(
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    *,
    include_benchmark_variables: bool,
):
    var_map = dict(variables.ARCO_CORE_VAR_MAP)
    if include_benchmark_variables:
        var_map.update(variables.ARCO_BENCHMARK_VAR_MAP)

    build_started = time.monotonic()
    _log_info(f"opening ARCO zarr store {source_cfg.arco_path}")
    ds = io._open_arco_zarr_with_retry(source_cfg)
    _log_info(f"opened ARCO zarr store in {_elapsed_seconds(build_started)}")
    _log_info(f"selecting ARCO source variables: {list(var_map)}")
    ds = ds[list(var_map.keys())]
    if source_cfg.time_start is not None or source_cfg.time_end is not None:
        _log_info(f"selecting ARCO time slice {source_cfg.time_start} to {source_cfg.time_end}")
        ds = ds.sel(time=slice(source_cfg.time_start, source_cfg.time_end))
    _log_info(f"renaming ARCO variables to staged schema: {list(var_map.values())}")
    ds = ds.rename(var_map)
    _log_info("standardizing ARCO dataset metadata and chunks")
    ds = io.standardize_era5_dataset(ds, source_cfg)
    _log_info("building wall-only staged tile metadata")
    tile = cache.build_arco_cache_tile(
        ds,
        request,
        include_benchmark_variables=include_benchmark_variables,
    )
    _log_info(
        f"built staged tile metadata in {_elapsed_seconds(build_started)} "
        f"sizes={dict(tile.sizes)} variables={list(tile.data_vars)}"
    )
    return tile


if __name__ == "__main__":
    main()
