from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import multiprocessing
from multiprocessing.connection import Connection
from pathlib import Path
import signal
import sys
import time
import traceback

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import run_budget
from src import cli, config, io, specs, terms
from src_arco import cache, variables


CHILD_EXIT_GRACE_SECONDS = 10.0
CHILD_POLL_SECONDS = 0.25


class _ChildStageAttemptError(RuntimeError):
    def __init__(
        self,
        error_type: str,
        message: str,
        remote_traceback: str,
        *,
        transient: bool,
    ):
        self.error_type = error_type
        self.message = message
        self.remote_traceback = remote_traceback
        self.transient = transient
        super().__init__(f"{error_type}: {message}\nChild traceback:\n{remote_traceback}")


class _UnexpectedChildExit(RuntimeError):
    pass


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
    parser.add_argument(
        "--stage-attempt-timeout-seconds",
        dest="stage_attempt_timeout_seconds",
        type=float,
        default=None,
        help=(
            "Wall-clock timeout for one ARCO stage/write attempt. "
            "Use 0 to disable the timeout."
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
    if args.include_benchmark_variables:
        terms.require_full_column_benchmark_domain(request)
    surface_specs = run_budget.build_surface_behaviour_from_cli(args)
    variables.require_no_surface_variables(surface_specs)

    cache_root = cache.init_cache_root(args.staged_cache_root)
    _log_info(f"staged ARCO cache root: {cache_root}")

    stage_time_chunk = getattr(args, "stage_time_chunk", "month")
    attempt_timeout_seconds = getattr(args, "stage_attempt_timeout_seconds", None)
    if attempt_timeout_seconds is None:
        attempt_timeout_seconds = config.DEFAULT_ARCO_STAGE_ATTEMPT_TIMEOUT_SECONDS
    _log_info(
        "staging options: "
        f"stage_time_chunk={stage_time_chunk}, "
        f"stage_attempt_timeout_seconds={attempt_timeout_seconds}"
    )
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
            tile_path, metadata = _stage_window_with_retry(
                cache_root,
                source_cfg,
                request,
                include_benchmark_variables=args.include_benchmark_variables,
                attempt_timeout_seconds=attempt_timeout_seconds,
            )
            _log_info(
                f"wrote staged tile {tile_path} "
                f"sizes={dict(metadata.sizes)} variables={list(metadata.variables)}"
            )


def _stage_window_with_retry(
    cache_root: Path,
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    *,
    include_benchmark_variables: bool,
    attempt_timeout_seconds: float | None = None,
):
    max_attempts = config.DEFAULT_ARCO_OPEN_MAX_ATTEMPTS
    base_delay_seconds = config.DEFAULT_ARCO_OPEN_RETRY_BASE_DELAY_SECONDS
    if attempt_timeout_seconds is None:
        attempt_timeout_seconds = config.DEFAULT_ARCO_STAGE_ATTEMPT_TIMEOUT_SECONDS

    for attempt in range(1, max_attempts + 1):
        attempt_started = time.monotonic()
        try:
            _log_info(
                f"ARCO stage/write attempt {attempt}/{max_attempts} "
                f"for {source_cfg.time_start} to {source_cfg.time_end} "
                f"starting (timeout={attempt_timeout_seconds:.0f}s)"
            )
            tile_path, metadata = _run_stage_attempt_in_child(
                cache_root,
                source_cfg,
                request,
                include_benchmark_variables=include_benchmark_variables,
                timeout_seconds=attempt_timeout_seconds,
            )
            _log_info(
                f"ARCO stage/write attempt {attempt}/{max_attempts} completed "
                f"in {_elapsed_seconds(attempt_started)}"
            )
            return tile_path, metadata
        except Exception as exc:
            transient = (
                exc.transient
                if isinstance(exc, _ChildStageAttemptError)
                else io._is_transient_arco_open_error(exc)
            )
            error_summary = _attempt_error_summary(exc)
            if not transient or attempt == max_attempts:
                _log_info(
                    f"ARCO stage/write attempt {attempt}/{max_attempts} failed "
                    f"after {_elapsed_seconds(attempt_started)} without retry: "
                    f"{error_summary}"
                )
                raise

            delay_seconds = base_delay_seconds * (2 ** (attempt - 1))
            _log_info(
                f"ARCO stage/write attempt {attempt}/{max_attempts} failed with a transient error: "
                f"{error_summary}. "
                f"Elapsed={_elapsed_seconds(attempt_started)}. Retrying in {delay_seconds:.0f} seconds..."
            )
            time.sleep(delay_seconds)

    raise RuntimeError("ARCO stage/write retry loop exhausted unexpectedly.")


def _attempt_error_summary(exc: Exception) -> str:
    if isinstance(exc, _ChildStageAttemptError):
        return f"{exc.error_type}: {exc.message}"
    return f"{type(exc).__name__}: {exc}"


def _run_stage_attempt_in_child(
    cache_root: Path,
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    *,
    include_benchmark_variables: bool,
    timeout_seconds: float | None,
    process_context=None,
    worker=None,
    child_exit_grace_seconds: float = CHILD_EXIT_GRACE_SECONDS,
) -> tuple[Path, cache._TileWriteMetadata]:
    # The child owns remote-backed computation only. The parent does not
    # promote or register its temporary store until that process is gone.
    context = process_context or multiprocessing.get_context("spawn")
    child_worker = worker or _stage_attempt_worker
    tmp_path = cache._create_temporary_tile_path(
        cache_root,
        source_cfg,
        request,
        include_benchmark_variables=include_benchmark_variables,
    )
    receive_conn, send_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=child_worker,
        args=(
            send_conn,
            str(cache_root),
            str(tmp_path),
            source_cfg,
            request,
            include_benchmark_variables,
        ),
        name=f"arco-stage-{source_cfg.time_start or 'unbounded'}",
    )

    try:
        with _sigterm_as_system_exit():
            process.start()
            send_conn.close()
            _log_info(f"started ARCO stage/write child pid={process.pid}")
            message = _wait_for_child_message(
                process,
                receive_conn,
                timeout_seconds=timeout_seconds,
            )
            _reap_child_after_message(
                process,
                grace_seconds=child_exit_grace_seconds,
            )

            status = message.get("status")
            if status == "error":
                raise _ChildStageAttemptError(
                    str(message.get("error_type", "Exception")),
                    str(message.get("message", "")),
                    str(message.get("traceback", "")),
                    transient=bool(message.get("transient", False)),
                )
            if status != "success" or not isinstance(message.get("metadata"), cache._TileWriteMetadata):
                raise RuntimeError(f"Invalid ARCO stage/write child response: {message!r}")

            metadata = message["metadata"]
            tile_path = cache._commit_temporary_tile(
                cache_root,
                tmp_path,
                metadata,
                source_cfg,
                request,
                include_benchmark_variables=include_benchmark_variables,
            )
            return tile_path, metadata
    finally:
        send_conn.close()
        receive_conn.close()
        if process.pid is not None:
            _terminate_and_reap_child(
                process,
                grace_seconds=child_exit_grace_seconds,
            )
        cache._remove_temporary_tile(
            tmp_path,
            reason="after child attempt cleanup",
        )


def _stage_attempt_worker(
    send_conn: Connection,
    cache_root: str,
    tmp_path: str,
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    include_benchmark_variables: bool,
) -> None:
    try:
        tile = _build_tile_from_arco(
            source_cfg,
            request,
            include_benchmark_variables=include_benchmark_variables,
        )
        _log_info(
            f"writing staged tile to local cache root {cache_root} "
            f"for {source_cfg.time_start} to {source_cfg.time_end}"
        )
        metadata = cache._write_temporary_tile(
            cache_root,
            tmp_path,
            tile,
            source_cfg,
            request,
            include_benchmark_variables=include_benchmark_variables,
        )
        send_conn.send({"status": "success", "metadata": metadata})
    except Exception as exc:
        send_conn.send(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "transient": io._is_transient_arco_open_error(exc),
            }
        )
    finally:
        send_conn.close()


def _wait_for_child_message(
    process,
    receive_conn: Connection,
    *,
    timeout_seconds: float | None,
) -> dict:
    deadline = (
        None
        if timeout_seconds is None or timeout_seconds <= 0
        else time.monotonic() + float(timeout_seconds)
    )

    while True:
        if deadline is None:
            poll_seconds = CHILD_POLL_SECONDS
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if receive_conn.poll(0):
                    return _receive_child_message(receive_conn, process=process)
                raise TimeoutError(
                    f"ARCO stage/write attempt exceeded {float(timeout_seconds):.0f} seconds"
                )
            poll_seconds = min(CHILD_POLL_SECONDS, remaining)

        if receive_conn.poll(poll_seconds):
            return _receive_child_message(receive_conn, process=process)

        if not process.is_alive():
            if receive_conn.poll(0):
                return _receive_child_message(receive_conn, process=process)
            process.join()
            raise _UnexpectedChildExit(
                f"ARCO stage/write child pid={process.pid} exited without a result "
                f"(exitcode={process.exitcode})"
            )


def _receive_child_message(receive_conn: Connection, *, process) -> dict:
    try:
        message = receive_conn.recv()
    except EOFError as exc:
        process.join()
        raise _UnexpectedChildExit(
            f"ARCO stage/write child pid={process.pid} closed its result pipe "
            f"without a message (exitcode={process.exitcode})"
        ) from exc
    if not isinstance(message, dict):
        raise RuntimeError(f"Invalid ARCO stage/write child message: {message!r}")
    return message


def _reap_child_after_message(process, *, grace_seconds: float) -> None:
    process.join(grace_seconds)
    if not process.is_alive():
        return
    _log_info(
        f"ARCO stage/write child pid={process.pid} did not exit within "
        f"{grace_seconds:.0f}s after reporting its result; terminating it"
    )
    _terminate_and_reap_child(process, grace_seconds=grace_seconds)


def _terminate_and_reap_child(process, *, grace_seconds: float) -> None:
    if not process.is_alive():
        process.join()
        return
    process.terminate()
    process.join(grace_seconds)
    if process.is_alive():
        _log_info(
            f"ARCO stage/write child pid={process.pid} did not terminate within "
            f"{grace_seconds:.0f}s; killing it"
        )
        process.kill()
        process.join()


@contextmanager
def _sigterm_as_system_exit():
    previous_handler = signal.getsignal(signal.SIGTERM)

    def _raise_system_exit(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _raise_system_exit)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


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
    # Do not construct a Dask graph for the complete multi-petabyte ARCO store.
    # The store has one full-globe chunk per variable and hour, so even native
    # Dask chunking creates enough graph metadata to exhaust a small VM before
    # any variable or regional selection is applied.
    ds = io._open_arco_zarr_with_retry(source_cfg, chunks=None)
    _log_info(f"opened ARCO zarr store in {_elapsed_seconds(build_started)}")
    _log_info(f"selecting ARCO source variables: {list(var_map)}")
    ds = ds[list(var_map.keys())]
    if source_cfg.time_start is not None or source_cfg.time_end is not None:
        _log_info(f"selecting ARCO time slice {source_cfg.time_start} to {source_cfg.time_end}")
        ds = ds.sel(time=slice(source_cfg.time_start, source_cfg.time_end))
    _log_info(f"renaming ARCO variables to staged schema: {list(var_map.values())}")
    ds = ds.rename(var_map)
    _log_info("standardizing ARCO dataset metadata and chunks")
    ds = io.standardize_era5_dataset(ds, source_cfg, rechunk=False)
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
    for variable_name in tuple(tile.data_vars):
        _log_info(f"materializing selected staged variable {variable_name}")
        tile[variable_name] = tile[variable_name].load()
    return tile


if __name__ == "__main__":
    main()
