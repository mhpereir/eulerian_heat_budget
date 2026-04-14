"""
Quick utility to check the maximum planetary boundary layer height
from ARCO ERA5 over a given year and bounding box.

Use this to inform what DEFAULT_ZG_TOP_PA should be set to —
the top of the control volume should be above the PBL to avoid
cutting through it.

Usage:
    mamba run -n dev_env python scripts/check_pbl.py \
        --year 1940 \
        --bbox 40 60 -130 -110
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from src import config


def _is_transient_arco_open_error(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        class_name = type(current).__name__

        if (
            "temporary failure in name resolution" in message
            or "cannot connect to host" in message
            or "name or service not known" in message
            or "connection reset by peer" in message
            or "service unavailable" in message
            or "timed out" in message
            or class_name in {
                "ClientConnectorDNSError",
                "ClientConnectorError",
                "ServerDisconnectedError",
                "TimeoutError",
            }
        ):
            return True

        current = current.__cause__ or current.__context__

    return False


def _open_arco_zarr_with_retry(
    arco_path: str = config.DEFAULT_ARCO_PATH,
    token: str = config.DEFAULT_ARCO_TOKEN,
    max_attempts: int = config.DEFAULT_ARCO_OPEN_MAX_ATTEMPTS,
    base_delay_seconds: float = config.DEFAULT_ARCO_OPEN_RETRY_BASE_DELAY_SECONDS,
) -> xr.Dataset:
    for attempt in range(1, max_attempts + 1):
        try:
            return xr.open_zarr(
                arco_path,
                storage_options={"token": token},
                decode_timedelta=False,
            )
        except Exception as exc:
            if not _is_transient_arco_open_error(exc) or attempt == max_attempts:
                raise

            delay_seconds = base_delay_seconds * (2 ** (attempt - 1))
            print(
                f"ARCO open_zarr attempt {attempt}/{max_attempts} failed with a transient error: {exc}. "
                f"Retrying in {delay_seconds:.0f} seconds..."
            )
            time.sleep(delay_seconds)

    raise RuntimeError("ARCO retry loop exhausted unexpectedly.")


def _pressure_at_height(
    z_target: np.ndarray,
    Z_chunk: np.ndarray,
    lnp_vals: np.ndarray,
) -> np.ndarray:
    """Interpolate pressure at a local target height field for one time chunk."""
    n_time, n_lev, n_lat, n_lon = Z_chunk.shape
    z_target = np.asarray(z_target, dtype=float)
    expected_shape = (n_time, n_lat, n_lon)
    if z_target.shape != expected_shape:
        raise ValueError(
            f"z_target must have shape {expected_shape}, got {z_target.shape}"
        )

    Z_flat = Z_chunk.reshape(n_time, n_lev, -1)
    z_target_flat = z_target.reshape(n_time, -1)
    n_col = Z_flat.shape[2]
    result = np.full((n_time, n_col), np.nan)

    for t in range(n_time):
        for c in range(n_col):
            z_col = Z_flat[t, :, c]
            z_here = z_target_flat[t, c]
            if np.isnan(z_here) or np.all(np.isnan(z_col)):
                continue
            result[t, c] = np.interp(
                z_here,
                z_col,
                lnp_vals,
                left=np.nan,
                right=np.nan,
            )

    return np.exp(result.reshape(n_time, n_lat, n_lon))


def _compute_domain_mean_pbl_top_pressure(
    Z_flipped: xr.DataArray,
    pbl: xr.DataArray,
    lnp_vals: np.ndarray,
    chunk_size: int = 200,
    progress: bool = True,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return domain-mean series, minimum local pressure, and full local pressure field."""
    n_times = Z_flipped.sizes["time"]
    p_domain_mean_ts_chunks: list[np.ndarray] = []
    p_field_chunks: list[np.ndarray] = []
    p_min_anywhere = np.inf

    for i_start in range(0, n_times, chunk_size):
        i_end = min(i_start + chunk_size, n_times)
        if progress:
            print(f"  Processing timesteps {i_start}–{i_end-1} of {n_times}...")
        Z_chunk = Z_flipped.isel(time=slice(i_start, i_end)).values
        pbl_chunk = pbl.isel(time=slice(i_start, i_end)).values
        p_field = _pressure_at_height(pbl_chunk, Z_chunk, lnp_vals)
        p_domain_mean_ts_chunks.append(np.nanmean(p_field, axis=(1, 2)))
        p_field_chunks.append(p_field.astype(np.float32, copy=False))

        if np.isfinite(p_field).any():
            p_min_anywhere = min(p_min_anywhere, float(np.nanmin(p_field)))

    return (
        np.concatenate(p_domain_mean_ts_chunks),
        p_min_anywhere,
        np.concatenate(p_field_chunks, axis=0),
    )


def _summarize_pressure_series(p_series: np.ndarray) -> dict[str, float]:
    """Summarize a pressure time series using the low-pressure tail."""
    return {
        "min": float(np.nanmin(p_series)),
        "p01": float(np.nanpercentile(p_series, 1)),
        "p05": float(np.nanpercentile(p_series, 5)),
        "mean": float(np.nanmean(p_series)),
    }


def _summarize_spatial_pressure_fields(
    p_field_time_series: np.ndarray,
) -> dict[str, np.ndarray]:
    """Summarize per-gridpoint pressure time series using low-pressure tail metrics."""
    return {
        "min": np.nanmin(p_field_time_series, axis=0),
        "p01": np.nanpercentile(p_field_time_series, 1, axis=0),
        "p05": np.nanpercentile(p_field_time_series, 5, axis=0),
    }


def _plot_spatial_pressure_metrics(
    pressure_maps: dict[str, xr.DataArray],
    plot_dir: Path,
    title_prefix: str,
) -> list[Path]:
    """Save one plot per spatial metric and return the output paths."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    finite_values = [
        np.asarray(field.values / 100.0)[np.isfinite(field.values)]
        for field in pressure_maps.values()
    ]
    finite_values = [vals for vals in finite_values if vals.size > 0]
    if finite_values:
        combined = np.concatenate(finite_values)
        vmin = float(combined.min())
        vmax = float(combined.max())
    else:
        vmin, vmax = 0.0, 1.0

    output_paths: list[Path] = []
    for metric_name, field in pressure_maps.items():
        fig, ax = plt.subplots(figsize=(9, 7), tight_layout=True)
        mesh = ax.pcolormesh(
            field["lon"].values,
            field["lat"].values,
            field.values / 100.0,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{title_prefix} {metric_name.upper()}")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("PBL-top pressure [hPa]")

        out_path = plot_dir / f"pbl_top_pressure_{metric_name}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out_path)

    return output_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check max PBL height from ARCO ERA5")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=list(config.DEFAULT_BBOX),
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("/home/mhpereir/eulerian_heat_budget/results/plots/check_pbl"),
        help="Base directory where year-specific check_pbl outputs will be written.",
    )
    return parser


def _write_run_info(payload: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    time_start = f"{args.year}-01-01"
    time_end = f"{args.year}-12-31"
    lat_min, lat_max, lon_min, lon_max = args.bbox
    output_dir = args.plot_dir / str(args.year)
    run_info_path = output_dir / "run_info.json"

    months_selection = [6, 7, 8]

    print(f"Opening ARCO ERA5 store: {config.DEFAULT_ARCO_PATH}")
    ds_full = _open_arco_zarr_with_retry()

    # Subset to only the variables we need BEFORE any coordinate work,
    # to avoid building/rearranging dask graphs for all ~200 ARCO variables.
    ds = ds_full[["boundary_layer_height", "geopotential"]]

    # Standardize coordinate names for slicing
    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if "valid_time" in ds.coords:
        rename["valid_time"] = "time"
    if "pressure_level" in ds.coords:
        rename["pressure_level"] = "level"
    if rename:
        ds = ds.rename(rename)

    # Normalise longitudes to [-180, 180]
    if float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360 - 180)).sortby("lon")

    pbl = ds["boundary_layer_height"]
    pbl = pbl.sel(
        time=slice(time_start, time_end),
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
    )
    pbl = pbl.sel(time=pbl.time.dt.month.isin(months_selection))

    print(f"Selected year: {args.year}")
    print(f"Months included: {months_selection}")
    print(f"Bbox: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]")
    print(f"Grid points: time={pbl.sizes.get('time', '?')}, lat={pbl.sizes.get('lat', '?')}, lon={pbl.sizes.get('lon', '?')}")
    print("Computing statistics (this may take a moment)...")

    pbl_max = float(pbl.max().compute())
    pbl_p99 = float(pbl.quantile(0.99).compute())
    pbl_p95 = float(pbl.quantile(0.95).compute())
    pbl_mean = float(pbl.mean().compute())

    # --- Estimate pressure at PBL top using ERA5 geopotential ---
    # geopotential is Φ [m²/s²]; geopotential height Z = Φ / g [m]
    print("Loading geopotential field to estimate pressure at PBL top...")
    geo = ds["geopotential"]
    # Ensure level is in Pa (ARCO stores pressure_level in hPa)
    if float(geo.coords["level"].max()) < 2000:
        geo = geo.assign_coords(level=geo.coords["level"] * 100.0)
    geo = geo.sel(
        time=slice(time_start, time_end),
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
    )
    geo = geo.sel(time=geo.time.dt.month.isin(months_selection))
    # Convert to geopotential height [m]
    Z = geo / config.g  # Φ/g, using project constant

    # pressure coordinate (Pa) — broadcast to match Z shape
    p_levels = Z.coords["level"]

    # For each time and grid point, interpolate pressure at the local PBL height
    # Z decreases with increasing pressure, so we interpolate p(Z) at z=PBL
    # Use log-pressure interpolation for accuracy: interp ln(p) as function of Z
    ln_p = np.log(p_levels)

    # Flip level axis once so Z is ascending (needed for np.interp)
    Z_flipped = Z.isel(level=slice(None, None, -1))
    ln_p_flipped = ln_p.isel(level=slice(None, None, -1))
    lnp_vals = ln_p_flipped.values

    # Process in time chunks to limit memory usage
    chunk_size = 200  # timesteps per chunk
    p_domain_mean_ts, p_min_anywhere, p_field_time_series = _compute_domain_mean_pbl_top_pressure(
        Z_flipped=Z_flipped,
        pbl=pbl,
        lnp_vals=lnp_vals,
        chunk_size=chunk_size,
        progress=True,
    )
    p_summary = _summarize_pressure_series(p_domain_mean_ts)
    p_spatial_maps = _summarize_spatial_pressure_fields(p_field_time_series)
    recommended_zg_top_pa = float(np.floor(p_min_anywhere / 100) * 100)

    plot_fields = {
        metric_name: xr.DataArray(
            values,
            coords={"lat": pbl.coords["lat"], "lon": pbl.coords["lon"]},
            dims=("lat", "lon"),
            name=f"pbl_top_pressure_{metric_name}",
            attrs={"units": "Pa"},
        )
        for metric_name, values in p_spatial_maps.items()
    }
    plot_paths = _plot_spatial_pressure_metrics(
        pressure_maps=plot_fields,
        plot_dir=output_dir,
        title_prefix="PBL-top pressure",
    )

    payload = {
        "year": args.year,
        "months_included": months_selection,
        "bbox": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        "grid_points": {
            "time": int(pbl.sizes.get("time", 0)),
            "lat": int(pbl.sizes.get("lat", 0)),
            "lon": int(pbl.sizes.get("lon", 0)),
        },
        "pbl_height_stats_m": {
            "max": pbl_max,
            "p99": pbl_p99,
            "p95": pbl_p95,
            "mean": pbl_mean,
        },
        "domain_mean_pbl_top_pressure_pa": p_summary,
        "recommendation": {
            "default_zg_top_pa": recommended_zg_top_pa,
            "lowest_local_pbl_top_pressure_pa": p_min_anywhere,
        },
        "artifacts": {
            "output_dir": str(output_dir.resolve()),
            "run_info_json": str(run_info_path.resolve()),
            "plots": {path.stem.removeprefix("pbl_top_pressure_"): str(path.resolve()) for path in plot_paths},
        },
    }
    _write_run_info(payload, run_info_path)

    print("Outputs written:")
    print(f"  Directory: {output_dir.resolve()}")
    print(f"  JSON: {run_info_path.resolve()}")
    for plot_path in plot_paths:
        print(f"  Plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
