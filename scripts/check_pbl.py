"""
Quick utility to check the maximum planetary boundary layer height
from ARCO ERA5 over a given time range and bounding box.

Use this to inform what DEFAULT_ZG_TOP_PA should be set to —
the top of the control volume should be above the PBL to avoid
cutting through it.

Usage:
    mamba run -n dev_env python scripts/check_pbl.py \
        --year-start 1940 --year-end 1949 \
        --bbox 40 60 -130 -110
"""

import argparse
import sys
import time

import numpy as np
import xarray as xr

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

def main() -> None:
    parser = argparse.ArgumentParser(description="Check max PBL height from ARCO ERA5")
    parser.add_argument("--year-start", type=int, required=True)
    parser.add_argument("--year-end", type=int, required=True)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=list(config.DEFAULT_BBOX),
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
    )
    args = parser.parse_args()

    time_start = f"{args.year_start}-01-01"
    time_end = f"{args.year_end}-12-31"
    lat_min, lat_max, lon_min, lon_max = args.bbox

    months_selection = [6,7,8]

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

    print(f"Year range : {args.year_start} to {args.year_end}")
    print(f"Months included: {months_selection}")
    print(f"Bbox       : lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]")
    print(f"Grid points: time={pbl.sizes.get('time', '?')}, lat={pbl.sizes.get('lat', '?')}, lon={pbl.sizes.get('lon', '?')}")
    print("Computing statistics (this may take a moment)...")

    pbl_max = float(pbl.max().compute())
    pbl_p99 = float(pbl.quantile(0.99).compute())
    pbl_p95 = float(pbl.quantile(0.95).compute())
    pbl_mean = float(pbl.mean().compute())

    print(f"\n--- PBL height statistics [m] ---")
    print(f"  Max  : {pbl_max:>10.1f} m")
    print(f"  P99  : {pbl_p99:>10.1f} m")
    print(f"  P95  : {pbl_p95:>10.1f} m")
    print(f"  Mean : {pbl_mean:>10.1f} m")

    # --- Estimate pressure at PBL top using ERA5 geopotential ---
    # geopotential is Φ [m²/s²]; geopotential height Z = Φ / g [m]
    print("\nLoading geopotential field to estimate pressure at PBL top...")
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

    # For each PBL height threshold, interpolate to find the pressure
    # Z decreases with increasing pressure, so we interpolate p(Z) at z=PBL
    # Use log-pressure interpolation for accuracy: interp ln(p) as function of Z
    ln_p = np.log(p_levels)

    # Flip level axis once so Z is ascending (needed for np.interp)
    Z_flipped = Z.isel(level=slice(None, None, -1))
    ln_p_flipped = ln_p.isel(level=slice(None, None, -1))
    lnp_vals = ln_p_flipped.values

    def pressure_at_height(z_target, Z_chunk):
        """Interpolate ln(p) at z_target for a single time chunk (numpy array)."""
        n_time, n_lev, n_lat, n_lon = Z_chunk.shape
        Z_flat = Z_chunk.reshape(n_time, n_lev, -1)
        n_col = Z_flat.shape[2]

        result = np.full((n_time, n_col), np.nan)
        for t in range(n_time):
            for c in range(n_col):
                z_col = Z_flat[t, :, c]
                if np.all(np.isnan(z_col)):
                    continue
                result[t, c] = np.interp(z_target, z_col, lnp_vals,
                                         left=np.nan, right=np.nan)
        return np.exp(result.reshape(n_time, n_lat, n_lon))

    # Process in time chunks to limit memory usage
    chunk_size = 200  # timesteps per chunk
    n_times = Z_flipped.sizes["time"]
    z_targets = {"max": pbl_max, "p99": pbl_p99, "p95": pbl_p95}
    p_min = {k: np.inf for k in z_targets}
    p_spatial_mean_chunks = {k: [] for k in z_targets}

    for i_start in range(0, n_times, chunk_size):
        i_end = min(i_start + chunk_size, n_times)
        print(f"  Processing timesteps {i_start}–{i_end-1} of {n_times}...")
        Z_chunk = Z_flipped.isel(time=slice(i_start, i_end)).values

        for key, z_target in z_targets.items():
            p_field = pressure_at_height(z_target, Z_chunk)
            p_min[key] = min(p_min[key], float(np.nanmin(p_field)))
            # Spatial mean for each timestep: (n_time,)
            p_spatial_mean_chunks[key].append(
                np.nanmean(p_field, axis=(1, 2))
            )

        del Z_chunk

    p_at_max = p_min["max"]
    p_at_p99 = p_min["p99"]
    p_at_p95 = p_min["p95"]

    print(f"\n--- Pressure at PBL top (from ERA5 geopotential) [Pa] ---")
    print(f"  At max PBL : {p_at_max:>10.0f} Pa  ({p_at_max/100:>7.1f} hPa)")
    print(f"  At P99 PBL : {p_at_p99:>10.0f} Pa  ({p_at_p99/100:>7.1f} hPa)")
    print(f"  At P95 PBL : {p_at_p95:>10.0f} Pa  ({p_at_p95/100:>7.1f} hPa)")

    # Spatially averaged pressure at PBL top — temporal statistics
    print(f"\n--- Pressure at PBL top (spatial mean, then temporal stats) [Pa] ---")
    for key in z_targets:
        ts = np.concatenate(p_spatial_mean_chunks[key])
        sm_min = float(np.nanmin(ts))
        sm_p99 = float(np.nanpercentile(ts, 99))
        sm_p95 = float(np.nanpercentile(ts, 95))
        sm_mean = float(np.nanmean(ts))
        label = key.upper().rjust(3)
        print(f"  At {label} PBL :  min={sm_min/100:>7.1f}  P01={sm_p99/100:>7.1f}  P05={sm_p95/100:>7.1f}  mean={sm_mean/100:>7.1f} hPa")

    print(f"\n--- Recommendation ---")
    print(f"  Set DEFAULT_ZG_TOP_PA to at most {round(p_at_max / 100) * 100:.0f} Pa")
    print(f"  (rounded down from {p_at_max:.0f} Pa at the max PBL height)")
    print(f"  This ensures the control volume top is above the PBL at all times.")


if __name__ == "__main__":
    main()
