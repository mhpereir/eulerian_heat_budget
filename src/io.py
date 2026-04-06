'''
Docstring for eulerian_heat_budget.src.io

Responsibilities:

- Load datasets (ERA5, model data, etc.)
- Harmonize variable names into the canonical internal schema
- Enforce pressure units (Pa) and consistent coordinate names
- Return standardized xarray.Dataset objects
- Should not perform analysis calculations (no integrals, no budgets).

Contract requirement: `io.py` is where any renaming between external conventions (ERA5 variable names) and internal names must happen (e.g., surface pressure → `sp`).

'''

import xarray as xr
import numpy as np
import time

from collections.abc import Mapping

from . import config, specs



def load_dataset(source_cfg: specs.DataSourceConfig, SurfaceSpecs: specs.SurfaceBehaviour) -> xr.Dataset:
    if source_cfg.kind == "local_era5":
        ds = _load_local_era5(source_cfg, SurfaceSpecs)
    elif source_cfg.kind == "arco_era5":
        ds = _load_arco_era5(source_cfg, SurfaceSpecs)
    else:
        raise ValueError(f"Unsupported data source: {source_cfg.kind}")
    ds = standardize_era5_dataset(ds, source_cfg)
    return ds


def _load_local_era5(cfg: specs.DataSourceConfig, SurfaceSpecs: specs.SurfaceBehaviour) -> xr.Dataset:

    # Example usage
    ds_T = load_era5_T(f"{cfg.path_data}/T.nc")
    ds_u = load_era5_u(f"{cfg.path_data}/ux.nc", 'u')
    ds_v = load_era5_u(f"{cfg.path_data}/uy.nc", 'v')
    ds_omega = load_era5_omega(f"{cfg.path_data}/uz.nc")
    ds_sp = load_era5_sp(f"{cfg.path_data}/sfp.nc") #surface pressure in Pa
    
    ds_sT = None
    ds_su = None
    ds_sv = None
    if SurfaceSpecs.use_surface_variables:
        ds_sT = load_era5_surface_T(f"{cfg.path_data}/surface_temperature.nc")
        ds_su = load_era5_surface_u(f"{cfg.path_data}/surface_ux.nc", 'u10')
        ds_sv = load_era5_surface_u(f"{cfg.path_data}/surface_uy.nc", 'v10')
    
    # Merge datasets on common coordinates and variables
    if SurfaceSpecs.use_surface_variables:
        ds_merged = load_era5_merge_dataset(ds_T=ds_T, ds_u=ds_u, 
                                            ds_v=ds_v, ds_w=ds_omega,
                                            ds_sp=ds_sp, cfg=cfg,
                                            ds_sT=ds_sT, ds_su=ds_su, ds_sv=ds_sv)
    else:
        ds_merged = load_era5_merge_dataset(ds_T=ds_T, ds_u=ds_u, 
                                            ds_v=ds_v, ds_w=ds_omega, 
                                            ds_sp=ds_sp, cfg=cfg)

    return ds_merged

def _load_arco_era5(cfg: specs.DataSourceConfig, SurfaceSpecs: specs.SurfaceBehaviour) -> xr.Dataset:
    ds = _open_arco_zarr_with_retry(cfg)

    var_map = {
        "temperature": "T",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
        "vertical_velocity": "w",
        "surface_pressure": "sp",
    }
    if SurfaceSpecs.use_surface_variables:
        var_map.update({
            "2m_temperature": "T2m",
            "10m_u_component_of_wind": "u10",
            "10m_v_component_of_wind": "v10",
        })

    ds = ds[list(var_map.keys())]

    if cfg.time_start is not None or cfg.time_end is not None:
        ds = ds.sel(time=slice(cfg.time_start, cfg.time_end))

    ds = ds.rename(var_map)

    return ds


def standardize_era5_dataset(ds: xr.Dataset, cfg: specs.DataSourceConfig) -> xr.Dataset:
    """
    Standardize ERA5-like datasets from either local NetCDF inputs or ARCO Zarr
    into the canonical internal schema expected by validate.py and downstream code.

    Canonical contract:
      - coords: time, level, lat, lon
      - 4D vars: (time, level, lat, lon)
      - 3D vars: (time, lat, lon)
      - level units: Pa
      - lat/lon ascending
      - level strictly descending
    """
    ds = ds.copy()

    # ------------------------------------------------------------------
    # 1) Rename common external coordinate names to canonical names
    # ------------------------------------------------------------------
    rename_map: dict[str, str] = {}

    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        rename_map["valid_time"] = "time"
    if "latitude" in ds.dims or "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.dims or "longitude" in ds.coords:
        rename_map["longitude"] = "lon"
    if "pressure_level" in ds.dims or "pressure_level" in ds.coords:
        rename_map["pressure_level"] = "level"
    if "isobaricInhPa" in ds.dims or "isobaricInhPa" in ds.coords:
        rename_map["isobaricInhPa"] = "level"

    if rename_map:
        ds = ds.rename(rename_map)

    # ------------------------------------------------------------------
    # 2) Drop auxiliary coords/vars that are not part of the project schema
    # ------------------------------------------------------------------
    drop_names = [
        name for name in ["number", "expver", "step", "surface"]
        if name in ds.coords or name in ds.variables
    ]
    if drop_names:
        ds = ds.drop_vars(drop_names, errors="ignore")

    # ------------------------------------------------------------------
    # 3) Require canonical coordinates to exist
    # ------------------------------------------------------------------
    required_coords = ["time", "lat", "lon"]
    missing_coords = [c for c in required_coords if c not in ds.coords]
    if missing_coords:
        raise ValueError(f"Dataset missing required coordinates after standardization: {missing_coords}")

    # level is required for 4D fields used by this project
    if "level" not in ds.coords:
        raise ValueError("Dataset missing required 'level' coordinate after standardization.")

    # ------------------------------------------------------------------
    # 4) Normalize coordinate attrs (avoid harmless merge/metadata conflicts)
    # ------------------------------------------------------------------
    ds["lat"].attrs = {
        "units": ds["lat"].attrs.get("units", "degrees_north"),
        "long_name": "latitude",
    }
    ds["lon"].attrs = {
        "units": ds["lon"].attrs.get("units", "degrees_east"),
        "long_name": "longitude",
    }
    ds["time"].attrs = {"long_name": "time"}

    # ------------------------------------------------------------------
    # 5) Sort coordinates into required monotonic direction
    # ------------------------------------------------------------------
    if "lat" in ds.coords and ds.sizes["lat"] > 1:
        lat_diff = ds["lat"].diff("lat")
        if not bool((lat_diff > 0).all()):
            ds = ds.sortby("lat")

    if "lon" in ds.coords and ds.sizes["lon"] > 1:
        lon_diff = ds["lon"].diff("lon")
        if not bool((lon_diff > 0).all()):
            ds = ds.sortby("lon")

    if "level" in ds.coords and ds.sizes["level"] > 1:
        level_diff = ds["level"].diff("level")
        if not bool((level_diff < 0).all()):
            ds = ds.sortby("level", ascending=False)

    # ------------------------------------------------------------------
    # 6) Convert pressure levels to Pa if they appear to be in hPa
    # ------------------------------------------------------------------
    level_units = ds["level"].attrs.get("units", "").strip().lower()
    level_max = float(ds["level"].max().values)

    level_in_hpa = (
        level_units in {"hpa", "hectopascal", "hectopascals", "millibar", "mbar"}
        or level_max < 2_000.0
    )
    if level_in_hpa:
        ds = ds.assign_coords(level=ds["level"] * 100.0)

    ds["level"].attrs["units"] = "Pa"

    # Re-sort after possible coordinate reassignment
    if ds.sizes["level"] > 1:
        level_diff = ds["level"].diff("level")
        if not bool((level_diff < 0).all()):
            ds = ds.sortby("level", ascending=False)

    # ------------------------------------------------------------------
    # 7) Force variable dim order to canonical schema
    # ------------------------------------------------------------------
    for name, da in list(ds.data_vars.items()):
        dims_set = set(da.dims)

        if dims_set == {"time", "level", "lat", "lon"}:
            if da.dims != ("time", "level", "lat", "lon"):
                ds[name] = da.transpose("time", "level", "lat", "lon")

        elif dims_set == {"time", "lat", "lon"}:
            if da.dims != ("time", "lat", "lon"):
                ds[name] = da.transpose("time", "lat", "lon")

        else:
            # keep only project-relevant vars; fail on anything unexpected if desired
            pass

    # ------------------------------------------------------------------
    # 8) Ensure essential variables are present and have minimal attrs
    # ------------------------------------------------------------------
    required_vars = ["T", "u", "v", "w", "sp"]
    missing_vars = [v for v in required_vars if v not in ds.data_vars]
    if missing_vars:
        raise ValueError(f"Dataset missing required variables after standardization: {missing_vars}")

    if ds["T"].attrs.get("units") in {"C", "celsius", "Celsius"}:
        ds["T"] = ds["T"] + 273.15
    ds["T"].attrs["units"] = "K"

    if "T2m" in ds and ds["T2m"].attrs.get("units") in {"C", "celsius", "Celsius"}:
        ds["T2m"] = ds["T2m"] + 273.15
    if "T2m" in ds:
        ds["T2m"].attrs["units"] = "K"


    # ------------------------------------------------------------------
    # 9) Ensure longitude values are in [-180, 180] range for consistency
    # ------------------------------------------------------------------
    if ds['lon'].max() > 180:
        ds['lon'] = (ds['lon'] + 180) % 360 - 180
        ds = ds.sortby("lon")

    # ------------------------------------------------------------------
    # 10) Apply time selection late, once schema is canonical
    # ------------------------------------------------------------------
    if cfg.time_start is not None or cfg.time_end is not None:
        ds = ds.sel(time=slice(cfg.time_start, cfg.time_end))

    # ------------------------------------------------------------------
    # 11) Chunk for dask workflows
    # ------------------------------------------------------------------
    chunk_map = dict(config.DEFAULT_CHUNKS_3D1)
    if cfg.kind == "arco_era5":
        chunk_map["time"] = cfg.chunks_time

    ds = ds.chunk(chunk_map)

    return ds




def load_era5_T(filepath: str) -> xr.Dataset:
    ds = xr.open_dataset(filepath)
    #check units are in Kelvin
    if 'units' in ds['t'].attrs:
        if ds['t'].attrs['units'] in ['K', 'kelvin', 'Kelvin']:
            pass  # already in Kelvin
        elif ds['t'].attrs['units'] in ['C', 'celsius', 'Celsius']:
            ds['t'] = ds['t'] + 273.15  # convert to Kelvin
            ds['t'].attrs['units'] = 'K'
        else:
            raise ValueError(f"Unexpected temperature units: {ds['t'].attrs['units']}")
    else:
        raise ValueError("Temperature variable 't' must have 'units' attribute.")

    return _standardize_surface_era5(ds, {'t': 'T'}) #[K]

def load_era5_u(filepath: str, varname: str) -> xr.Dataset:
    ds = xr.open_dataset(filepath)
    return _standardize_surface_era5(ds, {varname: varname}) #[m/s]

def load_era5_omega(filepath: str) -> xr.Dataset:
    ds = xr.open_dataset(filepath)
    return _standardize_surface_era5(ds, {'w': 'w'}) #[Pa/s]

def load_era5_sp(filepath: str) -> xr.Dataset:
    ds = xr.open_dataset(filepath)
    return _standardize_surface_era5(ds, {'sp': 'sp'}) #[Pa]

def load_era5_surface_u(filepath: str, varname: str) -> xr.Dataset: # u10, v10
    ds = xr.open_dataset(filepath)
    return _standardize_surface_era5(ds, {varname: varname}) #[m/s] 

def load_era5_surface_T(filepath: str) -> xr.Dataset: #t2m
    ds = xr.open_dataset(filepath)

    #check units are in Kelvin
    if 'units' in ds['t2m'].attrs:
        if ds['t2m'].attrs['units'] in ['K', 'kelvin', 'Kelvin']:
            pass  # already in Kelvin
        elif ds['t2m'].attrs['units'] in ['C', 'celsius', 'Celsius']:
            ds['t2m'] = ds['t2m'] + 273.15  # convert to Kelvin
            ds['t2m'].attrs['units'] = 'K'
        else:
            raise ValueError(f"Unexpected temperature units: {ds['t2m'].attrs['units']}")
    else:
        raise ValueError("Temperature variable 't2m' must have 'units' attribute.")

    return _standardize_surface_era5(ds, {'t2m': 'T2m'}) #[K]


def _standardize_surface_era5(ds: xr.Dataset, var_map: dict[str, str]) -> xr.Dataset:
    # rename dimensions if needed
    rename_map = {}
    if "valid_time" in ds.dims:
        rename_map["valid_time"] = "time"
    if "latitude" in ds.dims or "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.dims or "longitude" in ds.coords:
        rename_map["longitude"] = "lon"

    # rename variable too
    rename_map.update(var_map)
    ds = ds.rename(rename_map)

    # drop extra scalar/aux coords that often come from cfgrib
    drop_names = [name for name in ["number", "expver"] if name in ds.coords or name in ds.variables]
    if drop_names:
        ds = ds.drop_vars(drop_names, errors="ignore")

    # normalize coord attrs so merge does not fail on harmless metadata
    for c, std_name in [("lat", "latitude"), ("lon", "longitude")]:
        if c in ds.coords:
            ds[c].attrs = {
                "units": ds[c].attrs.get("units", "degrees_north" if c == "lat" else "degrees_east"),
                "long_name": std_name,
            }

    if "time" in ds.coords:
        ds["time"].attrs = {
            "long_name": "time",
        }

    return ds

def load_era5_merge_dataset(
    ds_T: xr.Dataset,
    ds_u: xr.Dataset,
    ds_v: xr.Dataset,
    ds_w: xr.Dataset,
    ds_sp: xr.Dataset,
    cfg: specs.DataSourceConfig,
    *,
    ds_sT: xr.Dataset | None = None,
    ds_su: xr.Dataset | None = None,
    ds_sv: xr.Dataset | None = None,
) -> xr.Dataset:
    datasets = [ds_T, ds_u, ds_v, ds_w, ds_sp]
    optional = [ds_sT, ds_su, ds_sv]

    datasets.extend(ds for ds in optional if ds is not None)
    return xr.merge(datasets, compat="identical")


def load_arco_benchmark_fluxes(
    cfg: specs.DataSourceConfig,
    variables: dict[str, str],
) -> xr.Dataset:
    ds = _open_arco_zarr_with_retry(cfg)

    ds = ds[list(variables.keys())]

    if cfg.time_start is not None or cfg.time_end is not None:
        ds = ds.sel(time=slice(cfg.time_start, cfg.time_end))

    ds = ds.rename(variables)

    # reuse only the parts of standardization that make sense for 3D single-level fields
    rename_map = {}
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        rename_map["valid_time"] = "time"
    if "latitude" in ds.dims or "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.dims or "longitude" in ds.coords:
        rename_map["longitude"] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)

    drop_names = [
        name for name in ["number", "expver", "step", "surface"]
        if name in ds.coords or name in ds.variables
    ]
    if drop_names:
        ds = ds.drop_vars(drop_names, errors="ignore")

    if ds["lat"].size > 1 and not bool((ds["lat"].diff("lat") > 0).all()):
        ds = ds.sortby("lat")
    if ds["lon"].size > 1 and not bool((ds["lon"].diff("lon") > 0).all()):
        ds = ds.sortby("lon")

    if ds["lon"].max() > 180:
        ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360 - 180)).sortby("lon")

    chunk_map = {
        "time": cfg.chunks_time,
        "lat": config.n_lat,
        "lon": config.n_lon,
    }
    ds = ds.chunk(chunk_map)

    return ds


def _open_arco_zarr_with_retry(cfg: specs.DataSourceConfig) -> xr.Dataset:
    max_attempts = config.DEFAULT_ARCO_OPEN_MAX_ATTEMPTS
    base_delay_seconds = config.DEFAULT_ARCO_OPEN_RETRY_BASE_DELAY_SECONDS

    for attempt in range(1, max_attempts + 1):
        try:
            return xr.open_zarr(
                cfg.arco_path,
                storage_options={"token": cfg.arco_storage_token},
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


'''
def load_era5_merge_dataset(
    ds_T: xr.Dataset,
    ds_u: xr.Dataset,
    ds_v: xr.Dataset,
    ds_w: xr.Dataset,
    ds_sp: xr.Dataset,
    cfg: specs.DataSourceConfig,
    *,
    ds_sT: xr.Dataset | None = None,
    ds_su: xr.Dataset | None = None,
    ds_sv: xr.Dataset | None = None,
) -> xr.Dataset:
    datasets = [ds_T, ds_u, ds_v, ds_w, ds_sp]

    optional = [ds_sT, ds_su, ds_sv]
    datasets.extend(ds for ds in optional if ds is not None)

    merged = xr.merge(datasets, compat="identical")

    # transpose only variables that actually have a level dimension
    for name, da in merged.data_vars.items():
        if da.dims == ("time", "level", "lat", "lon"):
            continue
        elif set(da.dims) == {"time", "level", "lat", "lon"}:
            merged[name] = da.transpose("time", "level", "lat", "lon")
        elif da.dims == ("time", "lat", "lon"):
            continue
        elif set(da.dims) == {"time", "lat", "lon"}:
            merged[name] = da.transpose("time", "lat", "lon")

    if not merged["lat"].diff("lat").min() > 0:
        merged = merged.sortby("lat")
    if not merged["lon"].diff("lon").min() > 0:
        merged = merged.sortby("lon")

    dlev = merged["level"].diff("level")
    if not ((dlev < 0).all() and merged["level"].to_index().is_unique):
        merged = merged.sortby("level", ascending=False)

    merged["level"] = merged["level"] * 100.0
    merged["level"].attrs["units"]  = "Pa"

    merged = merged.chunk(config.DEFAULT_CHUNKS_3D1)  # apply chunking for dask compatibility

    #hack to speed up calculation (cropped time range)
    merged = merged.sel(time=slice(cfg.time_start, cfg.time_end))

    return merged
'''
