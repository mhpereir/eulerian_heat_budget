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

from collections.abc import Mapping

from . import config

n_time:int = 24
n_lat :int = 32
n_lon :int = 32

DEFAULT_CHUNKS_3D1 = {
    "time": n_time,      # 1 day per chunk
    "level": -1,     # keep full vertical column together
    "lat": n_lat,
    "lon": n_lon,
}


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
<<<<<<< HEAD
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

    merged = merged.chunk(DEFAULT_CHUNKS_3D1)  # apply chunking for dask compatibility

    #hack to speed up calculation (cropped time range)
    merged = merged.sel(time=slice(config.DEFAULT_TIME_START, config.DEFAULT_TIME_END))
=======
    ds_sp = xr.open_dataset(filepath, chunks=DEFAULT_CHUNKS_2D1)
    ds_sp = ds_sp.rename({'latitude': 'lat', 'longitude': 'lon'})

    return ds_sp #[Pa]

# def load_era5_zg(filepath: str) -> xr.Dataset:
#     ds_z = xr.open_dataset(filepath)
#     ds_z = ds_z.rename({'latitude': 'lat', 'longitude': 'lon'})

#     ds_zg = ds_z * config.g  # convert geopotential to geopotential height
#     ds_zg = ds_zg.rename({'z': 'zg'})  # rename variable to 'zg' for consistency

#     return ds_zg  #[m]


def load_era5_merge_dataset(ds_T, ds_u, ds_v, ds_w, ds_sp) -> xr.Dataset:
    # Merge all datasets into a single dataset
    merged = xr.merge([ds_T, ds_u, ds_v, ds_w, ds_sp], 
                      compat='identical') 

    #Ensure dimensions are in correct order
    expected_order = ['time', 'level', 'lat', 'lon']
    merged = merged.transpose(*expected_order)

    #Enforce lat/lon monotonic ascending
    if not merged['lat'].diff('lat').min() > 0:
        merged = merged.sortby('lat')
    if not merged['lon'].diff('lon').min() > 0:
        merged = merged.sortby('lon')

    #Enforce pressure coordinates are monotonic decreasing
    dlev = merged['level'].diff('level')
    if not ((dlev < 0).all() and merged['level'].to_index().is_unique):
        merged = merged.sortby('level', ascending=False)

    merged['level']                = merged['level'] * 100.0  # convert hPa to Pa
    merged['level'].attrs['units'] = 'Pa'  # ensure pressure levels are in Pa
>>>>>>> main

    return merged


